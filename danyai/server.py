import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .infer import Engine
from .tokenizer import BPETokenizer

STATE = {}
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, obj):
        chunk = "data: " + json.dumps(obj, ensure_ascii=True) + "\n\n"
        self.wfile.write(chunk.encode("utf-8"))
        self.wfile.flush()

    def do_GET(self):
        if self.path == "/health":
            self._send(
                200, {"status": "ok", "model": STATE["cfg"].get("name", "DanyAI")}
            )
        elif self.path == "/v1/models":
            self._send(
                200,
                {
                    "data": [
                        {
                            "id": STATE["cfg"].get("name", "DanyAI").lower(),
                            "object": "model",
                        }
                    ]
                },
            )
        else:
            self._send(404, {"error": "not found"})

    def _parse(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        prompt = req.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        if not prompt and isinstance(req.get("messages"), list):
            parts = []
            for m in req["messages"]:
                role = str(m.get("role", "user"))
                content = str(m.get("content", ""))
                if role == "assistant":
                    parts.append(content + "<|end|>")
                elif role in ("user", "system"):
                    parts.append(f"<|user|>{content}<|assistant|>")
            prompt = "".join(parts)
        return req, prompt

    @staticmethod
    def _kw(req):
        stop = req.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        elif not isinstance(stop, list):
            stop = None
        tokens = int(req.get("max_tokens", req.get("tokens", 100)))
        tokens = max(1, min(tokens, 2048))
        if not stop:
            stop = ["<|end|>", "<|user|>", "<|assistant|>"]
        return {
            "tokens": tokens,
            "temp": float(req.get("temperature", req.get("temp", 0.8))),
            "top_k": int(req.get("top_k", 50)),
            "top_p": float(req.get("top_p", 0.95)),
            "repetition": float(req.get("repetition", 1.15)),
            "presence": float(req.get("presence", 0.0)),
            "frequency": float(req.get("frequency", 0.0)),
            "seed": int(req.get("seed", 0)),
            "stop": [str(x) for x in stop],
        }

    def _allowed(self):
        key = STATE.get("key", "")
        if not key:
            return True
        auth = self.headers.get("Authorization", "")
        if auth == "Bearer " + key:
            return True
        return self.headers.get("X-Api-Key", "") == key

    def do_POST(self):
        if self.path not in ("/generate", "/v1/completions", "/v1/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        if not self._allowed():
            self._send(401, {"error": "unauthorized"})
            return
        try:
            req, prompt = self._parse()
            kw = self._kw(req)
            if req.get("stream"):
                self._stream(prompt, kw)
                return
            out = []
            with LOCK:
                STATE["engine"].generate(
                    STATE["tok"], prompt, on_token=lambda d, _n: out.append(d), **kw
                )
            self._reply("".join(out), prompt)
        except Exception as e:
            self._send(400, {"error": str(e)})

    def _reply(self, text, prompt=""):
        name = STATE["cfg"].get("name", "DanyAI")
        tok = STATE["tok"]
        usage = None
        if tok is not None:
            n_in = len(tok.encode(prompt))
            n_out = len(tok.encode(text))
            usage = {
                "prompt_tokens": n_in,
                "completion_tokens": n_out,
                "total_tokens": n_in + n_out,
            }
        if self.path == "/v1/chat/completions":
            body = {
                "object": "chat.completion",
                "model": name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "length",
                    }
                ],
            }
            if usage:
                body["usage"] = usage
            self._send(200, body)
        elif self.path == "/v1/completions":
            body = {
                "object": "text_completion",
                "choices": [{"text": text, "index": 0, "finish_reason": "length"}],
                "model": name,
            }
            if usage:
                body["usage"] = usage
            self._send(200, body)
        else:
            self._send(200, {"text": text})

    def _stream(self, prompt, kw):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(delta, _n):
            try:
                self._sse({"choices": [{"delta": {"content": delta}, "index": 0}]})
            except (BrokenPipeError, ConnectionResetError):
                pass

        try:
            with LOCK:
                STATE["engine"].generate(STATE["tok"], prompt, on_token=emit, **kw)
            self._sse(
                {"choices": [{"delta": {}, "finish_reason": "length", "index": 0}]}
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/nano/ckpt.npz")
    ap.add_argument("--tok", default="data/tokenizer.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key", default="")
    args = ap.parse_args()

    STATE["engine"], STATE["cfg"] = Engine.from_file(args.ckpt)
    STATE["tok"] = BPETokenizer.load(args.tok)
    STATE["key"] = args.api_key
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"http://{args.host}:{args.port}/generate", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
