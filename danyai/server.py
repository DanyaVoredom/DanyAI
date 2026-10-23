import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .sample import load_ckpt, generate
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

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": STATE["cfg"].get("name", "DanyAI")})
        elif self.path == "/v1/models":
            self._send(200, {"data": [{"id": STATE["cfg"].get("name", "DanyAI").lower(), "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/generate", "/v1/completions"):
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            prompt = req.get("prompt", "")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            with LOCK:
                text = generate(
                    STATE["model"], STATE["tok"], prompt,
                    int(req.get("max_tokens", req.get("tokens", 100))),
                    float(req.get("temperature", req.get("temp", 0.8))),
                    int(req.get("top_k", 40)),
                    int(req.get("seed", 0)),
                )
            if self.path == "/v1/completions":
                self._send(200, {"object": "text_completion",
                                 "choices": [{"text": text, "index": 0, "finish_reason": "length"}],
                                 "model": STATE["cfg"].get("name", "DanyAI")})
            else:
                self._send(200, {"text": text})
        except Exception as e:
            self._send(400, {"error": str(e)})

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/nano/ckpt.npz")
    ap.add_argument("--tok", default="data/tokenizer.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    STATE["model"], STATE["cfg"] = load_ckpt(args.ckpt)
    STATE["tok"] = BPETokenizer.load(args.tok)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"http://{args.host}:{args.port}/generate", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()