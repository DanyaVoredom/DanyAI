import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoTokenizer, TextIteratorStreamer

STATE = {}
LOCK = threading.Lock()


def pick_model_class():
    import transformers
    for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise RuntimeError("no suitable transformers model class")


def load(model_id, adapter, device, dtype):
    tok = AutoTokenizer.from_pretrained(adapter or model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    cls = pick_model_class()
    model = cls.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to(device)
    model.eval()
    return tok, model


def build_messages(req):
    msgs = req.get("messages")
    if isinstance(msgs, list) and msgs:
        out = []
        for m in msgs:
            role = str(m.get("role", "user"))
            content = m.get("content")
            if isinstance(content, list):
                content = "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
            item = {"role": role, "content": str(content or "")}
            if m.get("tool_calls"):
                item["tool_calls"] = m["tool_calls"]
            if m.get("name"):
                item["name"] = str(m["name"])
            out.append(item)
        return out
    prompt = req.get("prompt", "")
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    return [{"role": "user", "content": str(prompt)}]


def parse_tool_calls(text):
    import re
    calls = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text or "", re.S):
        raw = m.group(1)
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        name = obj.get("name") or (obj.get("function") or {}).get("name")
        args = obj.get("arguments")
        if args is None:
            args = (obj.get("function") or {}).get("arguments", {})
        if not name:
            continue
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        calls.append({"id": "call_%d" % (len(calls) + 1), "type": "function",
                      "function": {"name": name, "arguments": args}})
    return calls


def generate(tok, model, messages, params, streamer=None, tools=None):
    kwargs = {}
    if tools:
        kwargs["tools"] = tools
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    kw = {
        "max_new_tokens": int(params["max_tokens"]),
        "do_sample": params["temperature"] > 0,
        "temperature": max(1e-4, float(params["temperature"])),
        "top_p": float(params["top_p"]),
        "repetition_penalty": float(params["repetition"]),
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    if int(params.get("top_k", 0)) > 0:
        kw["top_k"] = int(params["top_k"])
    if streamer is not None:
        kw["streamer"] = streamer
        thread = threading.Thread(target=model.generate, kwargs=dict(input_ids=ids, **kw), daemon=True)
        thread.start()
        return thread
    with torch.no_grad():
        out = model.generate(input_ids=ids, **kw)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


INIT = {}


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
        self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8"))
        self.wfile.flush()

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": INIT["name"], "backend": "hf"})
        elif self.path == "/v1/models":
            self._send(200, {"data": [{"id": INIT["name"], "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def _allowed(self):
        key = INIT.get("key", "")
        if not key:
            return True
        auth = self.headers.get("Authorization", "")
        return auth == "Bearer " + key or self.headers.get("X-Api-Key", "") == key

    def _params(self, req):
        return {
            "max_tokens": int(req.get("max_tokens", req.get("tokens", 128))),
            "temperature": float(req.get("temperature", req.get("temp", 0.7))),
            "top_p": float(req.get("top_p", 0.9)),
            "top_k": int(req.get("top_k", 0)),
            "repetition": float(req.get("repetition_penalty", req.get("repetition", 1.05))),
        }

    def _reply(self, text):
        name = INIT["name"]
        if self.path == "/v1/chat/completions":
            calls = parse_tool_calls(text)
            if calls:
                message = {"role": "assistant", "content": "", "tool_calls": calls}
                finish = "tool_calls"
            else:
                message = {"role": "assistant", "content": text}
                finish = "stop"
            self._send(200, {"object": "chat.completion", "model": name,
                             "choices": [{"index": 0, "message": message, "finish_reason": finish}]})
        elif self.path == "/v1/completions":
            self._send(200, {"object": "text_completion", "model": name,
                             "choices": [{"text": text, "index": 0, "finish_reason": "length"}]})
        else:
            self._send(200, {"text": text})

    def do_POST(self):
        if self.path not in ("/generate", "/v1/completions", "/v1/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        if not self._allowed():
            self._send(401, {"error": "unauthorized"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            params = self._params(req)
            messages = build_messages(req)
            if req.get("stream"):
                self._stream(messages, params, tools=req.get("tools") or None)
                return
            with LOCK:
                text = generate(INIT["tok"], INIT["model"], messages, params,
                                tools=req.get("tools") or None)
            self._reply(text)
        except Exception as e:
            self._send(400, {"error": str(e)})

    def _stream(self, messages, params, tools=None):
        name = INIT["name"]
        chat = self.path == "/v1/chat/completions"
        streamer = TextIteratorStreamer(INIT["tok"], skip_prompt=True, skip_special_tokens=True)
        with LOCK:
            generate(INIT["tok"], INIT["model"], messages, params, streamer=streamer, tools=tools)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for piece in streamer:
                if not piece:
                    continue
                if chat:
                    chunk = {"object": "chat.completion.chunk", "model": name,
                             "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                else:
                    chunk = {"object": "text_completion", "model": name,
                             "choices": [{"index": 0, "text": piece, "finish_reason": None}]}
                self._sse(chunk)
            done = {"object": "chat.completion.chunk" if chat else "text_completion", "model": name,
                    "choices": [{"index": 0, "delta": {} if chat else None, "text": None if chat else "",
                                 "finish_reason": "stop"}]}
            self._sse(done)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key", default=os.environ.get("DANYAI_API_KEY", ""))
    ap.add_argument("--name", default="")
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (device == "cuda" and args.dtype == "float16") else torch.float32
    print(f"loading {args.model} adapter={args.adapter or '-'} device={device}", flush=True)
    tok, model = load(args.model, args.adapter, device, dtype)
    INIT["tok"] = tok
    INIT["model"] = model
    INIT["name"] = args.name or (args.adapter or args.model)
    INIT["key"] = args.api_key
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"listening on http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

