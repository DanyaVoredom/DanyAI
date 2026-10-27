import argparse
import json
import queue
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from danyai.chatui import UI_HTML
except Exception:
    from chatui import UI_HTML

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>DanyAI</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;max-width:760px;margin:40px auto;padding:0 16px}
#log{white-space:pre-wrap;border:1px solid #333;border-radius:8px;padding:12px;min-height:320px;background:#181818}
textarea,input{width:100%;background:#181818;color:#eee;border:1px solid #333;border-radius:6px;padding:8px;box-sizing:border-box}
button{margin-top:8px;padding:8px 16px;border-radius:6px;border:1px solid #444;background:#222;color:#eee;cursor:pointer}
.row{display:flex;gap:8px}.row>*{flex:1}
</style></head><body>
<h3>DanyAI</h3>
<div id="log"></div>
<textarea id="prompt" rows="3" placeholder="сообщение"></textarea>
<div class="row">
<input id="temp" value="0.5"><input id="top_p" value="0.9"><input id="tokens" value="96">
</div>
<button onclick="go()">send</button>
<button onclick="reset()">reset</button>
<script>
const log=document.getElementById('log');
let turns=[];
function render(){
  log.innerHTML='';
  for(const t of turns){
    const d=document.createElement('div');
    d.textContent=(t.role==='user'?'you: ':'ai: ')+(t.content||'(пусто)');
    log.appendChild(d);
  }
  log.scrollTop=log.scrollHeight;
}
function reset(){turns=[];render();}
async function go(){
  const box=document.getElementById('prompt');
  const text=box.value.trim();
  if(!text)return;
  box.value='';
  turns.push({role:'user',content:text});
  render();
  let prompt='';
  for(const t of turns){
    prompt+=t.role==='user'?'<|user|>'+t.content+'<|assistant|>':t.content+'<|end|>';
  }
  const body={prompt:prompt,
    temperature:parseFloat(document.getElementById('temp').value),
    top_p:parseFloat(document.getElementById('top_p').value),
    max_tokens:parseInt(document.getElementById('tokens').value),repetition:1.15,stream:true};
  const r=await fetch('/v1/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok){turns.push({role:'ai',content:'error: '+await r.text()});render();return;}
  turns.push({role:'ai',content:''});
  const rd=r.body.getReader();const dec=new TextDecoder();
  let buf='';
  while(true){
    const {value,done}=await rd.read();
    if(done)break;
    buf+=dec.decode(value,{stream:true});
    const lines=buf.split(String.fromCharCode(10));
    buf=lines.pop();
    for(const line of lines){
      if(!line.startsWith('data: '))continue;
      const payload=line.slice(6);
      if(payload==='[DONE]')continue;
      let j;
      try{j=JSON.parse(payload);}catch(e){continue;}
      const piece=j.choices&&j.choices[0]&&j.choices[0].delta&&j.choices[0].delta.content;
      if(!piece)continue;
      turns[turns.length-1].content+=piece;
      if(turns[turns.length-1].content.includes('<|end|>'))break;
      render();
    }
  }
  const last=turns[turns.length-1];
  for(const stop of ['<|end|>','<|user|>','<|assistant|>']){
    const i=last.content.indexOf(stop);
    if(i>=0)last.content=last.content.slice(0,i);
  }
  render();
}
</script></body></html>
"""


def parse_turns(prompt):
    turns = []
    i = 0
    while True:
        u = prompt.find("<|user|>", i)
        if u < 0:
            break
        a = prompt.find("<|assistant|>", u)
        if a < 0:
            break
        user = prompt[u + 8 : a]
        nxt = prompt.find("<|user|>", a)
        seg = prompt[a + 13 : (nxt if nxt >= 0 else len(prompt))]
        ai = seg.split("<|end|>")[0]
        turns.append({"role": "user", "content": user})
        if ai:
            turns.append({"role": "assistant", "content": ai})
        i = nxt if nxt >= 0 else len(prompt)
    return turns


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, UI_HTML)
        elif self.path == "/api/meta":
            self._meta()
        else:
            self._send(404, "not found")

    def _meta(self):
        base = getattr(self.server, "api_base", None)
        model = getattr(self.server, "api_model", None) or "danyai-local"
        models = [model]
        if base:
            try:
                url = base.rstrip("/") + "/models"
                headers = {}
                key = getattr(self.server, "api_key", None)
                if key:
                    headers["Authorization"] = "Bearer " + key
                rq = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(rq, timeout=10) as r:
                    data = json.loads(r.read().decode("utf-8", "replace"))
                ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
                if ids:
                    models = ids
            except Exception:
                pass
        body = json.dumps({"proxy": bool(base), "model": model, "models": models})
        self._send(200, body, "application/json; charset=utf-8")

    def _chat(self, req):
        msgs = req.get("messages") or []
        turns = []
        for m in msgs:
            role = str(m.get("role") or "")
            text = str(m.get("content") or "")
            if role in ("system", "user", "assistant") and text:
                turns.append({"role": role, "content": text})
        if not turns:
            self._send(400, "empty", "text/plain; charset=utf-8")
            return
        base = getattr(self.server, "api_base", None)
        if not base:
            prompt = ""
            for t in turns:
                if t["role"] == "user":
                    prompt += "<|user|>" + t["content"]
                elif t["role"] == "assistant":
                    prompt += "<|assistant|>" + t["content"] + "<|end|>"
            prompt += "<|assistant|>"
            self._local(prompt, req)
            return
        body = {
            "messages": turns,
            "stream": True,
            "max_tokens": int(req.get("max_tokens", 512)),
            "temperature": float(req.get("temperature", 0.6)),
            "top_p": float(req.get("top_p", 0.95)),
        }
        model = req.get("model") or getattr(self.server, "api_model", None)
        if model:
            body["model"] = model
        headers = {"Content-Type": "application/json"}
        key = getattr(self.server, "api_key", None)
        if key:
            headers["Authorization"] = "Bearer " + key
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        rq = urllib.request.Request(
            base.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        self._relay(rq)
        self.close_connection = True

    def _local(self, prompt, req):
        q = self.server.jobs
        job = {
            "prompt": prompt,
            "tokens": int(req.get("max_tokens", 512)),
            "temp": float(req.get("temperature", 0.6)),
            "top_k": int(req.get("top_k", 40)),
            "top_p": float(req.get("top_p", 0.95)),
            "repetition": float(req.get("repetition", 1.05)),
            "presence": 0.0,
            "frequency": 0.0,
            "seed": 0,
            "done": threading.Event(),
        }
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        job["sink"] = lambda d: self._sse(
            {"choices": [{"delta": {"content": d}, "index": 0}]}
        )
        q.put(job)
        job["done"].wait()
        try:
            self._sse({"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True
        return

    def _proxy(self, req):
        turns = parse_turns(str(req.get("prompt", "")))
        body = {
            "messages": turns,
            "stream": True,
            "max_tokens": int(req.get("max_tokens", 100)),
            "temperature": float(req.get("temperature", 0.8)),
            "top_p": float(req.get("top_p", 0.95)),
        }
        model = getattr(self.server, "api_model", None)
        if model:
            body["model"] = model
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Connection", "close")
        self.end_headers()
        url = self.server.api_base.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        key = getattr(self.server, "api_key", None)
        if key:
            headers["Authorization"] = "Bearer " + key
        rq = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers=headers,
        )
        self._relay(rq)
        self.close_connection = True

    def _relay(self, rq):
        try:
            with urllib.request.urlopen(rq, timeout=900) as r:
                for raw in r:
                    self._feed(raw.decode("utf-8", "replace").strip())
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            self._fail(e)
        self._finish()

    def _feed(self, line):
        if not line.startswith("data:"):
            return
        payload = line[5:].strip()
        if not payload:
            return
        try:
            delta = json.loads(payload)["choices"][0]["delta"].get("content")
        except Exception:
            return
        if delta:
            self._sse({"choices": [{"delta": {"content": delta}, "index": 0}]})

    def _finish(self):
        try:
            self.wfile.write(b"data\n")
            self.wfile.flush()
        except Exception:
            pass

    def _fail(self, e):
        try:
            self._sse(
                {
                    "choices": [
                        {"delta": {"content": "backend: " + str(e)[:120]}, "index": 0}
                    ]
                }
            )
        except Exception:
            pass

    def _sse(self, obj):
        chunk = "data: " + json.dumps(obj, ensure_ascii=True) + "\n\n"
        self.wfile.write(chunk.encode("utf-8"))
        self.wfile.flush()

    def do_POST(self):
        if self.path == "/api/chat":
            n = int(self.headers.get("Content-Length", 0))
            self._chat(json.loads(self.rfile.read(n) or b"{}"))
            return
        if self.path != "/v1/completions":
            self._send(404, "not found")
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if getattr(self.server, "api_base", None):
            self._proxy(req)
            return
        q = self.server.jobs
        job = {
            "prompt": req.get("prompt", ""),
            "tokens": int(req.get("max_tokens", 100)),
            "temp": float(req.get("temperature", 0.8)),
            "top_k": int(req.get("top_k", 40)),
            "top_p": float(req.get("top_p", 0.95)),
            "repetition": float(req.get("repetition", 1.05)),
            "presence": float(req.get("presence", 0.0)),
            "frequency": float(req.get("frequency", 0.0)),
            "seed": int(req.get("seed", 0)),
            "done": threading.Event(),
        }
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        job["sink"] = lambda d: self._sse(
            {"choices": [{"delta": {"content": d}, "index": 0}]}
        )
        q.put(job)
        job["done"].wait()
        try:
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


def worker(engine, tok, q, lock):
    while True:
        job = q.get()
        if job is None:
            return
        sink = job.get("sink")

        def emit(delta, _n):
            if sink:
                try:
                    sink(delta)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        with lock:
            engine.generate(
                tok,
                job["prompt"],
                job["tokens"],
                job["temp"],
                job["top_k"],
                job["top_p"],
                job["repetition"],
                job["presence"],
                job["frequency"],
                job["seed"],
                on_token=emit,
            )
        job["done"].set()


def main():
    from .infer import Engine
    from .tokenizer import BPETokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/nano/ckpt.npz")
    ap.add_argument("--tok", default="data/tokenizer.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--api-base", default="")
    ap.add_argument("--api-model", default="")
    ap.add_argument("--api-key", default="")
    args = ap.parse_args()

    engine, _ = Engine.from_file(args.ckpt)
    tok = BPETokenizer.load(args.tok)
    q = queue.Queue()
    lock = threading.Lock()
    threading.Thread(target=worker, args=(engine, tok, q, lock), daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.jobs = q
    srv.api_base = args.api_base or None
    srv.api_model = args.api_model or None
    srv.api_key = args.api_key or None
    print(f"http://{args.host}:{args.port}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
