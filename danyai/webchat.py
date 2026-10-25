import argparse
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
            self._send(200, PAGE)
        else:
            self._send(404, "not found")

    def _sse(self, obj):
        chunk = "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"
        self.wfile.write(chunk.encode("utf-8"))
        self.wfile.flush()

    def do_POST(self):
        if self.path != "/v1/completions":
            self._send(404, "not found")
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
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
        job["sink"] = lambda d: self._sse({"choices": [{"delta": {"content": d}, "index": 0}]})
        q.put(job)
        job["done"].wait()
        try:
            self._sse({"choices": [{"delta": {}, "finish_reason": "length", "index": 0}]})
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
            engine.generate(tok, job["prompt"], job["tokens"], job["temp"],
                            job["top_k"], job["top_p"], job["repetition"],
                            job["presence"], job["frequency"], job["seed"],
                            on_token=emit)
        job["done"].set()


def main():
    from .infer import Engine
    from .tokenizer import BPETokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/nano/ckpt.npz")
    ap.add_argument("--tok", default="data/tokenizer.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    engine, _ = Engine.from_file(args.ckpt)
    tok = BPETokenizer.load(args.tok)
    q = queue.Queue()
    lock = threading.Lock()
    threading.Thread(target=worker, args=(engine, tok, q, lock), daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.jobs = q
    print(f"http://{args.host}:{args.port}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()