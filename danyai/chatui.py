UI_HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>DanyAI</title>
<style>
:root{
  --bg:#0f1115; --bg2:#151922; --line:#242a36; --fg:#e7ecf3; --dim:#8b95a7;
  --acc:#5b8cff; --acc2:#7aa2ff; --user:#1c2534; --ai:#141a24; --warn:#ff6b6b;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  display:flex;flex-direction:column;height:100dvh}
header{display:flex;align-items:center;gap:10px;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--bg2)}
.logo{font-weight:600;letter-spacing:.2px}
.logo span{color:var(--acc)}
.spacer{flex:1}
select,button,input,textarea{font:inherit;color:inherit}
select{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:6px 8px}
button{cursor:pointer;border:1px solid var(--line);background:var(--bg);border-radius:8px;padding:8px 14px}
button:hover{border-color:var(--acc)}
button.primary{background:var(--acc);border-color:var(--acc);color:#fff}
button.primary:disabled{opacity:.5;cursor:not-allowed}
#gear{background:none;border:none;font-size:18px;color:var(--dim)}
#gear:hover{color:var(--fg)}
main{flex:1;overflow-y:auto;padding:20px 16px 8px}
.wrap{max-width:820px;margin:0 auto;display:flex;flex-direction:column;gap:14px}
.msg{display:flex;gap:10px;align-items:flex-start}
.msg .avatar{width:28px;height:28px;border-radius:8px;flex:0 0 28px;display:grid;place-items:center;font-size:13px;
  background:var(--bg2);border:1px solid var(--line);color:var(--dim)}
.msg .body{background:var(--ai);border:1px solid var(--line);border-radius:12px;padding:10px 14px;max-width:100%;
  overflow-wrap:anywhere}
.msg.user{flex-direction:row-reverse}
.msg.user .body{background:var(--user);border-color:#2a3650}
.msg .body p{margin:0 0 8px}
.msg .body p:last-child{margin-bottom:0}
.msg .body pre{background:#0b0e13;border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto}
.msg .body code{background:#0b0e13;border:1px solid var(--line);border-radius:4px;padding:1px 4px;font-size:13px}
.msg .body pre code{border:0;padding:0;background:none}
.msg .body ul,.msg .body ol{margin:6px 0;padding-left:22px}
.msg .meta{font-size:11px;color:var(--dim);margin-top:6px;display:flex;gap:10px}
.msg .meta button{background:none;border:none;padding:0;color:var(--dim);font-size:11px}
.msg .meta button:hover{color:var(--acc)}
.caret{display:inline-block;width:7px;height:15px;background:var(--acc);vertical-align:-2px;animation:bl 1s steps(1) infinite}
@keyframes bl{50%{opacity:0}}
footer{border-top:1px solid var(--line);background:var(--bg2);padding:10px 16px calc(10px + env(safe-area-inset-bottom))}
.composer{max-width:820px;margin:0 auto;display:flex;gap:10px;align-items:flex-end}
.composer textarea{flex:1;resize:none;max-height:180px;min-height:44px;background:var(--bg);border:1px solid var(--line);
  border-radius:10px;padding:11px 12px;outline:none}
.composer textarea:focus{border-color:var(--acc)}
.hint{max-width:820px;margin:6px auto 0;font-size:11px;color:var(--dim);display:flex;gap:12px}
#panel{max-width:820px;margin:0 auto 10px;display:none;gap:14px;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:12px}
#panel.open{display:grid}
.field{display:flex;flex-direction:column;gap:4px;font-size:12px;color:var(--dim)}
.field b{color:var(--fg);font-weight:500}
.field input[type=range]{accent-color:var(--acc)}
.field input[type=number],.field textarea{background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:7px 9px;color:var(--fg)}
.field.wide{grid-column:1/-1}
.empty{color:var(--dim);text-align:center;padding:40px 0;font-size:14px}
#err{max-width:820px;margin:0 auto;color:var(--warn);font-size:12px;min-height:14px}
@media(max-width:600px){main{padding:14px 10px 4px}.composer{flex-direction:column}.composer button{width:100%}}
</style>
</head>
<body>
<header>
  <div class="logo">Dany<span>AI</span></div>
  <div class="spacer"></div>
  <select id="model"></select>
  <button id="gear" title="настройки">⚙</button>
</header>
<main><div class="wrap" id="chat"></div></main>
<footer>
  <div id="panel">
    <label class="field"><b>temperature</b>
      <input type="range" id="temp" min="0" max="1.5" step="0.05"><span id="tempv"></span>
    </label>
    <label class="field"><b>top_p</b>
      <input type="range" id="topp" min="0.05" max="1" step="0.05"><span id="toppv"></span>
    </label>
    <label class="field"><b>max tokens</b>
      <input type="number" id="tokens" min="16" max="4096" step="16">
    </label>
    <label class="field wide"><b>system prompt</b>
      <textarea id="sys" rows="2"></textarea>
    </label>
  </div>
  <div id="err"></div>
  <div class="composer">
    <textarea id="input" placeholder="сообщение" rows="1"></textarea>
    <button class="primary" id="send">отправить</button>
    <button id="stop" hidden>стоп</button>
    <button id="reset" title="очистить">сброс</button>
  </div>
  <div class="hint"><span>Enter — отправить</span><span>Shift+Enter — новая строка</span><span id="stat"></span></div>
</footer>
<script>
const $ = (id) => document.getElementById(id);
const chat = $('chat'), input = $('input'), sendBtn = $('send'), stopBtn = $('stop');
const DEF = {temp: 0.6, topp: 0.95, tokens: 512, sys: ''};
let cfg = Object.assign({}, DEF, JSON.parse(localStorage.getItem('danyai.cfg') || '{}'));
let history = JSON.parse(localStorage.getItem('danyai.hist') || '[]');
let models = [], ctrl = null;

function saveCfg(){ localStorage.setItem('danyai.cfg', JSON.stringify(cfg)); }
function saveHist(){ localStorage.setItem('danyai.hist', JSON.stringify(history.slice(-40))); }

function applyCfg(){
  $('temp').value = cfg.temp; $('tempv').textContent = cfg.temp;
  $('topp').value = cfg.topp; $('toppv').textContent = cfg.topp;
  $('tokens').value = cfg.tokens; $('sys').value = cfg.sys;
}
$('temp').oninput = e => { cfg.temp = +e.target.value; $('tempv').textContent = cfg.temp; saveCfg(); };
$('topp').oninput = e => { cfg.topp = +e.target.value; $('toppv').textContent = cfg.topp; saveCfg(); };
$('tokens').onchange = e => { cfg.tokens = Math.max(16, Math.min(4096, +e.target.value || 512)); e.target.value = cfg.tokens; saveCfg(); };
$('sys').oninput = e => { cfg.sys = e.target.value; saveCfg(); };
$('gear').onclick = () => $('panel').classList.toggle('open');

function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function md(s){
  let out = esc(s);
  out = out.replace(/```(\w*)\n?([\s\S]*?)```/g, (m,l,c) => '<pre><code>' + c.replace(/\n$/,'') + '</code></pre>');
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
  out = out.replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
  out = out.split(/\n{2,}/).map(p => /^<pre>/.test(p.trim()) ? p : '<p>' + p.replace(/\n/g,'<br>') + '</p>').join('');
  return out;
}

function bubble(role, text, meta){
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  el.innerHTML = '<div class="avatar">' + (role === 'user' ? 'я' : 'ai') + '</div>' +
                 '<div class="body"></div>';
  el.querySelector('.body').innerHTML = md(text);
  if (meta) {
    const m = document.createElement('div');
    m.className = 'meta'; m.textContent = meta;
    el.querySelector('.body').appendChild(m);
  }
  chat.appendChild(el);
  return el;
}

function render(){
  chat.innerHTML = '';
  if (!history.length){
    chat.innerHTML = '<div class="empty">пусто. напиши что-нибудь</div>';
    return;
  }
  for (const m of history) bubble(m.role === 'user' ? 'user' : 'ai', m.content);
  scrollDown();
}
function scrollDown(){ const m = document.querySelector('main'); m.scrollTop = m.scrollHeight; }

async function loadModels(){
  try {
    const r = await fetch('/api/meta'); const j = await r.json();
    models = j.models && j.models.length ? j.models : [j.model || 'model'];
    $('model').innerHTML = models.map(m => '<option>' + m + '</option>').join('');
    const saved = localStorage.getItem('danyai.model');
    if (saved && models.includes(saved)) $('model').value = saved;
    $('model').onchange = () => localStorage.setItem('danyai.model', $('model').value);
  } catch(e) {}
}

function setBusy(b){
  sendBtn.hidden = b; stopBtn.hidden = !b; input.disabled = b;
  if (!b) { input.focus(); }
}

async function ask(){
  const text = input.value.trim();
  if (!text || ctrl) return;
  $('err').textContent = '';
  input.value = ''; input.style.height = 'auto';
  history.push({role:'user', content:text});
  render(); saveHist();
  const holder = bubble('ai', '');
  const body = holder.querySelector('.body');
  const caret = document.createElement('span'); caret.className = 'caret';
  let acc = ''; const t0 = performance.now();
  ctrl = new AbortController();
  setBusy(true);
  try {
    const msgs = [];
    if (cfg.sys.trim()) msgs.push({role:'system', content: cfg.sys.trim()});
    for (const m of history) msgs.push(m);
    const r = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      signal: ctrl.signal,
      body: JSON.stringify({messages: msgs, model: $('model').value,
        temperature: cfg.temp, top_p: cfg.topp, max_tokens: cfg.tokens})
    });
    if (!r.ok) throw new Error('HTTP ' + r.status + ' ' + (await r.text()).slice(0,200));
    const rd = r.body.getReader(); const dec = new TextDecoder();
    let buf = '';
    body.innerHTML = ''; body.appendChild(caret);
    while (true){
      const {value, done} = await rd.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0){
        const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
        for (const line of chunk.split('\n')){
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (!payload || payload === '[DONE]') continue;
          try {
            const j = JSON.parse(payload);
            const d = j.choices && j.choices[0] && j.choices[0].delta;
            if (d && d.content){ acc += d.content; caret.remove(); body.innerHTML = md(acc); body.appendChild(caret); scrollDown(); }
          } catch(e) {}
        }
      }
    }
    caret.remove();
    const secs = (performance.now() - t0) / 1000;
    const chars = acc.length;
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = chars + ' симв · ' + secs.toFixed(1) + ' с · ' + (chars / Math.max(0.1, secs) / 3.3).toFixed(1) + ' ток/с';
    body.innerHTML = md(acc);
    body.appendChild(meta);
    history.push({role:'assistant', content: acc});
    saveHist(); scrollDown();
  } catch(e){
    caret.remove();
    if (e.name === 'AbortError'){
      body.innerHTML = md(acc || '');
    } else {
      $('err').textContent = 'ошибка: ' + e.message;
      body.innerHTML = md(acc || '');
    }
    if (acc) { history.push({role:'assistant', content: acc}); saveHist(); }
  } finally {
    ctrl = null; setBusy(false);
  }
}

sendBtn.onclick = ask;
stopBtn.onclick = () => { if (ctrl) ctrl.abort(); };
$('reset').onclick = () => { history = []; saveHist(); render(); };
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); ask(); }
});
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(180, input.scrollHeight) + 'px';
});

applyCfg(); loadModels(); render(); input.focus();
</script>
</body>
</html>"""
