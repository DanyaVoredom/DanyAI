import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import danyai.hf_server as hf


class DummyStreamer:
    def __init__(self, *a, **k):
        self.items = []
        self.done = False

    def put(self, text):
        self.items.append(text)

    def end(self):
        self.done = True

    def __iter__(self):
        while True:
            if self.items:
                yield self.items.pop(0)
                continue
            if self.done:
                return
            time.sleep(0.01)


class DummyTok:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        class Out:
            input_ids = torch_ids()
        return Out()

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"


def torch_ids():
    class Ids:
        shape = (1, 3)

        def to(self, *_a):
            return self

        def __getitem__(self, item):
            return [2, 3, 4]

    return Ids()


class DummyModel:
    device = "cpu"

    def generate(self, input_ids=None, streamer=None, **kw):
        if streamer is not None:
            streamer.put("hello ")
            streamer.put("world")
            streamer.end()
            return None

        class Out:
            def __getitem__(self, item):
                return [5, 6]

        return Out()


@pytest.fixture(scope="module")
def server():
    hf.TextIteratorStreamer = DummyStreamer
    hf.INIT["tok"] = DummyTok()
    hf.INIT["model"] = DummyModel()
    hf.INIT["name"] = "dummy"
    hf.INIT["key"] = "secret"
    srv = ThreadingHTTPServer(("127.0.0.1", 0), hf.Handler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


def get(url, key="secret"):
    req = urllib.request.Request(url, headers={"X-Api-Key": key})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, json.loads(r.read())


def post(url, payload, key="secret"):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "X-Api-Key": key})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, r.read().decode()


def test_health(server):
    status, data = get(server + "/health")
    assert status == 200
    assert data["backend"] == "hf"
    assert data["model"] == "dummy"


def test_models(server):
    status, data = get(server + "/v1/models")
    assert status == 200
    assert data["data"][0]["id"] == "dummy"


def test_completion(server):
    status, body = post(server + "/v1/completions", {"prompt": "hi", "max_tokens": 4})
    assert status == 200
    data = json.loads(body)
    assert data["choices"][0]["text"] == "decoded"


def test_chat(server):
    status, body = post(server + "/v1/chat/completions",
                        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
    assert status == 200
    data = json.loads(body)
    assert data["choices"][0]["message"]["content"] == "decoded"


def test_stream(server):
    status, body = post(server + "/v1/completions", {"prompt": "hi", "stream": True, "max_tokens": 4})
    assert status == 200
    assert "data: " in body
    assert "hello " in body
    assert "[DONE]" in body


def test_unauthorized(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(server + "/v1/completions", {"prompt": "hi"}, key="wrong")
    assert exc.value.code == 401
