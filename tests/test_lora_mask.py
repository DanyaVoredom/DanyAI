import sys

sys.path.insert(0, r"C:\folders\projects\DanyAI\DanyAI")
sys.path.insert(0, r"C:\folders\projects\DanyAI\DanyAI\tools")

from finetune_lora import ChatData


class CharTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        out = ""
        for m in messages:
            out += f"<{m['role']}>" + m["content"] + f"</{m['role']}>"
        if add_generation_prompt:
            out += "<assistant>"
        return out

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_assistant_only_unmasked():
    rows = [{"messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]}]
    data = ChatData(rows, CharTok(), 4096)
    ids, labels = data[0]
    text = "".join(chr(i) for i in ids)
    kept = "".join(chr(i) for i, l in zip(ids, labels) if l != -100)
    assert "a</assistant>" in kept
    assert "q" not in kept.replace("<assistant>", "")[-len("a</assistant>"):]
    assert text.startswith("<system>s</system><user>q</user><assistant>a</assistant>")


def test_tool_turns_masked():
    rows = [{"messages": [
        {"role": "system", "content": "tools"},
        {"role": "user", "content": "weather"},
        {"role": "assistant", "content": "<tool_call>{}</tool_call>"},
        {"role": "tool", "content": "<tool_response>sun</tool_response>"},
        {"role": "assistant", "content": "sunny"},
    ]}]
    data = ChatData(rows, CharTok(), 4096)
    ids, labels = data[0]
    kept = "".join(chr(i) for i, l in zip(ids, labels) if l != -100)
    assert "<tool_call>{}</tool_call>" in kept
    assert "<tool_response>sun</tool_response>" not in kept
    assert "sunny" in kept


def test_plain_pairs():
    data = ChatData([{"user": "hi", "assistant": "hello"}], CharTok(), 4096)
    ids, labels = data[0]
    kept = "".join(chr(i) for i, l in zip(ids, labels) if l != -100)
    assert kept == "<assistant>hello</assistant>"


def test_mask_and_truncate():
    rows = [{"user": "x" * 50, "assistant": "y" * 50}]
    data = ChatData(rows, CharTok(), 120)
    ids, labels = data[0]
    assert len(ids) <= 120
    assert len(labels) == len(ids)
    assert any(l != -100 for l in labels)


def test_skip_empty_target():
    data = ChatData([{"user": "hi", "assistant": ""}], CharTok(), 64)
    assert len(data) == 0
