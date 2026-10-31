import sys

from gguf import GGUFReader

BASE = sys.argv[1] if len(sys.argv) > 1 else "/home/lipton/models/abliterated-q4km.gguf"
ADAPTER = sys.argv[2] if len(sys.argv) > 2 else "/home/lipton/lora_work/danyai-lora.gguf"

base = GGUFReader(BASE, "r")
base_names = {t.name for t in base.tensors}
adapter = GGUFReader(ADAPTER, "r")
targets = []
other = []
for t in adapter.tensors:
    n = t.name
    if ".lora_" in n:
        targets.append(n.rsplit(".lora_", 1)[0])
    else:
        other.append(n)

missing = sorted(set(targets) - base_names)
present = sorted(set(targets) & base_names)
print("base tensors", len(base_names), "adapter tensors", len(adapter.tensors))
print("targets", len(set(targets)), "present", len(present), "missing", len(missing))
if other:
    print("non lora tensors", len(other), other[:5])
if missing:
    print("missing sample", missing[:10])
print("sample present", present[:6])
print("arch", base.get_field("general.architecture").contents() if base.fields.get("general.architecture") else "")
