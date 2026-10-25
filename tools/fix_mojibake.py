import argparse
import glob
import os
import shutil

COMMON = set("оеаинтсрвлкмдпуяыьгчзбйхжшюцщэфъё")


def score(text):
    cyr = [ch for ch in text if "\u0400" <= ch <= "\u04ff"]
    if len(cyr) < 8:
        return 1.0
    return sum(1 for ch in cyr if ch.islower()) / len(cyr)


def repair(text):
    try:
        return text.encode("koi8-r").decode("cp1251")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None


def repair_text(text):
    out = []
    changed = 0
    for line in text.splitlines(True):
        before = score(line)
        if before < 0.5 and len(line) > 6:
            fixed = repair(line)
            if fixed and score(fixed) > max(0.6, before + 0.2):
                out.append(fixed)
                changed += 1
                continue
        out.append(line)
    return "".join(out), changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--backup", default="data/raw_mojibake_backup")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.root, "**", "*.txt"), recursive=True)
    fixed = 0
    lines_total = 0
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        fixed_text, changed = repair_text(text)
        if changed < 3:
            continue
        rel = os.path.relpath(path, args.root)
        print(f"{rel}: lines {changed}")
        if args.apply:
            dst = os.path.join(args.backup, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copy2(path, dst)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(fixed_text)
        fixed += 1
        lines_total += changed
    print("repaired files", fixed, "lines", lines_total, "of", len(files))


if __name__ == "__main__":
    main()
