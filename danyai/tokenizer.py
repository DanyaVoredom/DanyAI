import json
import re
from collections import Counter, defaultdict

PAT = re.compile(r"[^\W\d_]+|\d+|\s+|[^\s\w]", re.UNICODE)


def pretokenize(text):
    return PAT.findall(text)


class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.ranks = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._cache = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    def train(self, text, vocab_size, max_words=30000):
        words = Counter(pretokenize(text))
        if len(words) > max_words:
            words = Counter(dict(words.most_common(max_words)))
        seqs = {w: tuple(w.encode("utf-8")) for w in words}
        counts = {w: words[w] for w in seqs}

        pair_counts = Counter()
        pair_words = defaultdict(set)
        for w, ids in seqs.items():
            c = counts[w]
            for a, b in zip(ids, ids[1:]):
                pair_counts[(a, b)] += c
                pair_words[(a, b)].add(w)

        target = max(0, vocab_size - 256)
        for _ in range(target):
            if not pair_counts:
                break
            pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[pair] < 2:
                break
            new_id = 256 + len(self.merges)
            self.merges[pair] = new_id
            self.ranks[pair] = len(self.merges) - 1
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            affected = list(pair_words.get(pair, ()))
            for w in affected:
                old = seqs[w]
                c = counts[w]
                for a, b in zip(old, old[1:]):
                    pair_counts[(a, b)] -= c
                    if pair_counts[(a, b)] <= 0:
                        del pair_counts[(a, b)]
                    s = pair_words.get((a, b))
                    if s is not None:
                        s.discard(w)
                        if not s:
                            del pair_words[(a, b)]
                seqs[w] = self._apply(list(old), pair, new_id)
                new = seqs[w]
                for a, b in zip(new, new[1:]):
                    pair_counts[(a, b)] += c
                    pair_words[(a, b)].add(w)
            pair_words.pop(pair, None)

    @staticmethod
    def _apply(ids, pair, new_id):
        out = []
        i = 0
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def _encode_word(self, word):
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        b = list(word.encode("utf-8"))
        while len(b) >= 2:
            best = None
            for i in range(len(b) - 1):
                r = self.ranks.get((b[i], b[i + 1]))
                if r is not None and (best is None or r < best[0]):
                    best = (r, i)
            if best is None:
                break
            b = b[:best[1]] + [256 + best[0]] + b[best[1] + 2:]
        res = tuple(b)
        if len(self._cache) < 200000:
            self._cache[word] = res
        return res

    def encode(self, text):
        ids = []
        for w in pretokenize(text):
            ids.extend(self._encode_word(w))
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        data = {"merges": [[a, b, n] for (a, b), n in self.merges.items()]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        for a, b, n in data["merges"]:
            tok.merges[(a, b)] = n
            tok.ranks[(a, b)] = len(tok.ranks)
            tok.vocab[n] = tok.vocab[a] + tok.vocab[b]
        return tok