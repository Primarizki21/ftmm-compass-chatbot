import math
import re
from typing import Any
import numpy as np


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.findall(r"\w+", text.lower()) if len(tok) > 1]


class BM25Okapi:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = np.array([len(tokenize(doc)) for doc in corpus], dtype=np.float32)
        self.avg_doc_len = float(np.mean(self.doc_lens)) if self.corpus_size > 0 else 1.0
        
        self.doc_freqs: dict[str, int] = {}
        self.doc_term_freqs: list[dict[str, int]] = []
        
        for doc in corpus:
            tokens = tokenize(doc)
            freqs: dict[str, int] = {}
            for tok in tokens:
                freqs[tok] = freqs.get(tok, 0) + 1
            self.doc_term_freqs.append(freqs)
            for tok in freqs:
                self.doc_freqs[tok] = self.doc_freqs.get(tok, 0) + 1
                
        self.idf: dict[str, float] = {}
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        query_tokens = tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for tok in query_tokens:
            if tok not in self.idf:
                continue
            idf_val = self.idf[tok]
            for doc_idx, freqs in enumerate(self.doc_term_freqs):
                tf = freqs.get(tok, 0)
                if tf > 0:
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[doc_idx] / self.avg_doc_len))
                    scores[doc_idx] += idf_val * ((tf * (self.k1 + 1.0)) / denom)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
