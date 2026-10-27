"""Simple offline TF-IDF retrieval.

No external model is required. This keeps the artifact reproducible for paper
figures and easy to replace with dense retrieval later.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .data import Document

TOKEN_RE = re.compile(r"[A-Za-z0-9_$!@.=-]+")


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class RetrievalResult:
    document: Document
    score: float


class TfidfRetriever:
    def __init__(self, documents: Sequence[Document]):
        self.documents = list(documents)
        self.doc_tokens = [tokenize(doc.title + " " + doc.content) for doc in self.documents]
        self.idf = self._compute_idf(self.doc_tokens)
        self.doc_vectors = [self._vectorize(tokens) for tokens in self.doc_tokens]
        self.doc_norms = [self._norm(vec) for vec in self.doc_vectors]

    @staticmethod
    def _compute_idf(token_lists: Iterable[List[str]]) -> Dict[str, float]:
        token_lists = list(token_lists)
        n_docs = len(token_lists)
        df: Dict[str, int] = defaultdict(int)
        for tokens in token_lists:
            for token in set(tokens):
                df[token] += 1
        return {token: math.log((1 + n_docs) / (1 + count)) + 1.0 for token, count in df.items()}

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        counts = Counter(tokens)
        if not counts:
            return {}
        max_tf = max(counts.values())
        return {tok: (count / max_tf) * self.idf.get(tok, 0.0) for tok, count in counts.items()}

    @staticmethod
    def _norm(vector: Dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values()))

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float], norm_a: float, norm_b: float) -> float:
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        common = set(a).intersection(b)
        return sum(a[tok] * b[tok] for tok in common) / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 4) -> List[RetrievalResult]:
        q_vec = self._vectorize(tokenize(query))
        q_norm = self._norm(q_vec)
        scored: List[Tuple[float, Document]] = []
        for doc, vec, norm in zip(self.documents, self.doc_vectors, self.doc_norms):
            score = self._cosine(q_vec, vec, q_norm, norm)
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [RetrievalResult(document=doc, score=score) for score, doc in scored[:top_k]]
