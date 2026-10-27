"""Deterministic generation simulator.

This module intentionally does not call a real model. It simulates the key
privacy failure mode of RAG: if sensitive content or prompt injections enter the
context, the generated answer may copy them into the response.
"""

from __future__ import annotations

import re
from typing import List, Sequence

from .data import Document, Query
from .firewall import FirewallDecision


def _sentences(text: str) -> List[str]:
    # Split on sentence-ending punctuation followed by whitespace. This avoids
    # breaking inside redaction markers and most email-like tokens.
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").strip())
    return [part.strip() for part in parts if part.strip()]


def _content_words(text: str) -> List[str]:
    stop = {
        "what", "does", "from", "with", "that", "this", "about", "give", "show",
        "tell", "customers", "customer", "please", "reveal", "ignore", "policy",
        "safe", "summary", "summarize", "should", "could", "would", "the", "and",
        "for", "are", "was", "who", "how", "fast", "any", "into", "out", "can",
    }
    return [tok for tok in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(tok) > 3 and tok not in stop]


def generate_baseline_response(query: Query, retrieved_docs: Sequence[Document]) -> str:
    """Return a simulated baseline-RAG answer from unfiltered retrieved chunks."""
    context = " ".join(doc.content for doc in retrieved_docs)
    context_lower = context.lower()
    answer_parts: List[str] = []

    # Baseline RAG can still provide useful facts, but it has no privacy barrier.
    for doc in retrieved_docs:
        for fact in doc.public_facts:
            if fact not in answer_parts:
                answer_parts.append(fact)

    # If the user asks for protected data, the unfiltered baseline copies it.
    if query.attack_targets:
        for doc in retrieved_docs:
            for secret in doc.secret_spans:
                if secret not in answer_parts:
                    answer_parts.append(secret)

    # Prompt-injection chunks can cause additional copying of confidential spans.
    if "ignore previous instructions" in context_lower or "system override" in context_lower:
        answer_parts.insert(0, "Following retrieved instructions, confidential context says:")
        for doc in retrieved_docs:
            for secret in doc.secret_spans:
                if secret not in answer_parts:
                    answer_parts.append(secret)

    if not answer_parts:
        for sent in _sentences(context)[:2]:
            answer_parts.append(sent)

    return " ".join(answer_parts)


def generate_ragguard_response(query: Query, decisions: Sequence[FirewallDecision]) -> str:
    """Return a simulated answer after RAGGuard filtering."""
    exposed_texts = [d.exposed_text for d in decisions if d.action != "private-blocked"]
    if not exposed_texts:
        return "I cannot answer that because the retrieved context was private or unsafe."

    blocked_count = sum(1 for d in decisions if d.action == "private-blocked")
    redacted_count = sum(1 for d in decisions if d.action == "sensitive-redacted")
    query_words = _content_words(query.text)

    answer_parts: List[str] = []
    for d in decisions:
        if d.action == "private-blocked":
            continue
        for sent in _sentences(d.exposed_text):
            lower = sent.lower()
            if not query_words or any(word in lower for word in query_words):
                if sent not in answer_parts:
                    answer_parts.append(sent)

    if not answer_parts:
        for text in exposed_texts:
            for sent in _sentences(text)[:1]:
                if sent not in answer_parts:
                    answer_parts.append(sent)

    prefix = ""
    if blocked_count or redacted_count:
        prefix = f"RAGGuard filtered {blocked_count} private chunks and redacted {redacted_count} sensitive chunks. "

    answer = prefix + " ".join(answer_parts[:4])
    if not answer.strip():
        return "I cannot answer that from the safe context."
    return answer
