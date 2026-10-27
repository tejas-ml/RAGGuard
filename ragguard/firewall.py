"""RAGGuard privacy firewall.

The firewall is deliberately interpretable: every decision is tied to pattern
features, metadata, and query-context features. That makes the method easy to
explain in a paper and easy to ablate in future work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .data import Document
from .retrieval import tokenize

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\d{3}[-.]?){2}\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
API_KEY_RE = re.compile(r"\b(?:sk-[A-Za-z0-9-]+|bearer_[A-Za-z0-9_]+|db_pass=[^\s,]+)\b")
MONEY_RE = re.compile(r"\$\d{4,}")
EQUITY_RE = re.compile(r"\bEQ-[A-Z0-9]+\b")

PROMPT_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"system override",
    r"developer message",
    r"reveal confidential",
    r"disclose private",
    r"bypass",
]

PRIVATE_TERMS = {
    "salary",
    "bonus",
    "ssn",
    "password",
    "api",
    "key",
    "token",
    "credential",
    "health",
    "migraine",
    "neurology",
    "accommodation",
    "personal",
    "private",
    "equity",
}

ACCESS_RISK = {
    "public": 0.0,
    "internal": 0.12,
    "confidential": 0.28,
    "private": 0.55,
}


@dataclass(frozen=True)
class FirewallDecision:
    doc_id: str
    title: str
    action: str
    sensitivity_score: float
    reasons: List[str]
    original_text: str
    exposed_text: str


class RAGGuardFirewall:
    """Classify retrieved chunks as safe, redactable, or blocked."""

    def __init__(self, threshold: float = 0.55, redact_margin: float = 0.30):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold
        self.redact_margin = redact_margin

    def inspect(self, document: Document, query: str) -> FirewallDecision:
        features = self._score_features(document, query)
        score = min(1.0, sum(features.values()))
        reasons = [name for name, value in features.items() if value > 0]

        lower = document.content.lower()
        has_injection = any(re.search(pattern, lower) for pattern in PROMPT_INJECTION_PATTERNS)
        hard_secret = bool(API_KEY_RE.search(document.content) or SSN_RE.search(document.content))
        private_level = document.access_level == "private"

        overblock = document.access_level in {"internal", "confidential"} and score >= min(1.0, self.threshold + 0.25)

        if has_injection or private_level or hard_secret or overblock:
            action = "private-blocked"
            exposed = "[BLOCKED: private or unsafe retrieved chunk removed by RAGGuard]"
        elif score >= max(0.0, self.threshold - self.redact_margin):
            action = "sensitive-redacted"
            exposed = self.redact(document.content)
        else:
            action = "public-safe"
            exposed = document.content

        return FirewallDecision(
            doc_id=document.doc_id,
            title=document.title,
            action=action,
            sensitivity_score=round(score, 4),
            reasons=reasons,
            original_text=document.content,
            exposed_text=exposed,
        )

    def filter_context(self, retrievals: Sequence, query: str) -> List[FirewallDecision]:
        return [self.inspect(item.document, query) for item in retrievals]

    def _score_features(self, document: Document, query: str) -> Dict[str, float]:
        text = document.content
        lower = text.lower()
        q_tokens = set(tokenize(query))
        doc_tokens = set(tokenize(text))
        sensitive_overlap = len((q_tokens | doc_tokens).intersection(PRIVATE_TERMS))

        pii_hits = 0
        for pattern in [EMAIL_RE, PHONE_RE, SSN_RE]:
            pii_hits += len(pattern.findall(text))

        secret_hits = len(API_KEY_RE.findall(text))
        money_hits = len(MONEY_RE.findall(text))
        equity_hits = len(EQUITY_RE.findall(text))
        injection_hits = sum(1 for pattern in PROMPT_INJECTION_PATTERNS if re.search(pattern, lower))

        private_term_hits = len(doc_tokens.intersection(PRIVATE_TERMS))
        query_risk_hits = len(q_tokens.intersection(PRIVATE_TERMS))

        return {
            "access_level": ACCESS_RISK.get(document.access_level, 0.1),
            "pii_pattern": min(0.28, 0.10 * pii_hits),
            "secret_pattern": min(0.42, 0.20 * secret_hits),
            "money_or_equity_pattern": min(0.24, 0.06 * (money_hits + equity_hits)),
            "prompt_injection_pattern": min(0.45, 0.22 * injection_hits),
            "private_term_density": min(0.22, 0.035 * private_term_hits),
            "query_sensitive_overlap": min(0.16, 0.025 * sensitive_overlap + 0.025 * query_risk_hits),
        }

    @staticmethod
    def redact(text: str) -> str:
        redacted = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
        redacted = PHONE_RE.sub("[REDACTED_PHONE]", redacted)
        redacted = SSN_RE.sub("[REDACTED_SSN]", redacted)
        redacted = API_KEY_RE.sub("[REDACTED_SECRET]", redacted)
        redacted = MONEY_RE.sub("[REDACTED_AMOUNT]", redacted)
        redacted = EQUITY_RE.sub("[REDACTED_EQUITY_ID]", redacted)
        redacted = re.sub(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", "[REDACTED_NAME]", redacted)
        return redacted
