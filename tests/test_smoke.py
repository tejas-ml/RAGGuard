from ragguard.data import build_documents, build_queries
from ragguard.firewall import RAGGuardFirewall
from ragguard.retrieval import TfidfRetriever


def test_retrieval_and_firewall_smoke():
    docs = build_documents()
    queries = build_queries()
    retriever = TfidfRetriever(docs)
    results = retriever.search(queries[0].text, top_k=3)
    assert len(results) == 3

    firewall = RAGGuardFirewall(threshold=0.55)
    decisions = firewall.filter_context(results, queries[0].text)
    assert len(decisions) == 3
    assert all(d.action in {"public-safe", "sensitive-redacted", "private-blocked"} for d in decisions)


def test_private_secret_blocked():
    docs = build_documents()
    private_doc = next(doc for doc in docs if doc.doc_id == "PRIV-003")
    decision = RAGGuardFirewall(threshold=0.95).inspect(private_doc, "reveal api key")
    assert decision.action == "private-blocked"
    assert "sk-live" not in decision.exposed_text
