"""
Backend adapter for retrieval functions.

This module attempts to import existing retrieval functions from the current
environment. If they are not found, it provides mock implementations so the
Streamlit UI can run end-to-end.

Expected function signatures:
    dense_retrieval(query: str, top_k: int = 5) -> list[dict]
    sparse_retrieval(query: str, top_k: int = 5) -> list[dict]
    hybrid_fusion(query: str, alpha: float, beta: float, top_k: int = 5) -> list[dict]

Return format for each function:
    list of dicts, each with keys:
        - doc_id: str | int
        - content: str
        - score: float
        - relevant: Optional[bool]  (if available)
"""

from __future__ import annotations

from typing import Callable, List, Dict, Optional
import random


def _try_import(func_name: str) -> Optional[Callable]:
    """Try to import a retrieval function from common module names.

    Looks in the local package namespace first, then common names like
    `retrieval`, `backend`, or `models`.
    """
    module_candidates = [
        "retrieval",
        "backend",
        "models",
        "search",
        "pipeline",
        "app",
        "sample",  # user's existing file may contain the functions
    ]
    for module_name in module_candidates:
        try:
            module = __import__(module_name, fromlist=[func_name])
            func = getattr(module, func_name, None)
            if callable(func):
                return func
        except Exception:
            # Ignore import errors; continue trying other modules
            continue
    return None


def _generate_mock_docs(query: str, top_k: int) -> List[Dict]:
    random.seed(hash(query) % (2**32))
    results: List[Dict] = []
    for i in range(top_k):
        score = round(random.uniform(0.3, 1.0), 4)
        relevant = random.choice([True, False, None])
        results.append(
            {
                "doc_id": f"DOC-{i+1}",
                "content": f"Mock content for '{query}' — synthetic snippet number {i+1}. This is placeholder text to simulate retrieved document content.",
                "score": float(score),
                "relevant": relevant,
            }
        )
    # Sort by score descending to resemble real retrieval output
    results.sort(key=lambda d: d["score"], reverse=True)
    return results


def _mock_dense(query: str, top_k: int = 5) -> List[Dict]:
    return _generate_mock_docs(query, top_k)


def _mock_sparse(query: str, top_k: int = 5) -> List[Dict]:
    return _generate_mock_docs(query, top_k)


def _mock_hybrid(query: str, alpha: float, beta: float, top_k: int = 5) -> List[Dict]:
    # Produce deterministic yet distinct results using weights
    docs = _generate_mock_docs(query + f"|a={alpha:.2f}|b={beta:.2f}", top_k)
    return docs


# Bind to real implementations if present; otherwise use mocks
_dense_impl = _try_import("dense_retrieval") or _mock_dense
_sparse_impl = _try_import("sparse_retrieval") or _mock_sparse
_hybrid_impl = _try_import("hybrid_fusion") or _mock_hybrid


def dense_retrieval(query: str, top_k: int = 5) -> List[Dict]:
    return _dense_impl(query=query, top_k=top_k)


def sparse_retrieval(query: str, top_k: int = 5) -> List[Dict]:
    return _sparse_impl(query=query, top_k=top_k)


def hybrid_fusion(query: str, alpha: float, beta: float, top_k: int = 5) -> List[Dict]:
    return _hybrid_impl(query=query, alpha=alpha, beta=beta, top_k=top_k)


def compute_precision_at_k(results: List[Dict], k: int = 5) -> Optional[float]:
    labeled = [r for r in results[:k] if r.get("relevant") is not None]
    if not labeled:
        return None
    relevant_in_top_k = sum(1 for r in labeled if r.get("relevant") is True)
    return relevant_in_top_k / len(labeled)


def compute_recall_at_k(results: List[Dict], k: int = 5) -> Optional[float]:
    # Estimate recall within the scope of labeled items in the whole result set
    labeled_all = [r for r in results if r.get("relevant") is not None]
    if not labeled_all:
        return None
    total_relevant = sum(1 for r in labeled_all if r.get("relevant") is True)
    if total_relevant == 0:
        return None
    relevant_in_top_k = sum(1 for r in results[:k] if r.get("relevant") is True)
    return relevant_in_top_k / total_relevant


def compute_ndcg_at_k(results: List[Dict], k: int = 5) -> Optional[float]:
    # Binary relevance nDCG@k using provided relevance labels
    gains = []
    for r in results[:k]:
        rel = r.get("relevant")
        if rel is None:
            gains.append(None)
        else:
            gains.append(1.0 if rel else 0.0)

    # If no labels present in top-k, return None
    if all(g is None for g in gains):
        return None

    def dcg(vals):
        s = 0.0
        for i, v in enumerate(vals, start=1):
            if v is None:
                continue
            s += (2**v - 1) / (math.log2(i + 1))
        return s

    import math

    observed = dcg(gains)

    # Ideal DCG: sort known labels descending (unknowns ignored)
    known_labels = [g for g in gains if g is not None]
    if not known_labels:
        return None
    ideal_labels = sorted(known_labels, reverse=True) + []
    ideal = dcg(ideal_labels)
    if ideal == 0:
        return None
    return observed / ideal


