from __future__ import annotations

import streamlit as st
from typing import List, Dict, Optional

from retrieval_backend import (
    dense_retrieval,
    sparse_retrieval,
    hybrid_fusion,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_ndcg_at_k,
)


APP_TITLE = "Retrieval UI: Dense, Sparse (BM25), and Hybrid Fusion"
TOP_K = 5


def _short_preview(text: str, max_len: int = 160) -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "\u2026"


def _format_relevance(val: Optional[bool]) -> str:
    if val is True:
        return "Relevant"
    if val is False:
        return "Not Relevant"
    return "Unknown"


def render_header() -> None:
    st.title(APP_TITLE)
    st.caption(
        "Run dense, sparse (BM25), or hybrid fusion retrieval. Adjust fusion weights and evaluate top-5 results."
    )


def render_controls() -> None:
    if "alpha" not in st.session_state:
        st.session_state.alpha = 0.7
    if "beta" not in st.session_state:
        st.session_state.beta = 0.3

    with st.container():
        query = st.text_input("Enter your query", key="query", placeholder="e.g., fire tablet")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.alpha = st.slider(
                "Alpha (dense weight)", 0.0, 1.0, float(st.session_state.alpha), 0.05
            )
        with col2:
            st.session_state.beta = st.slider(
                "Beta (sparse weight)", 0.0, 1.0, float(st.session_state.beta), 0.05
            )

        st.caption("Alpha and Beta are used for Hybrid Fusion only.")

    return query


def _run_dense(query: str) -> List[Dict]:
    return dense_retrieval(query=query, top_k=TOP_K)


def _run_sparse(query: str) -> List[Dict]:
    return sparse_retrieval(query=query, top_k=TOP_K)


def _run_hybrid(query: str, alpha: float, beta: float) -> List[Dict]:
    return hybrid_fusion(query=query, alpha=alpha, beta=beta, top_k=TOP_K)


def _compute_and_render_metrics(results: List[Dict]) -> None:
    p5 = compute_precision_at_k(results, TOP_K)
    r5 = compute_recall_at_k(results, TOP_K)
    ndcg5 = compute_ndcg_at_k(results, TOP_K)

    def _fmt(x: Optional[float]) -> str:
        return f"{x:.3f}" if isinstance(x, float) else "N/A"

    st.subheader("Evaluation Metrics (@5)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Precision@5", _fmt(p5))
    col2.metric("Recall@5", _fmt(r5))
    col3.metric("nDCG@5", _fmt(ndcg5))


def _render_results(results: List[Dict]) -> None:
    if not results:
        st.info("No results.")
        return
    for i, r in enumerate(results, start=1):
        with st.container():
            st.markdown(f"**{i}. Doc:** `{r.get('doc_id')}`  ")
            st.write(_short_preview(str(r.get("content", ""))))
            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                st.write(f"Score: `{r.get('score')}`")
            with meta_col2:
                st.write(f"Relevance: `{_format_relevance(r.get('relevant'))}`")
        st.divider()


def render_action_buttons(query: str) -> None:
    col1, col2, col3 = st.columns(3)

    def _can_run() -> bool:
        if not query or not query.strip():
            st.warning("Please enter a query first.")
            return False
        return True

    with col1:
        if st.button("Run Dense-Only", use_container_width=True):
            if _can_run():
                with st.spinner("Running dense retrieval..."):
                    results = _run_dense(query)
                st.session_state.last_mode = "Dense"
                st.session_state.last_results = results

    with col2:
        if st.button("Run Sparse-Only (BM25)", use_container_width=True):
            if _can_run():
                with st.spinner("Running sparse (BM25) retrieval..."):
                    results = _run_sparse(query)
                st.session_state.last_mode = "Sparse"
                st.session_state.last_results = results

    with col3:
        if st.button("Run Hybrid Fusion", use_container_width=True):
            if _can_run():
                with st.spinner("Running hybrid fusion retrieval..."):
                    results = _run_hybrid(query, st.session_state.alpha, st.session_state.beta)
                st.session_state.last_mode = "Hybrid"
                st.session_state.last_results = results


def render_results_panel() -> None:
    results: List[Dict] = st.session_state.get("last_results", [])
    mode: Optional[str] = st.session_state.get("last_mode")
    if not results:
        return
    st.subheader(f"Top-{TOP_K} Results — {mode}")
    _render_results(results)
    _compute_and_render_metrics(results)


def init_state() -> None:
    if "last_results" not in st.session_state:
        st.session_state.last_results = []
    if "last_mode" not in st.session_state:
        st.session_state.last_mode = None


def main() -> None:
    init_state()
    render_header()
    query = render_controls()
    render_action_buttons(query)
    render_results_panel()


if __name__ == "__main__":
    main()


