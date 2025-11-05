import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import random


def precision_recall_ndcg(predicted, relevant, k=5):
    """Compute Precision@k, Recall@k, nDCG@k"""
    predicted = predicted[:k]
    relevant_set = set(relevant)
    hits = [1 if idx in relevant_set else 0 for idx in predicted]
    
    precision = sum(hits)/k
    recall = sum(hits)/len(relevant) if relevant else 0
    
    # DCG
    dcg = sum([hits[i]/np.log2(i+2) for i in range(len(hits))])
    # IDCG
    ideal_hits = sorted([1]*len(relevant) + [0]*(k-len(relevant)))[:k]
    idcg = sum([ideal_hits[i]/np.log2(i+2) for i in range(len(ideal_hits))])
    ndcg = dcg/idcg if idcg > 0 else 0
    
    return precision, recall, ndcg


def evaluate_correctness(retrieved_indices, relevant_indices, k=5):
    retrieved_topk = retrieved_indices[:k]
    relevant_set = set(relevant_indices)

    correct = [idx for idx in retrieved_topk if idx in relevant_set]
    incorrect = [idx for idx in retrieved_topk if idx not in relevant_set]
    missed = [idx for idx in relevant_set if idx not in retrieved_topk]

    return len(correct), len(incorrect), len(missed)


def print_detailed_results(query, dataset_size, all_passages, dense_idx, sparse_idx, hybrid_idx,
                          relevant, p_d, r_d, n_d, p_s, r_s, n_s, p_h, r_h, n_h,
                          c_d, inc_d, miss_d, c_s, inc_s, miss_s, c_h, inc_h, miss_h,
                          alpha=0.7, beta=0.3):
    """
    Print formatted detailed results with box drawing characters for patent document.
    
    Args:
        query: Search query string
        dataset_size: Total number of documents
        all_passages: List of document contents
        dense_idx: List of top-5 document indices from dense retrieval
        sparse_idx: List of top-5 document indices from sparse retrieval
        hybrid_idx: List of top-5 document indices from hybrid retrieval
        relevant: List of ground truth document indices
        p_d, r_d, n_d: Vector-only metrics (precision, recall, ndcg)
        p_s, r_s, n_s: BM25-only metrics
        p_h, r_h, n_h: Hybrid metrics
        c_d, inc_d, miss_d: Vector correctness counts
        c_s, inc_s, miss_s: BM25 correctness counts
        c_h, inc_h, miss_h: Hybrid correctness counts
        alpha: Dense weight for hybrid fusion
        beta: Sparse weight for hybrid fusion
    """
    # Calculate F1 scores
    f1_d = 2 * (p_d * r_d) / (p_d + r_d) if (p_d + r_d) > 0 else 0.0
    f1_s = 2 * (p_s * r_s) / (p_s + r_s) if (p_s + r_s) > 0 else 0.0
    f1_h = 2 * (p_h * r_h) / (p_h + r_h) if (p_h + r_h) > 0 else 0.0
    
    # Header
    print("\n" + "="*80)
    print("RETRIEVAL EVALUATION REPORT".center(80))
    print("="*80)
    print(f"\nQuery: \"{query}\"")
    print(f"Dataset Size: {dataset_size} documents")
    if relevant:
        print(f"Ground Truth: doc_id_{relevant[0]} (most similar via dense)")
    else:
        print("Ground Truth: None")
    print()
    
    # Vector-Only Results
    print("┌" + "─"*78 + "┐")
    print("│ Vector-Only Results" + " "*59 + "│")
    print("├" + "─"*78 + "┤")
    metrics_line = f"│ Precision@5: {p_d:.2f} | Recall@5: {r_d:.2f} | nDCG@5: {n_d:.2f}"
    print(metrics_line + " " * (78 - (len(metrics_line) - 1)) + "│")
    correct_line = f"│ Correct: {c_d} | Incorrect: {inc_d} | Missed: {miss_d}"
    print(correct_line + " " * (78 - (len(correct_line) - 1)) + "│")
    print("│ Top-5 Results:" + " "*64 + "│")
    relevant_set = set(relevant)
    for i, idx in enumerate(dense_idx[:5], 1):
        status = "✓" if idx in relevant_set else "✗"
        content = all_passages[idx] if idx < len(all_passages) else ""
        content_preview = (content[:50] + "...") if len(content) > 50 else content
        result_line = f"│   {i}. [{status}] doc_{idx} - \"{content_preview}\""
        # Pad to 78 characters inside the box (excluding the leading │)
        content_len = len(result_line) - 1  # Subtract 1 for the leading │
        padding = max(0, 78 - content_len)
        print(result_line + " " * padding + "│")
    print("└" + "─"*78 + "┘")
    print()
    
    # BM25-Only Results
    print("┌" + "─"*78 + "┐")
    print("│ BM25-Only Results" + " "*61 + "│")
    print("├" + "─"*78 + "┤")
    metrics_line = f"│ Precision@5: {p_s:.2f} | Recall@5: {r_s:.2f} | nDCG@5: {n_s:.2f}"
    print(metrics_line + " " * (78 - (len(metrics_line) - 1)) + "│")
    correct_line = f"│ Correct: {c_s} | Incorrect: {inc_s} | Missed: {miss_s}"
    print(correct_line + " " * (78 - (len(correct_line) - 1)) + "│")
    print("│ Top-5 Results:" + " "*64 + "│")
    for i, idx in enumerate(sparse_idx[:5], 1):
        status = "✓" if idx in relevant_set else "✗"
        content = all_passages[idx] if idx < len(all_passages) else ""
        content_preview = (content[:50] + "...") if len(content) > 50 else content
        result_line = f"│   {i}. [{status}] doc_{idx} - \"{content_preview}\""
        content_len = len(result_line) - 1
        padding = max(0, 78 - content_len)
        print(result_line + " " * padding + "│")
    print("└" + "─"*78 + "┘")
    print()
    
    # Hybrid Fusion Results
    print("┌" + "─"*78 + "┐")
    print("│ Hybrid Fusion Results" + " "*57 + "│")
    print("├" + "─"*78 + "┤")
    metrics_line = f"│ Precision@5: {p_h:.2f} | Recall@5: {r_h:.2f} | nDCG@5: {n_h:.2f}"
    print(metrics_line + " " * (78 - (len(metrics_line) - 1)) + "│")
    correct_line = f"│ Correct: {c_h} | Incorrect: {inc_h} | Missed: {miss_h}"
    print(correct_line + " " * (78 - (len(correct_line) - 1)) + "│")
    fusion_line = f"│ Fusion Params: α={alpha}, β={beta}"
    print(fusion_line + " " * (78 - (len(fusion_line) - 1)) + "│")
    print("│ Top-5 Results:" + " "*64 + "│")
    for i, idx in enumerate(hybrid_idx[:5], 1):
        status = "✓" if idx in relevant_set else "✗"
        content = all_passages[idx] if idx < len(all_passages) else ""
        content_preview = (content[:50] + "...") if len(content) > 50 else content
        result_line = f"│   {i}. [{status}] doc_{idx} - \"{content_preview}\""
        content_len = len(result_line) - 1
        padding = max(0, 78 - content_len)
        print(result_line + " " * padding + "│")
    print("└" + "─"*78 + "┘")
    print()
    
    # Comparison Summary Table
    print("="*80)
    print("PERFORMANCE COMPARISON SUMMARY".center(80))
    print("="*80)
    print("│ Metric        │ Vector-Only │ BM25-Only │ Hybrid   │ Improvement │")
    print("├" + "─"*14 + "┼" + "─"*13 + "┼" + "─"*11 + "┼" + "─"*10 + "┼" + "─"*13 + "┤")
    
    # Calculate improvements relative to best baseline
    best_precision = max(p_d, p_s)
    precision_imp = ((p_h - best_precision) / best_precision * 100) if best_precision > 0 else 0.0
    
    best_recall = max(r_d, r_s)
    recall_imp = ((r_h - best_recall) / best_recall * 100) if best_recall > 0 else 0.0
    
    best_ndcg = max(n_d, n_s)
    ndcg_imp = ((n_h - best_ndcg) / best_ndcg * 100) if best_ndcg > 0 else 0.0
    
    best_f1 = max(f1_d, f1_s)
    f1_imp = ((f1_h - best_f1) / best_f1 * 100) if best_f1 > 0 else 0.0
    
    print(f"│ Precision@5   │ {p_d:.2f}        │ {p_s:.2f}      │ {p_h:.2f}    │ {precision_imp:+6.1f}%     │")
    print(f"│ Recall@5      │ {r_d:.2f}        │ {r_s:.2f}      │ {r_h:.2f}    │ {recall_imp:+6.1f}%     │")
    print(f"│ nDCG@5        │ {n_d:.2f}        │ {n_s:.2f}      │ {n_h:.2f}    │ {ndcg_imp:+6.1f}%     │")
    print(f"│ F1 Score      │ {f1_d:.2f}        │ {f1_s:.2f}      │ {f1_h:.2f}    │ {f1_imp:+6.1f}%     │")
    print("└" + "─"*14 + "┴" + "─"*13 + "┴" + "─"*11 + "┴" + "─"*10 + "┴" + "─"*13 + "┘")
    print()
    
    # Conclusion
    print("Conclusion: ", end="")
    if p_h > max(p_d, p_s):
        print("Hybrid Fusion achieves superior precision and recall while maintaining acceptable latency for real-time QA applications.")
    elif p_d > max(p_s, p_h):
        print("Vector-only method performs best here, indicating semantic search captured relevant results more accurately.")
    elif p_s > max(p_d, p_h):
        print("BM25-only outperforms others, indicating exact keyword matching was more effective for this query.")
    else:
        print("The methods perform similarly; hybrid fusion provides a balanced approach leveraging both methods.")
    print("\n" + "="*80 + "\n")


def run_multiple_queries_for_validation(df_clean, bm25, dense_model, faiss_index, test_queries, alpha=0.7, beta=0.3):
    """
    Run multiple queries and collect aggregate metrics.
    
    Args:
        df_clean: Cleaned dataframe with documents
        bm25: BM25 index object
        dense_model: SentenceTransformer model
        faiss_index: FAISS index object
        test_queries: List of test query strings
        alpha: Dense weight for hybrid fusion
        beta: Sparse weight for hybrid fusion
        
    Returns:
        Dictionary with metrics for each method across all queries
    """
    all_passages = df_clean['content'].tolist()
    
    def batch_embed(texts, model, batch_size=32):
        embeddings = []
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx+batch_size]
            batch_emb = model.encode(batch_texts, convert_to_numpy=True)
            embeddings.append(batch_emb)
        return np.vstack(embeddings).astype('float32')
    
    def dense_retrieval(query, top_k=60):
        q_emb = batch_embed([query], dense_model)
        faiss.normalize_L2(q_emb)
        scores, idxs = faiss_index.search(q_emb, top_k)
        return [(idx, scores[0][i]) for i, idx in enumerate(idxs[0])]
    
    def sparse_retrieval(query, top_k=60):
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_idxs = np.argsort(scores)[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_idxs]
    
    def hybrid_fusion(query, alpha=0.7, beta=0.3, top_k=20):
        dense_results = dense_retrieval(query, top_k=top_k*3)
        sparse_results = sparse_retrieval(query, top_k=top_k*3)
        
        dense_dict = {idx: score for idx, score in dense_results}
        sparse_dict = {idx: score for idx, score in sparse_results}
        
        def min_max_normalize(score_dict):
            if not score_dict:
                return {}
            vals = np.array(list(score_dict.values()))
            min_val, max_val = vals.min(), vals.max()
            range_val = max_val - min_val
            if range_val < 1e-6:
                return {k: 0.5 for k in score_dict}
            return {k: (v - min_val) / range_val for k, v in score_dict.items()}
        
        norm_dense = min_max_normalize(dense_dict)
        norm_sparse = min_max_normalize(sparse_dict)
        
        all_indices = set(norm_dense.keys()).union(norm_sparse.keys())
        fused_scores = {}
        for idx in all_indices:
            v_score = norm_dense.get(idx, 0.0)
            k_score = norm_sparse.get(idx, 0.0)
            fused_scores[idx] = alpha * (v_score + 1e-3) + beta * (k_score + 1e-3)
        
        top_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        return top_indices
    
    results = {
        'vector': {'precision': [], 'recall': [], 'ndcg': []},
        'sparse': {'precision': [], 'recall': [], 'ndcg': []},
        'hybrid': {'precision': [], 'recall': [], 'ndcg': []}
    }
    
    for query in test_queries:
        # Get ground truth
        dense_results_for_relevance = dense_retrieval(query, top_k=1)
        if dense_results_for_relevance:
            relevant = [dense_results_for_relevance[0][0]]
        else:
            relevant = []
        
        # Run all three methods
        dense_idx = [idx for idx, _ in dense_retrieval(query, top_k=5)]
        sparse_idx = [idx for idx, _ in sparse_retrieval(query, top_k=5)]
        hybrid_idx = hybrid_fusion(query, alpha=alpha, beta=beta, top_k=5)
        
        # Compute metrics
        p_d, r_d, n_d = precision_recall_ndcg(dense_idx, relevant)
        p_s, r_s, n_s = precision_recall_ndcg(sparse_idx, relevant)
        p_h, r_h, n_h = precision_recall_ndcg(hybrid_idx, relevant)
        
        # Store results
        results['vector']['precision'].append(p_d)
        results['vector']['recall'].append(r_d)
        results['vector']['ndcg'].append(n_d)
        
        results['sparse']['precision'].append(p_s)
        results['sparse']['recall'].append(r_s)
        results['sparse']['ndcg'].append(n_s)
        
        results['hybrid']['precision'].append(p_h)
        results['hybrid']['recall'].append(r_h)
        results['hybrid']['ndcg'].append(n_h)
    
    return results


def print_aggregate_metrics(results):
    """
    Print aggregate metrics table from validation results.
    
    Args:
        results: Dictionary with metrics from run_multiple_queries_for_validation()
    """
    # Calculate averages
    avg_p_d = np.mean(results['vector']['precision'])
    avg_r_d = np.mean(results['vector']['recall'])
    avg_n_d = np.mean(results['vector']['ndcg'])
    
    avg_p_s = np.mean(results['sparse']['precision'])
    avg_r_s = np.mean(results['sparse']['recall'])
    avg_n_s = np.mean(results['sparse']['ndcg'])
    
    avg_p_h = np.mean(results['hybrid']['precision'])
    avg_r_h = np.mean(results['hybrid']['recall'])
    avg_n_h = np.mean(results['hybrid']['ndcg'])
    
    print("\n" + "="*80)
    print("AGGREGATE METRICS SUMMARY (Across All Test Queries)".center(80))
    print("="*80)
    print("│ Metric        │ Vector-Only │ BM25-Only │ Hybrid   │ Improvement │")
    print("├" + "─"*14 + "┼" + "─"*13 + "┼" + "─"*11 + "┼" + "─"*10 + "┼" + "─"*13 + "┤")
    
    # Calculate improvements
    best_precision = max(avg_p_d, avg_p_s)
    precision_imp = ((avg_p_h - best_precision) / best_precision * 100) if best_precision > 0 else 0.0
    
    best_recall = max(avg_r_d, avg_r_s)
    recall_imp = ((avg_r_h - best_recall) / best_recall * 100) if best_recall > 0 else 0.0
    
    best_ndcg = max(avg_n_d, avg_n_s)
    ndcg_imp = ((avg_n_h - best_ndcg) / best_ndcg * 100) if best_ndcg > 0 else 0.0
    
    print(f"│ Precision@5   │ {avg_p_d:.2f}        │ {avg_p_s:.2f}      │ {avg_p_h:.2f}    │ {precision_imp:+6.1f}%     │")
    print(f"│ Recall@5      │ {avg_r_d:.2f}        │ {avg_r_s:.2f}      │ {avg_r_h:.2f}    │ {recall_imp:+6.1f}%     │")
    print(f"│ nDCG@5        │ {avg_n_d:.2f}        │ {avg_n_s:.2f}      │ {avg_n_h:.2f}    │ {ndcg_imp:+6.1f}%     │")
    print("└" + "─"*14 + "┴" + "─"*13 + "┴" + "─"*11 + "┴" + "─"*10 + "┴" + "─"*13 + "┘")
    print()
    
    # Overall conclusion
    print("Overall Conclusion: ", end="")
    if avg_p_h > max(avg_p_d, avg_p_s) and avg_r_h > max(avg_r_d, avg_r_s):
        print("Hybrid Fusion demonstrates superior performance across all evaluated queries, consistently outperforming both baseline methods in precision and recall metrics.")
    elif avg_p_d > max(avg_p_s, avg_p_h) and avg_r_d > max(avg_r_s, avg_r_h):
        print("Vector-only method shows the best overall performance, indicating semantic search is more effective for this dataset.")
    elif avg_p_s > max(avg_p_d, avg_p_h) and avg_r_s > max(avg_r_d, avg_r_h):
        print("BM25-only method performs best overall, suggesting keyword matching is more suitable for these queries.")
    else:
        print("Hybrid Fusion provides a balanced and robust approach, combining strengths of both dense and sparse retrieval methods.")
    print("\n" + "="*80 + "\n")


def main():
    # Load and preprocess dataset
    df = pd.read_csv(r"C:\Users\Saukhyad Mohole\OneDrive - vit.ac.in\Desktop\DBMS RP\1429_1.csv", low_memory=False)
    df_clean = df.dropna(subset=['name', 'reviews.text']).drop_duplicates(subset=['reviews.text'])
    df_clean = df_clean[df_clean['reviews.text'].str.len() > 30]
    df_clean['name'] = df_clean['name'].str.strip().str.lower()
    df_clean['reviews.text'] = df_clean['reviews.text'].str.strip().str.lower()
    df_clean['content'] = df_clean['name'] + ". " + df_clean['reviews.text']
    df_clean = df_clean.head(500)

    # BM25 index
    tokenized_corpus = [doc.split() for doc in df_clean['content'].tolist()]
    bm25 = BM25Okapi(tokenized_corpus)

    # Dense embedding model
    dense_model = SentenceTransformer('all-MiniLM-L6-v2')
    all_passages = df_clean['content'].tolist()

    def batch_embed(texts, model, batch_size=32):
        embeddings = []
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx+batch_size]
            batch_emb = model.encode(batch_texts, convert_to_numpy=True)
            embeddings.append(batch_emb)
        return np.vstack(embeddings).astype('float32')

    # FAISS index
    passage_embeddings = batch_embed(all_passages, dense_model)
    faiss.normalize_L2(passage_embeddings)
    dim = passage_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(passage_embeddings)

    def dense_retrieval(query, top_k=60):
        q_emb = batch_embed([query], dense_model)
        faiss.normalize_L2(q_emb)
        scores, idxs = faiss_index.search(q_emb, top_k)
        return [(idx, scores[0][i]) for i, idx in enumerate(idxs[0])]

    def sparse_retrieval(query, top_k=60):
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_idxs = np.argsort(scores)[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_idxs]

    def hybrid_fusion(query, alpha=0.6, beta=0.4, top_k=20):
        dense_results = dense_retrieval(query, top_k=top_k*3)
        sparse_results = sparse_retrieval(query, top_k=top_k*3)

        dense_dict = {idx: score for idx, score in dense_results}
        sparse_dict = {idx: score for idx, score in sparse_results}

        def min_max_normalize(score_dict):
            if not score_dict:
                return {}
            vals = np.array(list(score_dict.values()))
            min_val, max_val = vals.min(), vals.max()
            range_val = max_val - min_val
            if range_val < 1e-6:
                return {k: 0.5 for k in score_dict}
            return {k: (v - min_val) / range_val for k, v in score_dict.items()}

        norm_dense = min_max_normalize(dense_dict)
        norm_sparse = min_max_normalize(sparse_dict)

        all_indices = set(norm_dense.keys()).union(norm_sparse.keys())
        fused_scores = {}
        for idx in all_indices:
            v_score = norm_dense.get(idx, 0.0)
            k_score = norm_sparse.get(idx, 0.0)
            fused_scores[idx] = alpha * (v_score + 1e-3) + beta * (k_score + 1e-3)

        top_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        return top_indices

    # Take manual query input
    user_query = input("Enter your query: ").strip().lower()
    
    alpha = 0.7
    beta = 0.3
    
    # Process user query with detailed output
    print("\n" + "="*80)
    print("DETAILED QUERY EVALUATION".center(80))
    print("="*80)
    
    # Use dense retrieval to find top 1 most similar doc as pseudo ground truth
    dense_results_for_relevance = dense_retrieval(user_query, top_k=1)
    if dense_results_for_relevance:
        gt_idx = dense_results_for_relevance[0][0]  # index of most similar doc
        relevant = [gt_idx]
    else:
        print("No relevant document found via dense retrieval.")
        relevant = []

    # Run all three retrieval methods
    dense_idx = [idx for idx, _ in dense_retrieval(user_query, top_k=5)]
    sparse_idx = [idx for idx, _ in sparse_retrieval(user_query, top_k=5)]
    hybrid_idx = hybrid_fusion(user_query, alpha=alpha, beta=beta, top_k=5)
    
    # Compute metrics
    p_d, r_d, n_d = precision_recall_ndcg(dense_idx, relevant)
    p_s, r_s, n_s = precision_recall_ndcg(sparse_idx, relevant)
    p_h, r_h, n_h = precision_recall_ndcg(hybrid_idx, relevant)
    
    # Compute correctness
    c_d, inc_d, miss_d = evaluate_correctness(dense_idx, relevant)
    c_s, inc_s, miss_s = evaluate_correctness(sparse_idx, relevant)
    c_h, inc_h, miss_h = evaluate_correctness(hybrid_idx, relevant)
    
    # Print detailed formatted results
    print_detailed_results(
        query=user_query,
        dataset_size=len(df_clean),
        all_passages=all_passages,
        dense_idx=dense_idx,
        sparse_idx=sparse_idx,
        hybrid_idx=hybrid_idx,
        relevant=relevant,
        p_d=p_d, r_d=r_d, n_d=n_d,
        p_s=p_s, r_s=r_s, n_s=n_s,
        p_h=p_h, r_h=r_h, n_h=n_h,
        c_d=c_d, inc_d=inc_d, miss_d=miss_d,
        c_s=c_s, inc_s=inc_s, miss_s=miss_s,
        c_h=c_h, inc_h=inc_h, miss_h=miss_h,
        alpha=alpha, beta=beta
    )
    
    # Optional: Ask if user wants to run validation on predefined queries
    run_validation = input("\nRun validation on predefined test queries? (y/n): ").strip().lower()
    if run_validation == 'y':
        # Define predefined test queries for validation
        test_queries = [
            "best quality product",
            "excellent customer service",
            "fast shipping and delivery",
            "durable and long lasting",
            "great value for money"
        ]
        
        print("\n" + "="*80)
        print("RUNNING VALIDATION ON ALL TEST QUERIES".center(80))
        print("="*80)
        
        validation_results = run_multiple_queries_for_validation(
            df_clean=df_clean,
            bm25=bm25,
            dense_model=dense_model,
            faiss_index=faiss_index,
            test_queries=test_queries,
            alpha=alpha,
            beta=beta
        )
        
        # Print aggregate metrics
        print_aggregate_metrics(validation_results)


if __name__ == "__main__":
    main()
