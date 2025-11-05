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

    def hybrid_fusion(query, alpha=0.7, beta=0.3, top_k=20):
        dense_results = dense_retrieval(query, top_k=top_k*3)
        sparse_results = sparse_retrieval(query, top_k=top_k*3)

        dense_dict = {idx: score for idx, score in dense_results}
        sparse_dict = {idx: score for idx, score in sparse_results}

        # normalize
        if dense_dict:
            d_min, d_max = min(dense_dict.values()), max(dense_dict.values())
            dense_dict = {idx: (score - d_min) / (d_max - d_min + 1e-9) for idx, score in dense_dict.items()}
        if sparse_dict:
            s_min, s_max = min(sparse_dict.values()), max(sparse_dict.values())
            sparse_dict = {idx: (score - s_min) / (s_max - s_min + 1e-9) for idx, score in sparse_dict.items()}

        all_indices = set(dense_dict.keys()).union(sparse_dict.keys())
        fused_scores = {}
        for idx in all_indices:
            v_score = dense_dict.get(idx, 0.0)
            k_score = sparse_dict.get(idx, 0.0)
            fused_scores[idx] = alpha*v_score + beta*k_score

        top_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        return top_indices

    # Example: assume each review's true relevant doc is itself (for demo)
    sample_queries = random.sample(list(df_clean['reviews.text'].values), 5)

    print("Query Eval (Precision@5, Recall@5, nDCG@5):")
    for i, query in enumerate(sample_queries, 1):
        gt_idx = df_clean.index[df_clean['reviews.text']==query][0]
        relevant = [gt_idx]

        dense_idx = [idx for idx,_ in dense_retrieval(query, top_k=5)]
        sparse_idx = [idx for idx,_ in sparse_retrieval(query, top_k=5)]
        hybrid_idx = hybrid_fusion(query, alpha=0.7, beta=0.3, top_k=5)

        p_d, r_d, n_d = precision_recall_ndcg(dense_idx, relevant)
        p_s, r_s, n_s = precision_recall_ndcg(sparse_idx, relevant)
        p_h, r_h, n_h = precision_recall_ndcg(hybrid_idx, relevant)

        print(f"\nQuery {i}: {query[:80]}...")
        print(f"Vector-only   -> P@5: {p_d:.2f}, R@5: {r_d:.2f}, nDCG@5: {n_d:.2f}")
        print(f"BM25-only     -> P@5: {p_s:.2f}, R@5: {r_s:.2f}, nDCG@5: {n_s:.2f}")
        print(f"Hybrid Fusion -> P@5: {p_h:.2f}, R@5: {r_h:.2f}, nDCG@5: {n_h:.2f}")

if __name__ == "__main__":
    main()
