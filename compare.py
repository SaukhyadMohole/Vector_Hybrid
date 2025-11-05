import random

# Assume previous_method and fusion_search are functions implemented
# that take a query and return ranked list of document indices.

def generate_mock_ground_truth(df, num_queries=20, relevant_per_query=5):
    """Generate random test queries from the dataset with mock relevant docs for evaluation."""
    queries = random.sample(list(df['reviews.text']), num_queries)
    ground_truth = {}
    for query in queries:
        # Randomly assign relevant document indices (for demonstration)
        relevant_indices = random.sample(range(len(df)), relevant_per_query)
        ground_truth[query] = relevant_indices
    return ground_truth

def recall_at_k(retrieved_indices, relevant_indices, k=5):
    """Calculate Recall@k."""
    retrieved_topk = retrieved_indices[:k]
    hits = len(set(retrieved_topk).intersection(set(relevant_indices)))
    return hits / len(relevant_indices)

def evaluate_methods(df, ground_truth, k=5):
    results = {'previous': [], 'improved': []}

    for query, relevant_docs in ground_truth.items():
        prev_results = previous_method(query)  # Replace with your baseline method returning doc indices
        imp_results = fusion_search(query)  # type: ignore # Your improved method returning doc strings or indices

        # Get indices only as lists
        if all(isinstance(r, tuple) or isinstance(r, list) for r in prev_results):
            prev_indices = [r[0] if isinstance(r, tuple) else r for r in prev_results]
        else:
            prev_indices = prev_results

        if all(isinstance(r, tuple) or isinstance(r, list) for r in imp_results):
            imp_indices = [r[0] if isinstance(r, tuple) else r for r in imp_results]
        else:
            imp_indices = imp_results

        results['previous'].append(recall_at_k(prev_indices, relevant_docs, k))
        results['improved'].append(recall_at_k(imp_indices, relevant_docs, k))

    # Calculate average Recall@k
    avg_prev = sum(results['previous']) / len(results['previous'])
    avg_imp = sum(results['improved']) / len(results['improved'])
    print(f"Average Recall@{k} - Previous: {avg_prev:.3f}, Improved: {avg_imp:.3f}")

    from scipy.stats import ttest_rel
    t_stat, p_val = ttest_rel(results['previous'], results['improved'])
    print(f"Paired t-test p-value: {p_val:.4f} (statistically significant if p < 0.05)")

# Example usage
# ground_truth = generate_mock_ground_truth(df_clean)
# evaluate_methods(df_clean, ground_truth)
