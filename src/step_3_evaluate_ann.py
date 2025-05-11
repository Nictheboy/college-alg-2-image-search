import numpy as np
from scipy.spatial import KDTree
import time
from tqdm import tqdm
import os

from src import config, utils

def evaluate_ann_search(train_features, train_labels, query_features, query_labels):
    """
    Builds KD-Tree, performs ANN search with varying `eps` (proxy for 't'),
    and evaluates precision and similarity.
    """
    eval_results_path = config.EVALUATION_RESULTS_DIR / f"{config.DATASET_NAME}_evaluation.pkl"
    if os.path.exists(eval_results_path):
        print("Evaluation results found. Loading from disk.")
        return utils.load_data(eval_results_path)

    print("Building KD-Tree from training features...")
    # For KDTree, Euclidean distance is typically used.
    # Since we normalized features, Euclidean distance minimization is related to cosine similarity maximization.
    # d_euc^2 = ||A-B||^2 = ||A||^2 - 2A.B + ||B||^2. If ||A||=||B||=1, then d_euc^2 = 2 - 2A.B.
    # So minimizing Euclidean distance maximizes A.B (cosine similarity).
    kdtree = KDTree(train_features)
    print("KD-Tree built.")

    all_results = [] # Store results for each eps value

    for eps_val in tqdm(config.EPS_VALUES, desc="Evaluating EPS values"):
        total_search_time = 0
        # Initialize accumulators for metrics
        # precisions_at_k_prime[k_prime_idx] = sum_of_precisions_for_this_k_prime
        # similarities_at_k_prime[k_prime_idx] = sum_of_avg_similarities_for_this_k_prime
        precisions_at_k_prime_sum = [0.0] * len(config.K_PRIME_VALUES)
        similarities_at_k_prime_sum = [0.0] * len(config.K_PRIME_VALUES)
        num_queries = len(query_features)

        for i in range(num_queries):
            query_vec = query_features[i]
            query_label = query_labels[i]

            start_time = time.perf_counter()
            # Perform ANN search. `p=2` for Euclidean distance.
            # `k` is the number of nearest neighbors to find.
            # `eps` > 0 enables BBF-like approximate search.
            distances, indices = kdtree.query(query_vec, k=config.K_NEIGHBORS, eps=eps_val, p=2)
            search_time = time.perf_counter() - start_time
            total_search_time += search_time

            # Ensure results are iterable even if k=1
            if config.K_NEIGHBORS == 1:
                # scipy returns scalars if k=1 and query is 1D
                # but we query with single vector, so indices/distances are 1D arrays of length k
                # no special handling needed for k=1 here if kdtree.query behaves consistently.
                # However, if distances/indices are not array-like for k=1, wrap them.
                if not hasattr(indices, '__iter__'): # Check if it's not iterable (e.g. a scalar)
                    indices = [indices]
                    distances = [distances]


            retrieved_labels = train_labels[indices]
            retrieved_features = train_features[indices] # These are already normalized

            for k_prime_idx, k_prime in enumerate(config.K_PRIME_VALUES):
                # Consider only the top k_prime results
                current_k_prime_indices = indices[:k_prime]
                current_k_prime_labels = train_labels[current_k_prime_indices]
                current_k_prime_features = train_features[current_k_prime_indices]

                # 1. Precision@k_prime
                # Number of retrieved items with the same label as the query
                correct_matches = np.sum(current_k_prime_labels == query_label)
                precision = correct_matches / k_prime
                precisions_at_k_prime_sum[k_prime_idx] += precision

                # 2. Average Cosine Similarity@k_prime
                # query_vec and current_k_prime_features are L2 normalized
                # Cosine similarity = dot product
                if current_k_prime_features.ndim == 1: # If k_prime = 1
                     similarities = np.dot(current_k_prime_features, query_vec)
                else:
                    similarities = np.dot(current_k_prime_features, query_vec) # shape (k_prime,)

                avg_similarity = np.mean(similarities) if similarities.size > 0 else 0.0
                similarities_at_k_prime_sum[k_prime_idx] += avg_similarity


        avg_search_time = total_search_time / num_queries
        avg_precisions = [s / num_queries for s in precisions_at_k_prime_sum]
        avg_similarities = [s / num_queries for s in similarities_at_k_prime_sum]

        all_results.append({
            "eps": eps_val,
            "avg_search_time": avg_search_time,
            "avg_precisions_at_k_prime": avg_precisions, # List, one for each K'
            "avg_similarities_at_k_prime": avg_similarities # List, one for each K'
        })
        print(f"EPS: {eps_val:.2f}, Time: {avg_search_time:.6f}s, "
              f"P@{config.K_PRIME_VALUES[-1]}: {avg_precisions[-1]:.3f}, "
              f"Sim@{config.K_PRIME_VALUES[-1]}: {avg_similarities[-1]:.3f}")

    utils.save_data(eval_results_path, all_results)
    return all_results

if __name__ == '__main__':
    # Example usage (requires data from step 1 and 2)
    from src.step_1_download_data import download_and_prepare_dataset
    from src.step_2_extract_features import process_all_features

    # Step 1: Download data
    train_items, query_items, _ = download_and_prepare_dataset()

    # Step 2: Extract features
    train_f, train_l, query_f, query_l = process_all_features(train_items, query_items)

    # Step 3: Evaluate
    results = evaluate_ann_search(train_f, train_l, query_f, query_l)
    print("\nEvaluation complete. Results:")
    for res in results:
        print(f"Eps: {res['eps']}, Avg Time: {res['avg_search_time']:.5f}, "
              f"Avg P@{config.K_NEIGHBORS}: {res['avg_precisions_at_k_prime'][-1]:.3f}, "
              f"Avg Sim@{config.K_NEIGHBORS}: {res['avg_similarities_at_k_prime'][-1]:.3f}")
