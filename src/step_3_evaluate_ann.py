import numpy as np
import time
from tqdm import tqdm
import os

from src import config, utils
from src.kdtree_bbf import build_kdtree, bbf_knn_search # Import custom KD-Tree and BBF

def evaluate_ann_search(train_features, train_labels, query_features, query_labels):
    """
    Builds a custom KD-Tree, performs BBF ANN search with varying 't_max',
    and evaluates precision and similarity.
    """
    # Adjusted filename for BBF results
    eval_results_filename = f"{config.DATASET_NAME}_evaluation_bbf_t_leaf{config.KDTREE_LEAF_SIZE}.pkl"
    eval_results_path = config.EVALUATION_RESULTS_DIR / eval_results_filename
    
    if os.path.exists(eval_results_path):
        print(f"BBF Evaluation results found ({eval_results_filename}). Loading from disk.")
        return utils.load_data(eval_results_path)

    print(f"Building custom KD-Tree from training features (leaf_size={config.KDTREE_LEAF_SIZE})...")
    # train_features are already normalized. KD-Tree will use Euclidean distance.
    custom_kdtree_root = build_kdtree(train_features, leaf_size=config.KDTREE_LEAF_SIZE)
    
    if custom_kdtree_root is None:
        print("Error: KD-Tree construction failed (root is None). Check dataset or KDTREE_LEAF_SIZE.")
        return []
    print("Custom KD-Tree built.")

    all_results = []

    for t_val in tqdm(config.T_VALUES, desc="Evaluating T_MAX values for BBF"):
        total_search_time = 0.0
        precisions_at_k_prime_sum = np.zeros(len(config.K_PRIME_VALUES))
        similarities_at_k_prime_sum = np.zeros(len(config.K_PRIME_VALUES))
        num_queries = len(query_features)

        if num_queries == 0:
            print("No query features to evaluate.")
            return []

        for i in range(num_queries):
            query_vec = query_features[i]
            query_label = query_labels[i]

            start_time = time.perf_counter()
            # Perform BBF ANN search
            # bbf_knn_search returns SQUARED Euclidean distances and indices
            distances_sq, indices = bbf_knn_search(
                custom_kdtree_root,
                train_features, # Pass the original dataset for fetching points by index
                query_vec,
                k=config.K_NEIGHBORS,
                t_max=t_val
            )
            search_time = time.perf_counter() - start_time
            total_search_time += search_time
            
            indices = np.array(indices, dtype=int) # Ensure integer indices
            
            for k_prime_idx, k_prime in enumerate(config.K_PRIME_VALUES):
                actual_retrieved_count = len(indices)
                # Consider only the top k_prime results from what was found (up to K_NEIGHBORS)
                # indices are already sorted by distance by bbf_knn_search
                current_top_n = min(k_prime, actual_retrieved_count)

                if current_top_n == 0:
                    # Precision and similarity remain 0 for this k_prime if no results
                    continue

                current_k_prime_indices = indices[:current_top_n]
                current_k_prime_labels = train_labels[current_k_prime_indices]
                
                # 1. Precision@k_prime
                correct_matches = np.sum(current_k_prime_labels == query_label)
                # Precision is correct_matches / k_prime (the target number for this rank)
                precision = correct_matches / k_prime 
                precisions_at_k_prime_sum[k_prime_idx] += precision

                # 2. Average Cosine Similarity@k_prime
                # query_vec and retrieved features are L2 normalized. Cosine sim = dot product.
                retrieved_k_prime_features = train_features[current_k_prime_indices]
                
                similarities_values = np.array([])
                if retrieved_k_prime_features.size > 0 :
                    if retrieved_k_prime_features.ndim == 1: # Single neighbor found and k_prime=1
                        similarities_values = np.array([np.dot(retrieved_k_prime_features, query_vec)])
                    else: # Multiple neighbors
                        similarities_values = np.dot(retrieved_k_prime_features, query_vec)
                
                avg_similarity_for_this_query_k_prime = np.mean(similarities_values) if similarities_values.size > 0 else 0.0
                similarities_at_k_prime_sum[k_prime_idx] += avg_similarity_for_this_query_k_prime

        avg_search_time = total_search_time / num_queries
        avg_precisions = precisions_at_k_prime_sum / num_queries
        avg_similarities = similarities_at_k_prime_sum / num_queries

        # Store results for this t_val
        result_entry = {
            "t_max": t_val,
            "avg_search_time": avg_search_time,
            "avg_precisions_at_k_prime": avg_precisions.tolist(),
            "avg_similarities_at_k_prime": avg_similarities.tolist()
        }
        all_results.append(result_entry)

        # Display info for K_NEIGHBORS (the largest K')
        k_display_val = config.K_NEIGHBORS
        try:
            k_display_idx = config.K_PRIME_VALUES.index(k_display_val)
            print(f"T_max: {t_val:3d}, Time: {avg_search_time:.5f}s, "
                  f"P@{k_display_val}: {avg_precisions[k_display_idx]:.3f}, "
                  f"Sim@{k_display_val}: {avg_similarities[k_display_idx]:.3f}")
        except ValueError:
            print(f"T_max: {t_val:3d}, Time: {avg_search_time:.5f}s (K_NEIGHBORS not in K_PRIME_VALUES for display)")


    utils.save_data(eval_results_path, all_results)
    print(f"BBF Evaluation results saved to {eval_results_path}")
    return all_results

if __name__ == '__main__':
    # Ensure dependent steps can run if this is executed standalone
    from src.step_1_download_data import download_and_prepare_dataset
    from src.step_2_extract_features import process_all_features

    print("Running standalone Step 3: Evaluate BBF ANN Search")
    train_items, query_items, _ = download_and_prepare_dataset()
    train_f, train_l, query_f, query_l = process_all_features(train_items, query_items)

    if train_f.size == 0 or query_f.size == 0:
        print("No features found. Exiting evaluation.")
    else:
        results = evaluate_ann_search(train_f, train_l, query_f, query_l)
        if results:
            print("\nBBF Evaluation complete. Summary of results:")
            for res in results:
                k_disp_val = config.K_NEIGHBORS
                try:
                    k_disp_idx = config.K_PRIME_VALUES.index(k_disp_val)
                    print(f"T_max: {res['t_max']:3d}, Avg Time: {res['avg_search_time']:.5f}, "
                        f"Avg P@{k_disp_val}: {res['avg_precisions_at_k_prime'][k_disp_idx]:.3f}, "
                        f"Avg Sim@{k_disp_val}: {res['avg_similarities_at_k_prime'][k_disp_idx]:.3f}")
                except (ValueError, IndexError):
                     print(f"T_max: {res['t_max']:3d}, Avg Time: {res['avg_search_time']:.5f} (Error displaying specific K' metrics)")
        else:
            print("No results from BBF evaluation.")
