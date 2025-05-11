import matplotlib.pyplot as plt
from src import config, utils # config will now point to the BBF specific dirs/params
import os
import numpy as np

def plot_evaluation_results(results, param_name_for_plot="t_max", plot_file_prefix=""):
    """
    Generates and saves plots for time, precision, and similarity vs. the specified parameter.
    param_name_for_plot: The key in the results dictionary for the x-axis parameter (e.g., "t_max").
    plot_file_prefix: A string to prepend to plot filenames for specificity.
    """
    if not results:
        print("No results to plot.")
        return

    param_values = [r[param_name_for_plot] for r in results]
    avg_search_times = [r['avg_search_time'] for r in results]

    # Ensure plots directory from config exists
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- X-axis log scale decision ---
    apply_x_log = False
    positive_param_values = [p for p in param_values if p > 0]
    if len(positive_param_values) >= 2:
        min_pos_val = min(positive_param_values)
        if min_pos_val > 0 and (max(positive_param_values) / min_pos_val) > 10:
            apply_x_log = True
    elif len(positive_param_values) == 1 and positive_param_values[0] > sum(param_values)/len(param_values) * 5 and len(param_values) > 1 :
         apply_x_log = True # Single large positive value with others (possibly zero)

    # --- Y-axis log scale decision for Time plot ---
    apply_y_log_time = False
    positive_times = [t for t in avg_search_times if t > 0]
    if len(positive_times) >= 2:
        min_pos_time = min(positive_times)
        if min_pos_time > 0 and (max(positive_times) / min_pos_time) > 10:
            apply_y_log_time = True
    elif len(positive_times) == 1 and len(avg_search_times) > 1 and positive_times[0] > 0.1: # Single significant positive time
        apply_y_log_time = True


    # 1. Time vs. param_name_for_plot
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, avg_search_times, 'o-', label="Average Search Time")
    plt.xlabel(f"BBF Parameter '{param_name_for_plot}' (Max Leaves Inspected)")
    plt.ylabel("Average Search Time (seconds)")
    plt.title(f"Search Time vs. {param_name_for_plot} ({config.DATASET_NAME}, {config.MODEL_NAME}, LeafS={config.KDTREE_LEAF_SIZE})")
    if apply_x_log: plt.xscale('log')
    if apply_y_log_time: plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plot_path = config.PLOTS_DIR / f"{plot_file_prefix}time_vs_{param_name_for_plot}.png"
    plt.savefig(plot_path)
    print(f"Saved time plot to {plot_path}")
    plt.close()

    # 2. Precision vs. param_name_for_plot
    plt.figure(figsize=(12, 7))
    for k_prime_idx, k_prime in enumerate(config.K_PRIME_VALUES):
        precisions = [r['avg_precisions_at_k_prime'][k_prime_idx] for r in results]
        plt.plot(param_values, precisions, 'o-', label=f"Precision@{k_prime}")
    plt.xlabel(f"BBF Parameter '{param_name_for_plot}'")
    plt.ylabel("Average Precision")
    plt.title(f"Precision vs. {param_name_for_plot} ({config.DATASET_NAME}, {config.MODEL_NAME}, LeafS={config.KDTREE_LEAF_SIZE})")
    if apply_x_log: plt.xscale('log')
    plt.ylim(bottom=max(-0.05, plt.ylim()[0]), top=min(1.05, plt.ylim()[1])) # Ensure y-axis is roughly 0-1
    plt.grid(True, which="both", ls="--")
    plt.legend(title="K'")
    plot_path = config.PLOTS_DIR / f"{plot_file_prefix}precision_vs_{param_name_for_plot}.png"
    plt.savefig(plot_path)
    print(f"Saved precision plot to {plot_path}")
    plt.close()

    # 3. Average Cosine Similarity vs. param_name_for_plot
    plt.figure(figsize=(12, 7))
    for k_prime_idx, k_prime in enumerate(config.K_PRIME_VALUES):
        similarities = [r['avg_similarities_at_k_prime'][k_prime_idx] for r in results]
        plt.plot(param_values, similarities, 'o-', label=f"Avg Similarity@{k_prime}")
    plt.xlabel(f"BBF Parameter '{param_name_for_plot}'")
    plt.ylabel("Average Cosine Similarity")
    plt.title(f"Avg. Cosine Similarity vs. {param_name_for_plot} ({config.DATASET_NAME}, {config.MODEL_NAME}, LeafS={config.KDTREE_LEAF_SIZE})")
    if apply_x_log: plt.xscale('log')
    plt.ylim(bottom=max(-0.05, plt.ylim()[0] -0.05) , top=min(1.05, plt.ylim()[1])) # Ensure y-axis is roughly 0-1 for similarity
    plt.grid(True, which="both", ls="--")
    plt.legend(title="K'")
    plot_path = config.PLOTS_DIR / f"{plot_file_prefix}similarity_vs_{param_name_for_plot}.png"
    plt.savefig(plot_path)
    print(f"Saved similarity plot to {plot_path}")
    plt.close()

    print("All plots generated.")

if __name__ == '__main__':
    # This part now needs to load the BBF specific results
    param_key = "t_max"
    # Construct filename based on current config
    results_filename = f"{config.DATASET_NAME}_evaluation_bbf_t_leaf{config.KDTREE_LEAF_SIZE}.pkl"
    eval_results_path = config.EVALUATION_RESULTS_DIR / results_filename
    
    plot_filename_prefix_str = f"{config.DATASET_NAME}_LS{config.KDTREE_LEAF_SIZE}_"


    if os.path.exists(eval_results_path):
        evaluation_data = utils.load_data(eval_results_path)
        if evaluation_data:
            # Optional: Check consistency of K_PRIME_VALUES
            num_k_primes_in_results = len(evaluation_data[0]['avg_precisions_at_k_prime'])
            if len(config.K_PRIME_VALUES) != num_k_primes_in_results:
                print(f"Warning: K_PRIME_VALUES in config (len {len(config.K_PRIME_VALUES)}) "
                      f"differs from results (len {num_k_primes_in_results}). Adjust config.K_NEIGHBORS or expect mismatches in plot legend.")
            
            plot_evaluation_results(evaluation_data, param_name_for_plot=param_key, plot_file_prefix=plot_filename_prefix_str)
        else:
            print(f"No data in evaluation results file: {eval_results_path}")
    else:
        print(f"Evaluation results not found at {eval_results_path}. Run step 3 (evaluate_ann_search) first.")
