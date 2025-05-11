import matplotlib.pyplot as plt
from src import config, utils
import os
import numpy as np # Import numpy for inf

def plot_evaluation_results(results):
    """
    Generates and saves plots for time, precision, and similarity vs. eps (proxy for 't').
    """
    if not results:
        print("No results to plot.")
        return

    eps_values = [r['eps'] for r in results]
    avg_search_times = [r['avg_search_time'] for r in results]

    # Create directory for plots if it doesn't exist
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- X-axis log scale decision (for EPS) ---
    # Apply log scale if positive EPS values span a wide range.
    # Log scale cannot display non-positive values on the scaled portion.
    apply_x_log = False
    positive_eps_values = [e for e in eps_values if e > 0]
    if len(positive_eps_values) >= 2: # Need at least two positive eps values to form a meaningful ratio
        if max(positive_eps_values) / min(positive_eps_values) > 10:
            apply_x_log = True
    elif len(positive_eps_values) == 1 and 0 in eps_values and positive_eps_values[0] > 10:
        # Also consider log scale if only one positive eps is large and 0 is present (e.g., [0, 100])
        apply_x_log = True

    # --- Y-axis log scale decision (for Time plot) ---
    apply_y_log_time = False
    positive_times = [t for t in avg_search_times if t > 0]
    if len(positive_times) >= 2:
        if max(positive_times) / min(positive_times) > 10:
            apply_y_log_time = True
    elif len(positive_times) == 1 and avg_search_times.index(positive_times[0]) != -1: # Ensure the single positive time exists
        if positive_times[0] > 1.0: # If a single positive time is large
             apply_y_log_time = True
        elif len(avg_search_times) > 1 and positive_times[0] < 0.01 and max(avg_search_times)/positive_times[0] > 100 : # if one small positive time, but max is much larger
             apply_y_log_time = True


    # 1. Time vs. EPS (proxy for 't')
    plt.figure(figsize=(10, 6))
    plt.plot(eps_values, avg_search_times, 'o-', label="Average Search Time")
    plt.xlabel("Approximation Factor 'eps' (Proxy for BBF 't')")
    plt.ylabel("Average Search Time (seconds)")
    plt.title(f"Search Time vs. EPS ({config.DATASET_NAME}, {config.MODEL_NAME})")

    if apply_x_log:
        plt.xscale('log')
        if 0 in eps_values:
            print("Note for Time plot: X-axis (EPS) is log-scaled. Points with EPS <= 0 are not shown on the log-scaled portion of the axis.")
    if apply_y_log_time:
        plt.yscale('log')
        if any(t <= 0 for t in avg_search_times if t is not None): # Check for non-positive times
             print("Note for Time plot: Y-axis (Time) is log-scaled. Points with Time <= 0 are not shown on the log-scaled portion of the axis.")

    plt.grid(True, which="both", ls="--")
    plt.legend()
    plot_path = config.PLOTS_DIR / f"{config.DATASET_NAME}_time_vs_eps.png"
    plt.savefig(plot_path)
    print(f"Saved time plot to {plot_path}")
    plt.close()

    # 2. Precision vs. EPS (proxy for 't')
    plt.figure(figsize=(12, 7))
    for k_prime_idx, k_prime in enumerate(config.K_PRIME_VALUES):
        precisions = [r['avg_precisions_at_k_prime'][k_prime_idx] for r in results]
        plt.plot(eps_values, precisions, 'o-', label=f"Precision@{k_prime}")
    plt.xlabel("Approximation Factor 'eps' (Proxy for BBF 't')")
    plt.ylabel("Average Precision")
    plt.title(f"Precision vs. EPS ({config.DATASET_NAME}, {config.MODEL_NAME})")

    if apply_x_log: # Reuse decision for x-axis log scale
        plt.xscale('log')
        if 0 in eps_values:
             print("Note for Precision plot: X-axis (EPS) is log-scaled. Points with EPS <= 0 are not shown on the log-scaled portion of the axis.")
    # Y-axis for precision is typically 0-1, log scale not usually applied unless values are extremely close to 0.
    plt.ylim(bottom=max(0, plt.ylim()[0]), top=min(1.05, plt.ylim()[1])) # Ensure y-axis is roughly 0-1

    plt.grid(True, which="both", ls="--")
    plt.legend(title="K'")
    plot_path = config.PLOTS_DIR / f"{config.DATASET_NAME}_precision_vs_eps.png"
    plt.savefig(plot_path)
    print(f"Saved precision plot to {plot_path}")
    plt.close()

    # 3. Average Cosine Similarity vs. EPS (proxy for 't')
    plt.figure(figsize=(12, 7))
    for k_prime_idx, k_prime in enumerate(config.K_PRIME_VALUES):
        similarities = [r['avg_similarities_at_k_prime'][k_prime_idx] for r in results]
        plt.plot(eps_values, similarities, 'o-', label=f"Avg Similarity@{k_prime}")
    plt.xlabel("Approximation Factor 'eps' (Proxy for BBF 't')")
    plt.ylabel("Average Cosine Similarity")
    plt.title(f"Avg. Cosine Similarity vs. EPS ({config.DATASET_NAME}, {config.MODEL_NAME})")

    if apply_x_log: # Reuse decision for x-axis log scale
        plt.xscale('log')
        if 0 in eps_values:
            print("Note for Similarity plot: X-axis (EPS) is log-scaled. Points with EPS <= 0 are not shown on the log-scaled portion of the axis.")
    # Y-axis for similarity (0-1 for normalized vectors), log scale not usually applied.
    plt.ylim(bottom=max(0, plt.ylim()[0] -0.05) , top=min(1.05, plt.ylim()[1]))


    plt.grid(True, which="both", ls="--")
    plt.legend(title="K'")
    plot_path = config.PLOTS_DIR / f"{config.DATASET_NAME}_similarity_vs_eps.png"
    plt.savefig(plot_path)
    print(f"Saved similarity plot to {plot_path}")
    plt.close()

    print("All plots generated.")

if __name__ == '__main__':
    # Example usage: Load results and plot
    eval_results_path = config.EVALUATION_RESULTS_DIR / f"{config.DATASET_NAME}_evaluation.pkl"
    if os.path.exists(eval_results_path):
        evaluation_data = utils.load_data(eval_results_path)
        if evaluation_data:
            # Ensure K_PRIME_VALUES in config matches what was used during evaluation
            num_k_primes_in_results = len(evaluation_data[0]['avg_precisions_at_k_prime'])
            if len(config.K_PRIME_VALUES) != num_k_primes_in_results:
                print(f"Warning: K_PRIME_VALUES in config (length {len(config.K_PRIME_VALUES)}) derived from K_NEIGHBORS={config.K_NEIGHBORS} "
                      f"does not match number of K' values in loaded results ({num_k_primes_in_results}). "
                      f"Plotting will use K' from 1 to {num_k_primes_in_results} based on loaded data structure, but legends might be confusing if K_NEIGHBORS in config is different. "
                      f"Best to align config.K_NEIGHBORS with the data being plotted or regenerate results.")
                # Adjust K_PRIME_VALUES for plotting based on loaded data if there's a mismatch.
                # This is a pragmatic choice for plotting existing data.
                # config.K_PRIME_VALUES = list(range(1, num_k_primes_in_results + 1))
                # print(f"Adjusted config.K_PRIME_VALUES for plotting to: {config.K_PRIME_VALUES}")


        plot_evaluation_results(evaluation_data)
    else:
        print(f"Evaluation results not found at {eval_results_path}. Run step 3 first.")
