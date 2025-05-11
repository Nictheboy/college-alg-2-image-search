from src import config # Ensure config is loaded (creates BBF dirs)
from src.step_1_download_data import download_and_prepare_dataset
from src.step_2_extract_features import process_all_features
from src.step_3_evaluate_ann import evaluate_ann_search # This now uses BBF
from src.step_4_plot_results import plot_evaluation_results

def main():
    print("Starting ANN Evaluation Pipeline (with Custom KD-Tree & BBF)...")
    print(f"Using device: {config.DEVICE}")
    print(f"Dataset: {config.DATASET_NAME}, Model: {config.MODEL_NAME}")
    print(f"Max train samples: {config.MAX_TRAIN_SAMPLES}, Max query samples: {config.MAX_QUERY_SAMPLES}")
    print(f"K (neighbors): {config.K_NEIGHBORS}")
    print(f"BBF T_MAX values: {config.T_VALUES}")
    print(f"Custom KD-Tree Leaf Size: {config.KDTREE_LEAF_SIZE}")


    # --- Step 1: Download and Prepare Dataset ---
    print("\n--- Running Step 1: Download Data ---")
    train_data_items, query_data_items, class_names = download_and_prepare_dataset()
    if not class_names: # Should not happen if download is successful
        print("Error: Could not load class names. Exiting.")
        return
    print(f"Number of classes found: {len(class_names)}")

    # --- Step 2: Extract Features ---
    print("\n--- Running Step 2: Extract Features ---")
    train_features, train_labels, query_features, query_labels = process_all_features(
        train_data_items, query_data_items
    )
    if train_features.size == 0 or query_features.size == 0:
        print("Error: Feature extraction resulted in empty arrays. Exiting.")
        return
    print(f"Train features shape: {train_features.shape}, Train labels shape: {train_labels.shape}")
    print(f"Query features shape: {query_features.shape}, Query labels shape: {query_labels.shape}")

    # --- Step 3: Evaluate ANN Search (using BBF) ---
    print("\n--- Running Step 3: Evaluate BBF ANN Search ---")
    evaluation_results = evaluate_ann_search(
        train_features, train_labels, query_features, query_labels
    )
    if not evaluation_results:
        print("Error: Evaluation step produced no results. Exiting.")
        return

    # --- Step 4: Plot Results ---
    print("\n--- Running Step 4: Plot BBF Results ---")
    plot_filename_prefix = f"{config.DATASET_NAME}_LS{config.KDTREE_LEAF_SIZE}_"
    plot_evaluation_results(evaluation_results, 
                            param_name_for_plot="t_max", 
                            plot_file_prefix=plot_filename_prefix)

    print("\nPipeline finished successfully!")

if __name__ == '__main__':
    main()
