from src import config # To ensure config is loaded and directories are checked/created
from src.step_1_download_data import download_and_prepare_dataset
from src.step_2_extract_features import process_all_features
from src.step_3_evaluate_ann import evaluate_ann_search
from src.step_4_plot_results import plot_evaluation_results

def main():
    print("Starting ANN Evaluation Pipeline...")
    print(f"Using device: {config.DEVICE}")
    print(f"Dataset: {config.DATASET_NAME}, Model: {config.MODEL_NAME}")
    print(f"Max train samples: {config.MAX_TRAIN_SAMPLES}, Max query samples: {config.MAX_QUERY_SAMPLES}")
    print(f"K (neighbors): {config.K_NEIGHBORS}, EPS values: {config.EPS_VALUES}")


    # --- Step 1: Download and Prepare Dataset ---
    print("\n--- Running Step 1: Download Data ---")
    # train_data is list of dicts {'id': ..., 'image': PIL, 'label': ...}
    train_data_items, query_data_items, class_names = download_and_prepare_dataset()
    print(f"Number of classes found: {len(class_names)}")

    # --- Step 2: Extract Features ---
    print("\n--- Running Step 2: Extract Features ---")
    # Returns numpy arrays: features (N, D), labels (N,)
    train_features, train_labels, query_features, query_labels = process_all_features(
        train_data_items, query_data_items
    )
    print(f"Train features shape: {train_features.shape}, Train labels shape: {train_labels.shape}")
    print(f"Query features shape: {query_features.shape}, Query labels shape: {query_labels.shape}")

    # --- Step 3: Evaluate ANN Search ---
    print("\n--- Running Step 3: Evaluate ANN Search ---")
    # Returns a list of dicts, each dict for an eps value
    evaluation_results = evaluate_ann_search(
        train_features, train_labels, query_features, query_labels
    )

    # --- Step 4: Plot Results ---
    print("\n--- Running Step 4: Plot Results ---")
    plot_evaluation_results(evaluation_results)

    print("\nPipeline finished successfully!")

if __name__ == '__main__':
    main()
