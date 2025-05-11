import numpy as np
from tqdm import tqdm
import os

from src import config, utils

def extract_and_save_features(dataset_items, model, processor, output_file_prefix):
    """
    Extracts features for a list of dataset items (each item is a dict with 'image' and 'label').
    Saves features and corresponding labels.
    """
    features_list = []
    labels_list = []
    ids_list = []

    batch_images = []
    batch_labels = []
    batch_ids = []

    print(f"Extracting features using {config.MODEL_NAME} on {config.DEVICE}...")
    for item in tqdm(dataset_items, desc=f"Extracting for {output_file_prefix}"):
        batch_images.append(item['image'])
        batch_labels.append(item['label'])
        batch_ids.append(item['id'])

        if len(batch_images) == config.BATCH_SIZE:
            batch_features = utils.extract_features_batch(batch_images, model, processor)
            features_list.extend(batch_features)
            labels_list.extend(batch_labels)
            ids_list.extend(batch_ids)
            batch_images, batch_labels, batch_ids = [], [], []

    # Process any remaining images in the last batch
    if batch_images:
        batch_features = utils.extract_features_batch(batch_images, model, processor)
        features_list.extend(batch_features)
        labels_list.extend(batch_labels)
        ids_list.extend(batch_ids)

    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    ids_array = np.array(ids_list) # Keep track of original IDs

    # Normalize features for cosine similarity (important for KD-Tree with Euclidean distance)
    normalized_features = utils.normalize_vectors(features_array)

    # Save features and labels
    features_path = config.FEATURE_VECTORS_DIR / f"{output_file_prefix}_features.npy"
    labels_path = config.FEATURE_VECTORS_DIR / f"{output_file_prefix}_labels.npy"
    ids_path = config.FEATURE_VECTORS_DIR / f"{output_file_prefix}_ids.npy"

    np.save(features_path, normalized_features)
    np.save(labels_path, labels_array)
    np.save(ids_path, ids_array)
    print(f"Saved normalized features to {features_path}")
    print(f"Saved labels to {labels_path}")
    print(f"Saved ids to {ids_path}")

    return normalized_features, labels_array, ids_array


def process_all_features(train_data, query_data):
    """Main function to load model and process features for train and query sets."""
    train_features_path = config.FEATURE_VECTORS_DIR / f"{config.DATASET_NAME}_train_features.npy"
    train_labels_path = config.FEATURE_VECTORS_DIR / f"{config.DATASET_NAME}_train_labels.npy"
    query_features_path = config.FEATURE_VECTORS_DIR / f"{config.DATASET_NAME}_query_features.npy"
    query_labels_path = config.FEATURE_VECTORS_DIR / f"{config.DATASET_NAME}_query_labels.npy"

    if os.path.exists(train_features_path) and os.path.exists(train_labels_path) and \
       os.path.exists(query_features_path) and os.path.exists(query_labels_path):
        print("Features already extracted. Loading from disk.")
        train_features = np.load(train_features_path)
        train_labels = np.load(train_labels_path)
        # train_ids = np.load(config.FEATURE_VECTORS_DIR / f"{config.DATASET_NAME}_train_ids.npy") # Optional to load
        query_features = np.load(query_features_path)
        query_labels = np.load(query_labels_path)
        # query_ids = np.load(config.FEATURE_VECTORS_DIR / f"{config.DATASET_NAME}_query_ids.npy") # Optional to load
        return train_features, train_labels, query_features, query_labels


    model, processor = utils.get_model_and_processor()

    print("Processing train set features...")
    train_features, train_labels, _ = extract_and_save_features(
        train_data, model, processor, f"{config.DATASET_NAME}_train"
    )

    print("Processing query set features...")
    query_features, query_labels, _ = extract_and_save_features(
        query_data, model, processor, f"{config.DATASET_NAME}_query"
    )

    return train_features, train_labels, query_features, query_labels

if __name__ == '__main__':
    # Example usage (requires data from step 1)
    # First, ensure step_1_download_data.py has run or run it here
    from src.step_1_download_data import download_and_prepare_dataset
    train_data_items, query_data_items, _ = download_and_prepare_dataset()

    # Then extract features
    train_feats, train_lbls, query_feats, query_lbls = process_all_features(train_data_items, query_data_items)
    print(f"Train features shape: {train_feats.shape}")
    print(f"Query features shape: {query_feats.shape}")
