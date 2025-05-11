from datasets import load_dataset
from src import config, utils
import os
from tqdm import tqdm

def download_and_prepare_dataset():
    """Downloads the dataset and stores relevant splits."""
    print(f"Downloading dataset: {config.DATASET_NAME}")

    # Define file paths for train and query data
    train_data_path = config.DOWNLOADED_DATASET_DIR / f"{config.DATASET_NAME}_train.pkl"
    query_data_path = config.DOWNLOADED_DATASET_DIR / f"{config.DATASET_NAME}_query.pkl"
    class_names_path = config.DOWNLOADED_DATASET_DIR / f"{config.DATASET_NAME}_class_names.pkl"

    if os.path.exists(train_data_path) and \
       os.path.exists(query_data_path) and \
       os.path.exists(class_names_path):
        print("Dataset already downloaded and prepared. Loading from disk.")
        train_data_hf = utils.load_data(train_data_path)
        query_data_hf = utils.load_data(query_data_path)
        class_names = utils.load_data(class_names_path)
        return train_data_hf, query_data_hf, class_names

    # Load the full dataset
    dataset = load_dataset(config.DATASET_NAME)

    # Extract train and query splits
    # Using try-except for datasets that might not have standard "train" or "test" splits
    try:
        train_split_data = dataset[config.DATASET_SPLIT_TRAIN]
        query_split_data = dataset[config.DATASET_SPLIT_QUERY]
    except KeyError:
        print(f"Warning: Specified splits ({config.DATASET_SPLIT_TRAIN}, {config.DATASET_SPLIT_QUERY}) not found.")
        # Fallback: try to use available splits or split the first one
        available_splits = list(dataset.keys())
        if len(available_splits) >= 2:
            print(f"Using splits: {available_splits[0]} for train, {available_splits[1]} for query.")
            train_split_data = dataset[available_splits[0]]
            query_split_data = dataset[available_splits[1]]
        elif len(available_splits) == 1:
            print(f"Only one split found: {available_splits[0]}. Splitting it 80/20 for train/query.")
            # May need a more robust split method if the dataset is very ordered
            dataset_shuffled = dataset[available_splits[0]].shuffle(seed=42)
            split_data = dataset_shuffled.train_test_split(test_size=0.2)
            train_split_data = split_data['train']
            query_split_data = split_data['test']
        else:
            raise ValueError("Dataset has no available splits.")


    # Get class names (assuming 'label' feature exists and has names)
    try:
        # Try to get class names from features if available
        # Most image classification datasets have a 'label' feature of type ClassLabel
        label_feature_name = 'label' # common default
        if 'label' not in train_split_data.features:
            # try other common names
            potential_label_keys = [k for k, v in train_split_data.features.items() if 'label' in k.lower()]
            if potential_label_keys:
                label_feature_name = potential_label_keys[0]

        if hasattr(train_split_data.features[label_feature_name], 'names'):
            class_names = train_split_data.features[label_feature_name].names
        else:
            # If no names attribute, try to infer from unique labels
            all_labels = set(train_split_data[label_feature_name]) | set(query_split_data[label_feature_name])
            class_names = sorted(list(all_labels)) # Will be integers if no names
            print(f"Warning: Class names not directly available. Using unique label values: {class_names}")
    except Exception as e:
        print(f"Could not automatically determine class names: {e}. Using integers as class names.")
        all_labels = set(train_split_data[label_feature_name]) | set(query_split_data[label_feature_name])
        class_names = sorted(list(all_labels))


    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")

    # Store data (images and labels)
    # We store them in a list of dicts for easier processing later
    # Each dict: {'image': PIL.Image, 'label': int, 'id': unique_id}
    # unique_id helps map back if needed, especially if shuffling occurs.

    train_data = []
    for i, item in enumerate(tqdm(train_split_data, desc="Processing train data")):
        if i >= config.MAX_TRAIN_SAMPLES:
            break
        train_data.append({
            'id': f"train_{i}",
            'image': utils.get_image_from_dataset_item(item),
            'label': item[label_feature_name]
        })

    query_data = []
    for i, item in enumerate(tqdm(query_split_data, desc="Processing query data")):
        if i >= config.MAX_QUERY_SAMPLES:
            break
        query_data.append({
            'id': f"query_{i}",
            'image': utils.get_image_from_dataset_item(item),
            'label': item[label_feature_name]
        })

    utils.save_data(train_data_path, train_data)
    utils.save_data(query_data_path, query_data)
    utils.save_data(class_names_path, class_names)

    print(f"Train data: {len(train_data)} samples.")
    print(f"Query data: {len(query_data)} samples.")

    return train_data, query_data, class_names

if __name__ == '__main__':
    # Example usage
    train_set, query_set, classes = download_and_prepare_dataset()
    print(f"First train sample label: {train_set[0]['label']}")
    print(f"First query sample label: {query_set[0]['label']}")
