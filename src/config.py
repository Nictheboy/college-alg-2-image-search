from pathlib import Path
import torch

# --- Project Structure ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOWNLOADED_DATASET_DIR = DATA_DIR / "1_downloaded_dataset"
FEATURE_VECTORS_DIR = DATA_DIR / "2_feature_vectors"
EVALUATION_RESULTS_DIR = DATA_DIR / "3_evaluation_results"
PLOTS_DIR = DATA_DIR / "4_plots"

# --- Dataset Configuration ---
# Example: "beans", "cifar10", "mnist" (mnist is grayscale, less ideal for general models)
# "beans" is a good small color image dataset with 3 classes.
DATASET_NAME = "beans"
# For Hugging Face datasets, specify split if needed.
# 'train' for KD-tree, 'test' for queries.
DATASET_SPLIT_TRAIN = "train"
DATASET_SPLIT_QUERY = "test"
# For demonstration, limit the number of samples
MAX_TRAIN_SAMPLES = 500 # Max samples for KD-tree dataset
MAX_QUERY_SAMPLES = 100 # Max samples for query set

# --- Feature Extraction Configuration ---
# A powerful model from Hugging Face.
# e.g., "google/vit-base-patch16-224-in21k", "facebook/dino-vits16"
# "microsoft/resnet-50" is also a solid choice.
MODEL_NAME = "google/vit-base-patch16-224-in21k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # Batch size for feature extraction

# --- KD-Tree and ANN Configuration ---
K_NEIGHBORS = 10  # Top K neighbors to retrieve
# `eps` in scipy.spatial.KDTree.query() controls the approximation.
# An `eps` of 0 means an exact search. Larger `eps` allows for a larger
# approximation error bound, potentially speeding up the search.
# This will serve as our proxy for the "t" parameter (max candidates/effort).
# We will map these "effort levels" to `eps` values.
# Scipy's KDTree doesn't have a direct "t" (max leaves checked) like some BBF implementations.
# We'll use `eps` to vary the trade-off. Smaller eps = more accurate, more time.
EPS_VALUES = [0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0] # Approximation levels for BBF-like search

# --- Evaluation Configuration ---
# K_PRIME_VALUES will be 1, 2, ..., K_NEIGHBORS
K_PRIME_VALUES = list(range(1, K_NEIGHBORS + 1))

# --- Ensure directories exist ---
Path(DOWNLOADED_DATASET_DIR).mkdir(parents=True, exist_ok=True)
Path(FEATURE_VECTORS_DIR).mkdir(parents=True, exist_ok=True)
Path(EVALUATION_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
