from pathlib import Path
import torch

# --- Project Structure ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOWNLOADED_DATASET_DIR = DATA_DIR / "1_downloaded_dataset"
FEATURE_VECTORS_DIR = DATA_DIR / "2_feature_vectors"
# Make evaluation results filename specific to BBF
EVALUATION_RESULTS_DIR = DATA_DIR / "3_evaluation_results_bbf"
PLOTS_DIR = DATA_DIR / "4_plots_bbf"


# --- Dataset Configuration ---
DATASET_NAME = "cifar10"
DATASET_SPLIT_TRAIN = "train"
DATASET_SPLIT_QUERY = "test"
MAX_TRAIN_SAMPLES = 1000
MAX_QUERY_SAMPLES = 1000

# --- Feature Extraction Configuration ---
MODEL_NAME = "google/vit-base-patch16-224-in21k"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

# --- KD-Tree and BBF ANN Configuration ---
K_NEIGHBORS = 10  # Top K neighbors to retrieve

# Parameter 't': maximum number of leaf nodes to inspect during BBF search.
T_VALUES = [1, 2, 5, 10, 15, 20, 30, 50] # Max leaf nodes to check, adjust as needed
# For larger datasets/trees, you might need larger t_values.

# Leaf size for custom KD-Tree construction.
# Smaller leaf_size -> deeper tree, more (smaller) leaves.
# Larger leaf_size -> shallower tree, fewer (larger) leaves.
KDTREE_LEAF_SIZE = 20 # Example value

# --- Evaluation Configuration ---
K_PRIME_VALUES = list(range(1, K_NEIGHBORS + 1))

# --- Ensure directories exist ---
Path(DOWNLOADED_DATASET_DIR).mkdir(parents=True, exist_ok=True)
Path(FEATURE_VECTORS_DIR).mkdir(parents=True, exist_ok=True)
Path(EVALUATION_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
