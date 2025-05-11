import pickle
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import time

from src import config

def save_data(filepath, data):
    """Saves data to a file using pickle."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filepath}")

def load_data(filepath):
    """Loads data from a pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Data loaded from {filepath}")
    return data

def get_model_and_processor():
    """Loads the pre-trained model and image processor."""
    processor = AutoImageProcessor.from_pretrained(config.MODEL_NAME)
    model = AutoModel.from_pretrained(config.MODEL_NAME).to(config.DEVICE)
    model.eval() # Set model to evaluation mode
    return model, processor

def extract_features_batch(images, model, processor):
    """
    Extracts features for a batch of PIL images.
    Images should be a list of PIL Image objects.
    """
    if not images:
        return np.array([])

    # Preprocess images
    try:
        inputs = processor(images=images, return_tensors="pt", padding=True).to(config.DEVICE)
    except Exception as e:
        print(f"Error processing images: {e}")
        # Fallback for images that might be in a different mode (e.g. grayscale for some datasets)
        processed_images = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            processed_images.append(img)
        inputs = processor(images=processed_images, return_tensors="pt", padding=True).to(config.DEVICE)


    with torch.no_grad():
        outputs = model(**inputs)

    # We usually take the [CLS] token embedding or mean pool of last hidden states
    # For ViT, the last_hidden_state[:, 0, :] is the [CLS] token embedding
    # For ResNet, often an adaptive average pooling is applied by the model itself or a custom head.
    # AutoModel typically gives `last_hidden_state` and `pooler_output`.
    # `pooler_output` is often suitable for classification tasks.
    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
        features = outputs.pooler_output
    else:
        # Fallback for models like ViT that don't have a separate pooler_output in AutoModel output
        features = outputs.last_hidden_state[:, 0, :] # CLS token

    return features.cpu().numpy()

def normalize_vectors(vectors):
    """Normalizes a list of vectors to unit length (L2 norm)."""
    return normalize(vectors, norm='l2', axis=1)

def calculate_cosine_similarity_matrix(query_vectors, dataset_vectors):
    """
    Calculates cosine similarity between each query vector and all dataset vectors.
    Assumes vectors are already L2 normalized.
    Returns a matrix where C[i, j] is sim(query_i, dataset_j).
    """
    # Ensure they are numpy arrays
    query_vectors = np.array(query_vectors)
    dataset_vectors = np.array(dataset_vectors)

    # Cosine similarity for normalized vectors is their dot product
    similarity_matrix = np.dot(query_vectors, dataset_vectors.T)
    return similarity_matrix

def get_image_from_dataset_item(item, image_key='image'):
    """Extracts PIL image from a Hugging Face dataset item."""
    img = item[image_key]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img)) # Convert if it's a dict or array
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img
