import os
import requests
from pathlib import Path
from tqdm import tqdm

MODEL_URL = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt"
DEFAULT_MODEL_PATH = "data/comictextdetector.pt"

def download_model(model_path=None, url=None, show_progress=True):
    """Download the model if it doesn't exist.

    Args:
        model_path: Path to save the model. Defaults to DEFAULT_MODEL_PATH
        url: URL to download from. Defaults to MODEL_URL
        show_progress: Show download progress bar

    Returns:
        Path to the downloaded model
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    model_path = Path(model_path)

    # If model already exists, return early
    if model_path.exists():
        return model_path

    # Ensure parent directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Use default URL if not provided
    if url is None:
        url = MODEL_URL

    print(f"Downloading model from {url}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192

    with open(model_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, disable=not show_progress) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"Model saved to {model_path}")
    return model_path

def get_model_path(model_path=None):
    """Get the model path, downloading if necessary.

    Args:
        model_path: Custom model path. If None, uses DEFAULT_MODEL_PATH

    Returns:
        Path to the model (downloaded if needed)
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    model_path = Path(model_path)

    if not model_path.exists():
        return download_model(model_path)

    return model_path
