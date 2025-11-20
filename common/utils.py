"""
Shared utilities for Replicator models.
"""

import os
import requests
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download


def download_model_weights(
    repo_id: str,
    filename: str,
    cache_dir: Optional[str] = None
) -> str:
    """
    Download model weights from Hugging Face Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., 'facebook/sam-3d-objects')
        filename: Name of the file to download
        cache_dir: Optional cache directory for weights

    Returns:
        Path to downloaded file
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/replicator")

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"Downloading {filename} from {repo_id}...")

    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            local_files_only=False
        )
        print(f"✓ Downloaded to {file_path}")
        return file_path

    except Exception as e:
        print(f"✗ Download failed: {e}")
        raise


def download_file(url: str, output_path: str) -> bool:
    """
    Download a file from URL.

    Args:
        url: URL to download from
        output_path: Local path to save file

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading {url}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✓ Saved to {output_path}")
        return True

    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def validate_image(image_path: str) -> bool:
    """
    Validate that a file is a valid image.

    Args:
        image_path: Path to image file

    Returns:
        True if valid, False otherwise
    """
    try:
        from PIL import Image

        img = Image.open(image_path)
        img.verify()
        return True

    except Exception:
        return False


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_cache_dir(model_name: str) -> Path:
    """
    Get cache directory for a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Path to cache directory
    """
    cache_dir = Path.home() / ".cache" / "replicator" / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def check_gpu_available() -> bool:
    """
    Check if GPU is available for inference.

    Returns:
        True if GPU available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_info() -> dict:
    """
    Get information about available GPU(s).

    Returns:
        Dictionary with GPU information
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}

        return {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_reserved": torch.cuda.memory_reserved(0),
        }

    except ImportError:
        return {"available": False, "error": "PyTorch not installed"}


def format_bytes(bytes: int) -> str:
    """
    Format bytes as human-readable string.

    Args:
        bytes: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} PB"


class ModelCache:
    """Simple cache for model weights and outputs."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[Path]:
        """Get cached file path if exists."""
        cache_path = self.cache_dir / key
        return cache_path if cache_path.exists() else None

    def put(self, key: str, data: bytes) -> Path:
        """Cache data and return path."""
        cache_path = self.cache_dir / key
        cache_path.write_bytes(data)
        return cache_path

    def clear(self):
        """Clear all cached files."""
        import shutil
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
