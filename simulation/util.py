import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm
from pathlib import Path


def add_noise(image: NDArray[np.float32], snr: float = 0.1) -> NDArray[np.float32]:
    """
    Add noise to the input image.

    Args:
        image (NDArray[np.float32]): Input image.
        snr (float): Signal-to-noise ratio.

    Returns:
        NDArray[np.float32]: Noisy image.
    """
    sample_var = image.var()
    target_noise_var = sample_var / snr
    print(
        f"Sample Variance: {sample_var:.7f}. Target Noise Variance: {target_noise_var:.7f}")
    noise = np.random.normal(0, np.sqrt(target_noise_var), size=image.shape)
    return (image + noise).astype(np.float32)
    
from PIL import Image
from skimage import morphology
from skimage.filters import threshold_li


def threshold(image: NDArray[np.float32]) -> NDArray[np.uint8]:
    """
    Threshold the input image.

    Args:
        image (NDArray[np.float32]): Input image.

    Returns:
        NDArray[np.uint8]: Thresholded image.
    """
    image_threshold = 0.015
    # image_threshold: np.float32 = threshold_li(image)
    return (image > image_threshold).astype(np.uint8) * 255


def processing(image: NDArray[np.float32]) -> NDArray[np.uint8]:
    """
    Apply processing steps to the input image.

    Args:
        image (NDArray[np.float32]): Input image.

    Returns:
        NDArray[np.uint8]: Processed image.
    """
    image = morphology.black_tophat(image, morphology.disk(20))
    image = morphology.remove_small_objects(
        image > 0, min_size=128, connectivity=1)
    image = morphology.closing(image, morphology.disk(20))
    return image.astype(np.uint8) * 255


def processing2(image: NDArray[np.float32]) -> NDArray[np.uint8]:
    """
    Apply processing steps to the input image.

    Args:
        image (NDArray[np.float32]): Input image.

    Returns:
        NDArray[np.uint8]: Processed image.
    """
    return image

def numpy_to_PIL(image: NDArray) -> Image:
    """
    Convert numpy array to PIL Image.

    Args:
        image (NDArray): Input image.

    Returns:
        Image: PIL Image.
    """
    return Image.fromarray(image).convert('L')


def self2binary(image: NDArray[np.float64]) -> NDArray[np.uint8]:
    """
    Convert image to binary.

    Args:
        image (NDArray[np.float64]): Input image.

    Returns:
        NDArray[np.uint8]: Binary image.
    """
    clean_image = threshold(image)
    clean_image = processing(clean_image)
    return clean_image

def self2binary2(image: NDArray[np.float64]) -> NDArray[np.uint8]:
    """
    Convert image to binary.

    Args:
        image (NDArray[np.float64]): Input image.

    Returns:
        NDArray[np.uint8]: Binary image.
    """
    clean_image = threshold(image)
    clean_image = processing2(clean_image)
    return clean_image