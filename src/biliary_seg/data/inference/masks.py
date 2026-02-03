import cv2
import numpy as np


def get_slide_bg_mask(
    img: np.ndarray,
    threshold: int = 200,
    kernel_size: int = 5,
    apply_morphology: bool = True,
) -> np.ndarray:
    """Get background mask of a slide image using thresholding across
    RGB channels.

    Args:
        img: np.ndarray, slide image
        threshold: int, threshold value
        kernel_size: int, kernel size for morphological operations
        apply_morphology: bool, whether to apply morphological operations
    """
    bg_mask = np.all(img > threshold, axis=-1).astype(np.uint8) * 255
    if apply_morphology:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        # remove small tissue regions inside white background
        bg_mask = ~cv2.morphologyEx(~bg_mask, cv2.MORPH_OPEN, kernel)
        # remove small white background regions inside tissue
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)

    return bg_mask
