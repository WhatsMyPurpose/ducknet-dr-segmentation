import numpy as np
from typing import Tuple, Optional
from biliary_seg.data.patching import window_generator, insert_subimage

def predict_mask(
    image: np.ndarray,
    model,
    image_size: Tuple[int, int],
    window_overlap: float = 0.35,
    threshold: float = 0.5,
    batch_size: int = 4,
    context_factor: Optional[float] = None,
) -> np.ndarray:
    """Predict segmentation mask for a given image using a sliding window approach."""
    pred_mask = np.zeros_like(image[..., 0], dtype=bool)
    window_generator = window_generator(
        image,
        window_size=image_size,
        window_overlap=window_overlap,
        context_factor=context_factor,
    )
    batch_windows = []
    batch_coords = []
    for window, (top, left) in window_generator:
        batch_windows.append(window[np.newaxis, ...] / 255.0)
        batch_coords.append((top, left))

        if len(batch_windows) == batch_size:
            batch_windows = np.vstack(batch_windows)
            preds = model.predict(batch_windows, verbose=0)

            for pred, (top, left) in zip(preds, batch_coords):
                pred_mask = insert_subimage(
                    pred_mask, pred > threshold, top, left, operation="or"
                )

            batch_windows = []
            batch_coords = []

    # Process any remaining windows in the batch
    if batch_windows:
        batch_windows = np.vstack(batch_windows)
        preds = model.predict(batch_windows)

        for pred, (top, left) in zip(preds, batch_coords):
            pred_mask = insert_subimage(
                pred_mask, pred > threshold, top, left, operation="or"
            )

    return pred_mask