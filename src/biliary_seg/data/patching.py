import numpy as np
from typing import List, Tuple, Optional, Generator, Any, Literal
from PIL import Image


# def _to_3c(img: np.ndarray) -> np.ndarray:
#     """
#     Ensures 3 colour channels by either clipping or replicating.

#     Args:
#         img: np.ndarray, image to ensure 3 channels
#     """
#     img = np.atleast_3d(img)
#     if img.shape[2] == 1:
#         return np.repeat(img, 3, axis=2)
#     elif img.shape[2] >= 3:
#         return img[:, :, :3]
#     else:
#         raise ValueError(f"Invalid image shape: {img.shape}")


def _resize_image(image: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to the given output size.

    Args:
        image: np.ndarray, image to resize
        output_size: Tuple[int, int], output size
    """
    return np.array(Image.fromarray(image).resize(output_size))


def _safe_subimage(
    image: np.ndarray, top: int, left: int, size: Tuple[int, int]
) -> np.ndarray:
    """
    Get a subimage from the image with the given top-left corner and size. Safe for out-of-bounds
    indices.

    Args:
        image: np.ndarray, image to extract subimage from
        top: int, top coordinate of subimage
        left: int, left coordinate of subimage
        size: Tuple[int, int], size of subimage
    """
    image = np.atleast_3d(image)
    h, w, z = image.shape[:3]
    subimage = np.zeros((size[0], size[1], z), dtype=np.uint8)

    start_top = min(max(0, top), h - 1)
    start_left = min(max(0, left), w - 1)
    end_top = max(min(h, top + size[0]), 0)
    end_left = max(min(w, left + size[1]), 0)

    # Offsets
    start_top_offset = max(0, -top)
    start_left_offset = max(0, -left)

    subimage[start_top_offset : end_top - top, start_left_offset : end_left - left] = (
        image[start_top:end_top, start_left:end_left]
    )

    return subimage


def get_subimage(
    image: np.ndarray,
    top: int,
    left: int,
    size: Tuple[int, int],
    *,
    context_factor: Optional[float] = None,
) -> np.ndarray:
    """
    Get a subimage from the image with the given top-left corner and size.

    Args:
        image: np.ndarray, image to extract subimage from
        top: int, top coordinate of subimage
        left: int, left coordinate of subimage
        size: Tuple[int, int], size of subimage
        context_factor: Optional[float], context factor
    """
    tight = _safe_subimage(image, top, left, size)
    if context_factor is None:
        return tight

    # Extract context window from the image
    context_size = (
        int(size[0] * context_factor),
        int(size[1] * context_factor),
    )
    context = get_subimage(
        image=image,
        top=(top + (size[0] // 2) - (context_size[1] // 2)),
        left=(left + (size[1] // 2) - (context_size[1] // 2)),
        size=context_size,
    )
    context = _resize_image(context, size)

    # Concatenate tight and context windows along the channel dimension
    return np.concatenate([tight, context], axis=2)


def insert_subimage(
    image: np.ndarray,
    subimage: np.ndarray,
    top: int,
    left: int,
    operation: Literal["replace", "or", "max"] = "replace",
) -> np.ndarray:
    """
    Insert a subimage into an image, handling out-of-bounds cases properly.

    Args:
        image: np.ndarray, image to insert subimage into
        subimage: np.ndarray, subimage to insert
        top: int, top coordinate of subimage
        left: int, left coordinate of subimage
        operation: Literal["replace", "or", "max"], operation to perform
    """
    image = np.atleast_3d(image)
    subimage = np.atleast_3d(subimage)

    h, w, z = image.shape[:3]
    sh, sw, sz = subimage.shape[:3]

    # Ensure channel compatibility
    if z != sz:
        raise ValueError("Number of channels in image and subimage must match")

    # Calculate bounds for the area to insert
    start_top = max(0, top)
    start_left = max(0, left)
    end_top = min(h, top + sh)
    end_left = min(w, left + sw)

    # If there's no overlap, return the original image unchanged
    if start_top >= end_top or start_left >= end_left:
        return image

    # Calculate offsets for the subimage
    start_top_offset = max(0, -top)
    start_left_offset = max(0, -left)
    end_top_offset = sh - max(0, (top + sh) - h)
    end_left_offset = sw - max(0, (left + sw) - w)

    # Check if the calculated regions are valid
    if (end_top_offset <= start_top_offset) or (end_left_offset <= start_left_offset):
        return image

    if operation == "replace":
        image[start_top:end_top, start_left:end_left] = subimage[
            start_top_offset:end_top_offset, start_left_offset:end_left_offset
        ]

    elif operation == "or":
        # Get the relevant slices
        img_slice = image[start_top:end_top, start_left:end_left]
        sub_slice = subimage[
            start_top_offset:end_top_offset, start_left_offset:end_left_offset
        ]

        # Perform the OR operation
        if img_slice.size > 0 and sub_slice.size > 0:
            image[start_top:end_top, start_left:end_left] |= sub_slice

    elif operation == "max":
        # Get the relevant slices
        img_slice = image[start_top:end_top, start_left:end_left]
        sub_slice = subimage[
            start_top_offset:end_top_offset, start_left_offset:end_left_offset
        ]

        # Perform the max operation
        if img_slice.size > 0 and sub_slice.size > 0:
            image[start_top:end_top, start_left:end_left] = np.maximum(
                img_slice, sub_slice
            )

    return image


def window_generator(
    image: np.ndarray,
    window_size: Tuple[int, int],
    window_overlap: float = 0.35,
    padding: int = 0,
    *,
    context_factor: Optional[float] = None,
) -> Generator[Tuple[np.ndarray, Tuple[int, int]], None, None]:
    """
    Generate overlapping windows from an image.

    Args:
        image: np.ndarray, image to generate windows from
        window_size: Tuple[int, int], size of windows to generate
        window_overlap: float, overlap between windows
        padding: int, padding around the image
        context_factor: Optional[float], context factor
    """
    h_step = int(window_size[0] * (1 - window_overlap))
    w_step = int(window_size[1] * (1 - window_overlap))
    h, w, *_ = image.shape
    for i in range(-padding, h + padding, h_step):
        for j in range(-padding, w + padding, w_step):
            subimage = get_subimage(
                image=image,
                top=i,
                left=j,
                size=window_size,
                context_factor=context_factor,
            )
            yield subimage, (i, j)
