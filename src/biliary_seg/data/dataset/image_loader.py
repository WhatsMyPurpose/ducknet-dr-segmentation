import os
import numpy as np

# import albumentations as A
from typing import List, Tuple, Optional, Generator, Any, Literal
from PIL import Image
from functools import cached_property
from src.biliary_seg.data.vision.masks import get_slide_bg_mask
from src.biliary_seg.data.vision.patching import get_subimage

Image.MAX_IMAGE_PIXELS = None


class ImageLoader:

    def __init__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ):
        self.image: np.ndarray = image
        self.mask: np.ndarray = (
            mask if mask is not None else np.zeros(image.shape[:2], dtype=bool)
        )
    
    @staticmethod
    def from_data_dir(id: str, data_dir: str) -> "ImageLoader":
        image = ImageLoader._load_image(id, data_dir)
        try:
            mask = ImageLoader._load_mask(id, data_dir)
        except FileNotFoundError:
            mask = None
        return ImageLoader(image, mask)

    @staticmethod
    def _load_image(id: str, data_dir: str):
        path = os.path.join(data_dir, "images", f"{id}.png")
        image = np.array(Image.open(path), dtype=np.uint8)
        return image

    @staticmethod
    def _load_mask(id: str, data_dir: str):
        path = os.path.join(data_dir, "masks", f"{id}.npy")
        mask = np.load(path)
        return mask.astype(bool)
    
    @cached_property
    def background_mask(self):
        return get_slide_bg_mask(self.image)

    @cached_property
    def tissue_mask(self):
        return ~self.background_mask

    @cached_property
    def mask_coords(self):
        return np.where(self.mask)

    @cached_property
    def contains_biliary_populations(self):
        return self.mask.sum() > 0
    
    def random_sample(
        self,
        image_size: Tuple[int, int] = (512, 512),
        require_mask: bool = False,
        augmentation: Optional[Any] = None,
        variance_range: Optional[Tuple[float, float]] = (0.75, 1.25),
        context_factor: Optional[float] = None,
    ):

        image_h, image_w = self.image.shape[:2]

        if variance_range is None:
            sampled_var = 1.0
        else:
            sampled_var = np.random.uniform(*variance_range)
        sample_h, sample_w = int(image_size[0] * sampled_var), int(
            image_size[1] * sampled_var
        )

        if require_mask and self.mask_coords is not None:
            rand_index = np.random.randint(0, len(self.mask_coords[0]))
            # Generate random offsets to ensure the sample is centered around a mask pixel
            # This also ensures that the sample is not too close to the edges
            delta_top = np.random.randint(0, np.clip(int(sample_h * 0.9), 0, image_h))
            delta_left = np.random.randint(0, np.clip(int(sample_w * 0.9), 0, image_w))
            sample_top = self.mask_coords[0][rand_index] - delta_top
            sample_left = self.mask_coords[1][rand_index] - delta_left
        else:
            sample_top = np.random.randint(0, np.clip(image_h - sample_h, 1, image_h))
            sample_left = np.random.randint(0, np.clip(image_w - sample_w, 1, image_w))

        subimage = get_subimage(
            image=self.image,
            top=sample_top,
            left=sample_left,
            size=(sample_h, sample_w),
            context_factor=context_factor,
        )

        submask = get_subimage(
            image=self.mask,
            top=sample_top,
            left=sample_left,
            size=(sample_h, sample_w),
        ).squeeze()

        if augmentation is None:
            return subimage, submask

        if context_factor is None:
            augmented = augmentation(
                image=subimage,
                masks=[submask],
            )
            subimage, submask = augmented["image"], augmented["masks"][0]
        else:
            augmented = augmentation(
                image=subimage[..., :3],
                image_context=subimage[..., 3:],
                masks=[submask],
            )
            subimage = np.concatenate(
                [augmented["image"], augmented["image_context"]], axis=-1
            )
            submask = augmented["masks"][0]

        return subimage, submask