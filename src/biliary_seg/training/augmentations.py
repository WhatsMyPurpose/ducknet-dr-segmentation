# hed_aug.py
import numpy as np
from skimage.color import rgb2hed, hed2rgb
import albumentations as A
from typing import Tuple


class RandomHEDJitter(A.ImageOnlyTransform):
    """
    Randomly perturb the Hematoxylin & Eosin channels in HED space.
    """

    def __init__(
        self,
        alpha_range=(0.7, 1.3),
        beta_range=(-0.02, 0.02),
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.alpha_range = alpha_range
        self.beta_range = beta_range

    def get_params(self):
        """Get parameters for HED jittering."""
        alphas = np.random.uniform(*self.alpha_range, size=2)
        betas = np.random.uniform(*self.beta_range, size=2)
        return {"alphas": alphas, "betas": betas}

    def apply(
        self, 
        img: np.ndarray, 
        alphas: Tuple[float, float] = (1, 1), 
        betas: Tuple[float, float] = (0, 0)
        ) -> np.ndarray:
        """Apply HED jittering to the image.
        
        Args:
            img:  Input RGB image as a numpy array.
            alphas: Scaling factors for H and E channels.
            betas: Shifting factors for H and E channels.
        """
        img_f = img.astype(np.float32) / 255.0
        hed = rgb2hed(img_f)
        # apply jitter on channels 0 (H) & 1 (E)
        hed[..., 0] = hed[..., 0] * alphas[0] + betas[0]
        hed[..., 1] = hed[..., 1] * alphas[1] + betas[1]

        rgb_out = np.clip(hed2rgb(hed), 0, 1)
        return (rgb_out * 255).astype(np.uint8)

training_augmentations = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.1),
        RandomHEDJitter(p=0.7),
    ],
    additional_targets={"image_context": "image"},
)