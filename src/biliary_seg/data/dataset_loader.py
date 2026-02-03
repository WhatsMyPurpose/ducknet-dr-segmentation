
import numpy as np
import albumentations as A
from typing import Optional, Literal
from biliary_seg.data.image_loader import ImageLoader
from biliary_seg.data.patching import window_generator, insert_subimage
from biliary_seg.training.metrics import compute_metrics

class Dataset:
    
    def __init__(
        self,
        train_images: list[ImageLoader],
        val_images: list[ImageLoader],
        train_weights: Optional[list[float]] = None,
    ):   
        self.train_images = train_images
        self.val_images = val_images
        self.train_weights = train_weights if train_weights is not None else [1.0 / len(train_images)] * len(train_images)
    
    def train_generator(
        self,
        batch_size: int,
        image_size: tuple[int, int],
        augmentation: Optional[A.Compose] = None,
    ):
        """Generator yielding training batches.
        
        Args:
            batch_size: Number of samples per batch.
            image_size: Size of the input images for the model.
            augmentation: Optional albumentations augmentation to apply.
        """
        while True:
            x_batch = []
            y_batch = []
            for i in range(batch_size):
                image_loader: ImageLoader = np.random.choice(self.train_images, p=self.train_weights)
                x, _, y = image_loader.random_sample(
                    image_size=image_size,
                    require_mask=(i < batch_size // 2),
                    augmentation=augmentation,
                    variance_range=None,
                )
                x_batch.append(x)
                y_batch.append(y)
            x_batch = np.array(x_batch, dtype=np.float32) / 255.0
            y_batch = np.array(y_batch, dtype=np.float32)
            yield x_batch, y_batch
    
    def validation_evaluator(
        self,
        model,
        batch_size: int,
        image_size: tuple[int, int]
    ) -> tuple[float, dict]:
        """Evaluate the model on the full validation images.
        
        Args:
            model: The trained segmentation model.
            batch_size: Batch size for prediction.
            image_size: Size of the input images for the model.
        """
        tp, fp, tn, fn = [0] * 4
        
        for val_image in self.val_images:
            pred_mask = np.zeros_like(val_image.mask, dtype=bool)
            val_window_generator = window_generator(
                val_image.image, window_size=image_size, window_overlap=0.35
            )
            batch_windows = []
            batch_coords = []
            for window, (top, left) in val_window_generator:
                batch_windows.append(window[np.newaxis, ...] / 255.0)
                batch_coords.append((top, left))

                if len(batch_windows) == batch_size:
                    batch_windows = np.vstack(batch_windows)
                    preds = model.predict(batch_windows, verbose=0)

                    for pred, (top, left) in zip(preds, batch_coords):
                        pred_mask = insert_subimage(
                            pred_mask, pred > 0.5, top, left, operation="or"
                        )

                    batch_windows = []
                    batch_coords = []

            # Process any remaining windows in the batch
            if batch_windows:
                batch_windows = np.vstack(batch_windows)
                preds = model.predict(batch_windows)

                for pred, (top, left) in zip(preds, batch_coords):
                    pred_mask = insert_subimage(
                        pred_mask, pred > 0.5, top, left, operation="or"
                    )
            
            # Calculate metrics
            y_true = np.squeeze(val_image.val_mask).astype(bool)
            y_pred = np.squeeze(pred_mask).astype(bool)

            tp += np.sum(y_true & y_pred)
            fp += np.sum(~y_true & y_pred)
            tn += np.sum(~y_true & ~y_pred)
            fn += np.sum(y_true & ~y_pred)

        metrics = compute_metrics(tp, fp, tn, fn)

        return 1 - metrics["Dice"], metrics

class DatasetLoader:
    
    @staticmethod
    def _compute_weights_by_tissue(images: list[ImageLoader]) -> list[float]:
        """Compute sampling weights based on tissue area in each image.
        
        Args:
            images: List of ImageLoader objects.
        """
        tissue_sums = np.array([img.mask.sum() for img in images], dtype=np.float32)
        total_tissue = tissue_sums.sum()
        if total_tissue == 0:
            raise ValueError("No tissue found in any images to compute weights.")
        return (tissue_sums / total_tissue).tolist()
    
    @staticmethod
    def _split_image(
        image: ImageLoader,
        threshold: float,
        by: Literal["tissue", "slide"] = "tissue"
    ) -> tuple[ImageLoader, ImageLoader]:
        """Split an ImageLoader into two based on a threshold.
        
        Args:
            image: The ImageLoader to split.
            threshold: The threshold for splitting.
            by: Method of splitting, either by "tissue" or "slide".
        """
        
        if by == "tissue":
            tissue_total = (image.tissue_mask / 255).sum()
            tissue_cumulative_col = (image.tissue_mask / 255).sum(
                axis=1
            ).cumsum() / tissue_total
            row_cutoff = np.argmax(tissue_cumulative_col >= threshold)
        else:
            row_cutoff = int(image.image.shape[0] * threshold)
        
        return (
            ImageLoader(
                image.image[:row_cutoff, :, :],
                image.mask[:row_cutoff, :],
            ),
            ImageLoader(
                image.image[row_cutoff:, :, :],
                image.mask[row_cutoff:, :],
            ),
        )
        
    
    @staticmethod
    def from_data_dir(data_dir: str):
        # Implementation to load dataset from directory
        pass