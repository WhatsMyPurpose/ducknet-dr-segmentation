import os
import numpy as np
import pandas as pd
import albumentations as A
from typing import Optional, Literal
from biliary_seg.data.image_loader import ImageLoader
from biliary_seg.data.patching import window_generator, insert_subimage
from biliary_seg.training.metrics import compute_metrics
from typing import Iterator, Tuple, Dict, Any, Literal


class Dataset:

    def __init__(
        self,
        image_loaders: list[ImageLoader],
        image_weights: Optional[list[float]] = None,
    ):
        self.image_loaders = image_loaders
        self.image_weights = (
            image_weights
            if image_weights is not None
            else [1.0 / len(image_loaders)] * len(image_loaders)
        )

    def sample_generator(
        self,
        batch_size: int,
        image_size: tuple[int, int],
        augmentation: Optional[A.Compose] = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generator yielding image/mask batches.

        Args:
            batch_size: Number of samples per batch.
            image_size: Size of the input images for the model.
            augmentation: Optional albumentations augmentation to apply.
        """
        while True:
            x_batch = []
            y_batch = []
            for i in range(batch_size):
                image_loader: ImageLoader = np.random.choice(
                    self.image_loaders, p=self.image_weights
                )
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

    def evaluate_model(
        self, model, batch_size: int, image_size: tuple[int, int]
    ) -> tuple[float, dict]:
        """Evaluate the model on the full images.

        Args:
            model: The trained segmentation model.
            batch_size: Batch size for prediction.
            image_size: Size of the input images for the model.
        """
        tp, fp, tn, fn = [0] * 4

        for image_loader in self.image_loaders:
            pred_mask = np.zeros_like(image_loader.mask, dtype=bool)
            val_window_generator = window_generator(
                image_loader.image, window_size=image_size, window_overlap=0.35
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
            y_true = np.squeeze(image_loader.mask).astype(bool)
            y_pred = np.squeeze(pred_mask).astype(bool)

            tp += np.sum(y_true & y_pred)
            fp += np.sum(~y_true & y_pred)
            tn += np.sum(~y_true & ~y_pred)
            fn += np.sum(y_true & ~y_pred)

        metrics = compute_metrics(tp, fp, tn, fn)

        return 1 - metrics["Dice"], metrics


class DatasetLoader:

    @staticmethod
    def _compute_weights_by_tissue(image_loaders: list[ImageLoader]) -> list[float]:
        """Compute sampling weights based on tissue area in each image.

        Args:
            image_loaders: List of ImageLoader objects.
        """
        tissue_sums = np.array(
            [image_loader.mask.sum() for image_loader in image_loaders],
            dtype=np.float32,
        )
        total_tissue = tissue_sums.sum()
        if total_tissue == 0:
            raise ValueError("No tissue found in any images to compute weights.")
        return (tissue_sums / total_tissue).tolist()

    @staticmethod
    def _split_image(
        image_loader: ImageLoader,
        threshold: float,
        by: Literal["tissue", "slide"] = "tissue",
    ) -> tuple[ImageLoader, ImageLoader]:
        """Split an ImageLoader into two based on a threshold.

        Args:
            image_loader: The ImageLoader to split.
            threshold: The threshold for splitting.
            by: Method of splitting, either by "tissue" or "slide".
        """

        if by == "tissue":
            tissue_total = (image_loader.tissue_mask / 255).sum()
            tissue_cumulative_col = (image_loader.tissue_mask / 255).sum(
                axis=1
            ).cumsum() / tissue_total
            row_cutoff = np.argmax(tissue_cumulative_col >= threshold)
        else:
            row_cutoff = int(image_loader.image.shape[0] * threshold)

        return (
            ImageLoader(
                image_loader.id,
                image_loader.image[:row_cutoff, :, :],
                image_loader.mask[:row_cutoff, :],
            ),
            ImageLoader(
                image_loader.id,
                image_loader.image[row_cutoff:, :, :],
                image_loader.mask[row_cutoff:, :],
            ),
        )

    @staticmethod
    def _compute_tissue_br_weighted_total(
        image_loader: ImageLoader, weight: float
    ) -> float:
        """Compute a weighted total of tissue and background pixels for an ImageLoader."""
        tissue_sum = image_loader.tissue_mask.sum()
        background_sum = image_loader.background_mask.sum()
        return tissue_sum * weight + background_sum * (1 - weight)

    @staticmethod
    def from_data_dir(
        data_dir: str, tissue_weight: float = 0.7
    ) -> Dict[Literal["train", "validation", "test"], Dataset]:
        """Load dataset from a directory containing images, masks, and metadata."""

        df = pd.read_csv(
            os.path.join(data_dir, "metadata.csv"),
            usecols=["id", "Train Weights %", "Validation Weights %", "Test Weights %"],
        )
        train_images, test_images, val_images = [], [], []
        train_weights = []
        for _, row in df.iterrows():
            image_loader = ImageLoader.from_data_dir(row["id"], data_dir)

            train_pct, val_pct, test_pct = (
                row["Train Weights %"],
                row["Validation Weights %"],
                row["Test Weights %"],
            )
            if train_pct == 1:
                train_images.append(image_loader)
                train_weights.append(
                    DatasetLoader._compute_tissue_br_weighted_total(
                        image_loader, weight=tissue_weight
                    )
                )
            elif val_pct == 1:
                val_images.append(image_loader)
            elif test_pct == 1:
                test_images.append(image_loader)
            else:
                # Assumes only train/val images are split
                train_image, val_image = DatasetLoader._split_image(
                    image_loader, threshold=train_pct, by="tissue"
                )
                train_images.append(train_image)
                train_weights.append(
                    DatasetLoader._compute_tissue_br_weighted_total(
                        train_image, weight=tissue_weight
                    )
                )
                val_images.append(val_image)

        # Normalize train weights
        total_train_weight = np.sum(train_weights)
        if total_train_weight > 0:
            train_weights = np.array([w / total_train_weight for w in train_weights])

        return {
            "train": Dataset(train_images, train_weights),
            "validation": Dataset(val_images),
            "test": Dataset(test_images),
        }
