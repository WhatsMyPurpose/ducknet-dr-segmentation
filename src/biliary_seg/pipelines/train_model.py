import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional, Union
from biliary_seg.training.augmentations import training_augmentations
from biliary_seg.data import DatasetLoader
from biliary_seg.training.losses import dice_metric_loss
from biliary_seg.training.callbacks import (
    LoadBestModelOnLRChange,
    FullImageValidationCallback,
)
from biliary_seg.models.ducknet import DUCK_Net

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def load_ducknet_model(filepath: Optional[str], **kwargs) -> tf.keras.Model:
    model = DUCK_Net.create_model(**kwargs)
    if filepath:
        model.load_weights(filepath)
    return model


def train_model(
    data_dir: str,
    model: Optional[Union[tf.keras.Model, str]] = None,
    image_size: Tuple[int, int] = (512, 512),
    batch_size: int = 8,
    epochs: int = 100,
    steps_per_epoch: int = 2000,
    learning_rate: float = 1e-4,
    ducknet_filters: int = 17,
    checkpoint_dir: str = "./checkpoints",
):
    # Load/initialize model if not provided
    if model is None or isinstance(model, str):
        model = load_ducknet_model(
            filepath=model if isinstance(model, str) else None,
            img_height=image_size[0],
            img_width=image_size[1],
            input_chanels=3,
            out_classes=1,
            starting_filters=ducknet_filters,
        )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=dice_metric_loss)

    # Load dataset
    dataset = DatasetLoader.from_data_dir(data_dir)

    train_gen = dataset["train"].sample_generator(
        batch_size=batch_size,
        image_size=image_size,
        augmentation=training_augmentations,
    )

    # Prepare callbacks
    best_model_checkpoint = Path(checkpoint_dir) / "best_model.h5"
    full_image_callback = FullImageValidationCallback(
        image_size=image_size, batch_size=4, dataset=dataset["validation"]
    )
    load_best_model_callback = LoadBestModelOnLRChange(
        filepath=str(best_model_checkpoint)
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="full_image_dice_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1,
    )
    save_best_model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_checkpoint),
        monitor="full_image_dice_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=str(Path(checkpoint_dir) / "logs"),
        histogram_freq=1,
    )

    # Train the model
    model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[
            reduce_lr,
            full_image_callback,
            save_best_model_callback,
            load_best_model_callback,
            tensorboard_callback,
        ],
    )

    return model


if __name__ == "__main__":
    trained_model = train_model(
        data_dir="./data/biliary_segmentation",
        model=None,
        image_size=(512, 512),
        batch_size=8,
        epochs=100,
        steps_per_epoch=2000,
        learning_rate=1e-4,
        ducknet_filters=17,
        checkpoint_dir="./checkpoints",
    )
