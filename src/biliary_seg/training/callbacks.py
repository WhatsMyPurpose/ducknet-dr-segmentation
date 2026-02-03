import tensorflow as tf
from biliary_seg.data.dataset_loader import Dataset


class LoadBestModelOnLRChange(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.best_weights = None
        self.last_lr = None

    def on_epoch_end(self, epoch, logs=None):
        """Load best model weights if learning rate has decreased."""
        current_lr = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        )
        if self.last_lr is not None and current_lr < self.last_lr:
            print(
                f"\nLoading best model due to LR change: {self.last_lr} -> {current_lr}"
            )
            self.model.load_weights(self.filepath)
        self.last_lr = current_lr


class FullImageValidationCallback(tf.keras.callbacks.Callback):

    def __init__(self, image_size, batch_size, dataset: Dataset):
        super().__init__()
        self.image_size = image_size
        self.batch_size = batch_size
        self.best_dice = 1.0
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        """Evaluate model on full images at the end of each epoch."""
        dice_loss, metrics = self.dataset.validation_evaluator(
            self.model, self.image_size, self.batch_size
        )
        
        metric_logs = {f"val_full_{k.lower()}": v for k, v in metrics.items()}
        metric_logs["val_full_dice_loss"] = dice_loss
        logs.update(metric_logs)

        # Update best dice if improved
        if dice_loss < self.best_dice:
            self.best_dice = dice_loss
        
        print(
            f"\n[Epoch {epoch+1}] Full Image Eval: "
            f"Dice Loss: {dice_loss:.4f} | "
            f"Precision: {metrics.get('Precision', 0):.4f} | "
            f"Recall: {metrics.get('Recall', 0):.4f} | "
            f"Specificity: {metrics.get('Specificity', 0):.4f}"
        )