import keras.backend as K
import tensorflow as tf


def dice_metric_loss(ground_truth, predictions, smooth=1e-6):
    """Calculate Dice loss between ground truth and predictions.
    
    Args:
        ground_truth: Ground truth tensor.
        predictions: Predicted tensor.
        smooth: Smoothing factor to avoid division by zero.
    """
    ground_truth = K.cast(ground_truth, tf.float32)
    predictions = K.cast(predictions, tf.float32)
    ground_truth = K.flatten(ground_truth)
    predictions = K.flatten(predictions)
    intersection = K.sum(predictions * ground_truth)
    union = K.sum(predictions) + K.sum(ground_truth)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice


def combined_dice_bce_loss(ground_truth, predictions, dice_weight=0.5):
    """Combine Dice loss and Binary Cross-Entropy loss.
    
    Args:
        ground_truth: Ground truth tensor.
        predictions: Predicted tensor.
        dice_weight: Weight for the Dice loss component.
    """
    dice = dice_metric_loss(ground_truth, predictions)
    bce = tf.keras.losses.binary_crossentropy(ground_truth, predictions)
    return dice_weight * dice + (1 - dice_weight) * bce
