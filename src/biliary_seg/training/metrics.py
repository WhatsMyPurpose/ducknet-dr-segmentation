def get_precision(tp, fp, tn, fn):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def get_recall(tp, fp, tn, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def get_specificity(tp, fp, tn, fn):
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def get_accuracy(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    return (tp + tn) / total if total > 0 else 0.0

def get_iou(tp, fp, tn, fn):
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

def get_dice(tp, fp, tn, fn):
    denominator = (2 * tp + fp + fn)
    return 2 * tp / denominator if denominator > 0 else 0.0

def compute_metrics(tp, fp, tn, fn):
    """Computes all metrics of interest given TP, FP, TN, FN."""
    
    metric_functions = {
        "Precision": get_precision,
        "Recall": get_recall,
        "Specificity": get_specificity,
        "Accuracy": get_accuracy,
        "IoU": get_iou,
        "Dice": get_dice
    }
    # Compute each metric
    results = {name: func(tp, fp, tn, fn) for name, func in metric_functions.items()}
    
    # Add the raw counts back in
    results.update({"TP": tp, "FP": fp, "TN": tn, "FN": fn})
    
    return results