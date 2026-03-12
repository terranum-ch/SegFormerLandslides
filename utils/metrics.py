import numpy as np
import torch


def compute_iou(preds, labels, num_classes):
    """
    Compute per-class Intersection over Union (IoU) and mean IoU for segmentation predictions.
    Parameters: 
        preds (np.ndarray) - predicted class indices with shape (N, H, W); 
        labels (np.ndarray) - ground-truth class indices with shape (N, H, W); 
        num_classes (int) - number of segmentation classes.
    Returns: 
        dict - dictionary containing mean IoU and IoU for each class.
    """

    ious = []

    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)

        intersection = np.logical_and(pred_cls, label_cls).sum()
        union = np.logical_or(pred_cls, label_cls).sum()

        iou = np.nan if union == 0 else intersection / union
        ious.append(iou)

    ious = np.array(ious)
    mean_iou = np.nanmean(ious)
    metrics = {"mean_iou": mean_iou}
    for cls, iou in enumerate(ious):
        metrics[f"iou_class_{cls}"] = iou

    return metrics


def compute_mean_dice(pred, label, num_classes):
    """
    Compute the mean Dice coefficient across segmentation classes.
    Parameters: 
        pred (np.ndarray | torch.Tensor) - predicted class indices with shape (N, H, W); 
        label (np.ndarray | torch.Tensor) - ground-truth class indices with shape (N, H, W); 
        num_classes (int) - number of segmentation classes.
    Returns: 
        float - mean Dice coefficient across classes.
    """

    # Move to cpu + numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()

    dice_scores = []

    for c in range(num_classes):
        pred_c = (pred == c)
        label_c = (label == c)

        intersection = np.sum(pred_c & label_c)
        sum_pred = np.sum(pred_c)
        sum_label = np.sum(label_c)

        if sum_pred + sum_label == 0:
            # No pixels of this class at all → ignore this class
            continue

        dice = (2 * intersection) / (sum_pred + sum_label)
        dice_scores.append(dice)

    if len(dice_scores) == 0:
        return 0.0

    return float(np.mean(dice_scores))


def compute_pixel_accuracy(pred, label):
    """
    Compute pixel-wise accuracy between predicted and ground-truth segmentation masks.
    Parameters: 
        pred (np.ndarray | torch.Tensor) - predicted class indices with shape (N, H, W); 
        label (np.ndarray | torch.Tensor) - ground-truth class indices with shape (N, H, W).
    Returns: 
        float - pixel accuracy value.
    """

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()

    correct = np.sum(pred == label)
    total = pred.size

    return float(correct / total)


def compute_metrics(p):
    """
    Compute segmentation evaluation metrics including IoU, Dice score, and pixel accuracy.
    Parameters: 
        p (dict | EvalPrediction) - object containing predictions logits and ground-truth labels.
    Returns: 
        dict - dictionary containing mean IoU, per-class IoU, pixel accuracy, and mean Dice score.
    """

    if isinstance(p, dict):
        preds = p['predictions']  # raw logits
        labels = p['label_ids']   # ground-truth labels
    else:
        preds = p.predictions  # raw logits
        labels = p.label_ids   # ground-truth labels

    # Resize predictions to match label size
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    logits = torch.nn.functional.interpolate(
        preds,
        size=labels.shape[-2:],   # (H_lbl, W_lbl)
        mode="bilinear",
        align_corners=False
    )

    # Final predicted class mask
    pred = logits.argmax(dim=1).cpu().numpy()  # (batch_size, H_lbl, W_lbl)

    # compute ious
    metrics = compute_iou(pred, labels, num_classes=preds.shape[1])
    metrics['pa'] = compute_pixel_accuracy(pred, labels)
    metrics['mean_dice'] = compute_mean_dice(pred, labels, preds.shape[1])

    return metrics


def confusion_matrix_numpy(y_true, y_pred, num_classes):
    """
    Compute a confusion matrix for segmentation predictions using NumPy.
    Parameters: 
        y_true (np.ndarray) - ground-truth class indices with shape (H, W); 
        y_pred (np.ndarray) - predicted class indices with shape (H, W); 
        num_classes (int) - number of segmentation classes.
    Returns: 
        np.ndarray - confusion matrix with shape (num_classes, num_classes).
    """

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    cm = np.bincount(
        num_classes * y_true + y_pred,
        minlength=num_classes**2
    ).reshape(num_classes, num_classes).astype(np.uint64)

    return cm


def compute_cm_from_dict(dict_preds_lbls, num_classes=2):
    """
    Compute a global confusion matrix from a dictionary of prediction and label pairs.
    Parameters: 
        dict_preds_lbls (dict) - dictionary mapping sample identifiers to tuples of (predictions, labels); 
        num_classes (int) - number of segmentation classes.
    Returns: 
        np.ndarray - aggregated confusion matrix with shape (num_classes, num_classes).
    """

    cf = np.zeros((len(dict_preds_lbls), num_classes, num_classes), dtype=np.uint64)

    for id_val, (preds, labels) in enumerate(dict_preds_lbls.values()):
        cf[id_val, :,:] = confusion_matrix_numpy(labels, preds, 2)

    cf = np.sum(cf, axis=0, dtype=np.uint64)

    return cf
