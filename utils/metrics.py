import numpy as np
import torch


def compute_iou(preds, labels, num_classes):
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
    pred: (N, H, W) predicted class indices
    label: (N, H, W) ground-truth class indices
    num_classes: int
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
    pred: (N, H, W) predicted class indices
    label: (N, H, W) ground-truth class indices
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
        Compute per-class IoU and mean IoU for semantic segmentation predictions.

        p.predictions: (batch_size, num_classes, H_pred, W_pred)
        p.label_ids:   (batch_size, H_lbl, W_lbl)
        """
        if isinstance(p, dict):
            preds = p['predictions']  # raw logits
            labels = p['label_ids']   # ground-truth labels
        else:
            preds = p.predictions  # raw logits
            labels = p.label_ids   # ground-truth labels

        # Resize predictions to match label size
        logits = torch.nn.functional.interpolate(
            torch.tensor(preds),
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
    Computes a confusion matrix for segmentation (H×W arrays).

    y_true : ndarray of shape (H, W)
    y_pred : ndarray of shape (H, W)
    num_classes : int, number of classes

    Returns: ndarray (num_classes, num_classes)
    """
    # mask = (y_true >= 0) & (y_true < num_classes)  # valid pixels only
    # y_true = y_true[mask].flatten()
    # y_pred = y_pred[mask].flatten()
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    cm = np.bincount(
        num_classes * y_true + y_pred,
        minlength=num_classes**2
    ).reshape(num_classes, num_classes).astype(np.uint64)

    return cm


def compute_cm_from_dict(dict_preds_lbls, num_classes=2):
    cf = np.zeros((len(dict_preds_lbls), num_classes, num_classes), dtype=np.uint64)

    for id_val, (preds, labels) in enumerate(dict_preds_lbls.values()):
        cf[id_val, :,:] = confusion_matrix_numpy(labels, preds, 2)

    cf = np.sum(cf, axis=0, dtype=np.uint64)

    return cf
