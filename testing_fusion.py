import os
import torch
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
import pickle
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from torchvision.transforms.functional import to_pil_image

from utils.metrics import compute_metrics, compute_cm_from_dict
from utils.visualization import show_confusion_matrix
from utils.trainer import MultiScaleSegformer


def load_latest_checkpoint(model_dir):
    """
    Returns the path to the latest checkpoint folder inside model_dir.
    If none found, return model_dir (trained_model directory).
    """
    if not os.path.isdir(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    ckpts = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
    if not ckpts:
        print("[INFO] No checkpoints found. Using main model directory.")
        return model_dir

    # Sort by step number
    ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
    last_ckpt = ckpts_sorted[-1]

    print(f"[INFO] Using checkpoint: {last_ckpt}")
    return os.path.join(model_dir, last_ckpt)


def predict_image(model, processor, image_path, device="cuda"):
    """
    Runs inference on a single image and returns:
    - predicted_mask (H, W) with class indices
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Preprocess
    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, num_classes, h, w)

    # Resize logits to original size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False
    )

    pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    return pred_mask, logits


def dict_to_list(_dict):
    res_list = []
    col_names = ['filename']
    for id, (key, val) in enumerate(_dict.items()):
        sub_list = [key]
        for sub_val in val.values():
            sub_list.append(sub_val)
        res_list.append(sub_list)

        if id == 0:
            for key in val.keys():
                col_names.append(key)

    return res_list, col_names


def run_testing(conf):
    """
    Runs inference on all images in a directory.
    """
    # ----------------------------
    # Load parameters
    # ----------------------------
    MODEL_DIR = conf.model_dir
    DATA_DIR = conf.data_dir
    OUTPUT_DIR = conf.output_dir
    OUTPUT_SUFFIXE = conf.output_suffixe
    SAVE_MASK_AS_IMG = conf.save_mask_as_img
    SAVE_PREDS_LBLS = conf.save_preds_lbls
    NUM_SAMPLES_PER_BUFFER = int(conf.num_samples_per_buffer)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----------------------------
    # Load best checkpoint
    # ----------------------------
    ckpt_path = load_latest_checkpoint(MODEL_DIR)

    processor = AutoImageProcessor.from_pretrained(ckpt_path)
    model = MultiScaleSegformer.from_pretrained(ckpt_path)
    # model = SegformerForSemanticSegmentation.from_pretrained(ckpt_path)
    model.to(DEVICE)
    model.eval()

    RESULTS_DIR = os.path.join(OUTPUT_DIR, datetime.now().strftime(r"%Y%m%d_%H%M%S_") + f"testing_" + OUTPUT_SUFFIXE)
    OUTPUT_DIR_PRED = os.path.join(RESULTS_DIR, "predictions")
    OUTPUT_DIR_PRED_AS_IMG = os.path.join(OUTPUT_DIR_PRED, "images") if SAVE_MASK_AS_IMG else None
    PERF_DIR = os.path.join(RESULTS_DIR, "metrics")
    CONF_MATS_DIR = os.path.join(PERF_DIR, "conf_mats")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_PRED_AS_IMG, exist_ok=True)
    os.makedirs(PERF_DIR, exist_ok=True)
    if SAVE_PREDS_LBLS:
        os.makedirs(CONF_MATS_DIR, exist_ok=True)

    # ----------------------------
    # Loop over images
    # ----------------------------
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    image_list = [f for f in os.listdir(os.path.join(DATA_DIR, "images")) if f.lower().endswith(exts) and f in os.listdir(os.path.join(DATA_DIR, 'masks'))]

    if not image_list:
        print("No images found in:", DATA_DIR)
        return

    print(f"[INFO] Running inference on {len(image_list)} images")

    dict_conf_mat = {}
    conf_mat = np.zeros((2,2), dtype=np.uint64)
    conf_mat_cmpt = 0
    dict_metrics = {}
    for cmpt, img_name in tqdm(enumerate(image_list, start=1), total=len(image_list), desc="Computing"):
        input_path_img = os.path.join(DATA_DIR, 'images', img_name)
        input_path_mask = os.path.join(DATA_DIR, 'masks', img_name)
        output_path = os.path.join(OUTPUT_DIR_PRED, ''.join(img_name.split('.')[:-1]) + ".tif")

        preds, logits = predict_image(model, processor, input_path_img, device=DEVICE)
        preds = preds.astype(np.uint8)
        labels = np.asarray(Image.open(input_path_mask))

        # Save preds
        pil_mask = Image.fromarray(preds.astype(np.uint8))
        pil_mask.save(output_path)

        if SAVE_MASK_AS_IMG:
            rgb_mask = np.zeros((preds.shape[0], preds.shape[1], 3))
            rgb_mask[preds == 1] = 255
            pil_rgb_mask = Image.fromarray(rgb_mask.astype(np.uint8))
            pil_rgb_mask.save(os.path.join(OUTPUT_DIR_PRED_AS_IMG, img_name))
        
        # Compute perf and metrics
        dict_conf_mat[img_name] = (preds, labels)
        metrics = compute_metrics({'predictions': logits, 'label_ids': labels})
        dict_metrics[img_name] = metrics

        if cmpt % NUM_SAMPLES_PER_BUFFER == 0:
            conf_mat += compute_cm_from_dict(dict_conf_mat).astype(np.uint64)

            # Save confmat data
            if SAVE_PREDS_LBLS:
                with open(os.path.join(CONF_MATS_DIR, f'conf_mat_{conf_mat_cmpt}.pickle'), 'wb') as f:
                    pickle.dump(dict_conf_mat, f)

            conf_mat_cmpt += 1
            dict_conf_mat.clear()

    # Save confmat data
    if SAVE_PREDS_LBLS:
        with open(os.path.join(CONF_MATS_DIR, f'conf_mat_{conf_mat_cmpt}.pickle'), 'wb') as f:
            pickle.dump(dict_conf_mat, f)
        dict_conf_mat.clear()
    
    # Save perf and metrics
    lst_metrics, col_names = dict_to_list(dict_metrics)
    df_metrics = pd.DataFrame(lst_metrics, columns=col_names)
    df_confmat = pd.DataFrame(conf_mat, index=[0,1], columns=[0,1])
    df_confmat.to_csv(os.path.join(PERF_DIR, 'metrics.csv'), sep=';', index=False)
    df_metrics.to_csv(os.path.join(PERF_DIR, 'metrics.csv'), sep=';', index=False)
    df_metrics[[x for x in df_metrics.columns.values if x != 'filename']].mean().to_csv(os.path.join(PERF_DIR, 'metrics_mean.csv'), sep=';', header=['mean'])

    # Show confusion matrix
    sum_for_recall = np.sum(conf_mat, axis=1).reshape(-1, 1)
    sum_for_precision = np.sum(conf_mat, axis=0).reshape(1, -1)
    show_confusion_matrix(os.path.join(PERF_DIR, 'confusion_matrix.png'), conf_mat, ['Background', 'Landslide'])
    show_confusion_matrix(os.path.join(PERF_DIR, 'confusion_matrix_recall.png'), conf_mat / sum_for_recall, ['Background', 'Landslide'], "Confusion Matrix - Producer accuracy")
    show_confusion_matrix(os.path.join(PERF_DIR, 'confusion_matrix_precision.png'), conf_mat / sum_for_precision, ['Background', 'Landslide'], "Confusion Matrix - User accuracy")


if __name__ == "__main__":
    conf = OmegaConf.load('config/testing.yaml')

    run_testing(conf)
