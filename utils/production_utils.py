import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import rasterio
import requests
import datetime

import torch
import torch.nn.functional as F
from PIL import Image
from itertools import product
from time import time
import tifffile as tiff

if __name__ == "__main__":
    sys.path.append(os.getcwd())

from inference import predict_image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


# ===========================================
# ==== DOWNLOADING TILES ====================
# =========================================== 
def download_tile(E, N, dest, suffixe=''):
    year = datetime.date.today().year
    url_img = f"https://data.geo.admin.ch/ch.swisstopo.swissimage-dop10/swissimage-dop10_{year}_{E}-{N}/swissimage-dop10_{year}_{E}-{N}_0.1_2056.tif"
    count_down = 0
    while requests.get(url_img).status_code != 200:
        if count_down > 10:
            print(f"Could not find a tile for coordinates {E}-{N}")
            return
        year -= 1
        count_down += 1
        url_img = f"https://data.geo.admin.ch/ch.swisstopo.swissimage-dop10/swissimage-dop10_{year}_{E}-{N}/swissimage-dop10_{year}_{E}-{N}_0.1_2056.tif"

    img_data = requests.get(url_img).content
    if suffixe != '':
        file_src = os.path.join(dest, f"tile_{E}-{N}_{year}_{suffixe}.tif")
    else:
        file_src = os.path.join(dest, f"tile_{E}-{N}_{year}.tif")
    with open(file_src, 'wb') as handler:
        handler.write(img_data)
    return file_src


# ===========================================
# ==== PREDICTIONS ===========================
# ===========================================
def mirror_pad_image(img, tile_size, stride):
    """
    img: (H, W, C)
    """
    H, W = img.shape[:2]

    pad_h = (stride - (H - tile_size) % stride) % stride
    pad_w = (stride - (W - tile_size) % stride) % stride

    padded = np.pad(
        img,
        ((0, pad_h),
         (0, pad_w),
         (0, 0)),
        mode="reflect"
    )

    return padded, (pad_h, pad_w), (H, W)


def load_latest_checkpoint(model_dir, verbose=False):
    """
    Returns the path to the latest checkpoint folder inside model_dir.
    If none found, return model_dir (trained_model directory).
    """
    if not os.path.isdir(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    ckpts = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
    if not ckpts and verbose:
        print("[INFO] No checkpoints found. Using main model directory.")
        return model_dir

    # Sort by step number
    ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
    last_ckpt = ckpts_sorted[-1]
    if verbose:
        print(f"[INFO] Using checkpoint: {last_ckpt}")
    return os.path.join(model_dir, last_ckpt)


def gaussian_weight(size, sigma=0.125):
    ax = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(ax, ax)
    return np.exp(-(xx**2 + yy**2) / (2 * sigma**2))


def predict(image, model_dir, img_path=None, tile_size=512, stride=256, th=0.5, output_format='png', do_show=True, do_save=True, do_save_mask_as_img=True):
    if not isinstance(image, Image.Image):
        img_path = image
        image = Image.open(image)
    img_arr = np.array(image)

    img_padded, _, _ = mirror_pad_image(img_arr, tile_size, stride)
    H_original, W_original  = img_arr.shape[:2]
    H, W = img_padded.shape[:2]
    
    # prepare arrays
    prob_acc = np.zeros((H,W), dtype=np.float32)
    weight_acc = np.zeros((H,W), dtype=np.float32)
    weights = gaussian_weight(tile_size)

    # load model
    ckpt_path = load_latest_checkpoint(model_dir)
    processor = AutoImageProcessor.from_pretrained(ckpt_path)
    model = SegformerForSemanticSegmentation.from_pretrained(ckpt_path)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(DEVICE)
    model.eval()
    time_to_predict = 0
    for x in range(0, W - tile_size + 1, stride):# manage if borders reached
        for y in range(0, H - tile_size + 1, stride):
            x0 = min(x, W - tile_size)
            y0 = min(y, H - tile_size)

            # Crop region (handles border tiles automatically)
            tile = img_padded[y0:y0 + tile_size, x0:x0 + tile_size, :]
            tile_PIL = Image.fromarray(tile).convert("RGB")
            dt = time()
            _, logits  = predict_image(model, processor, tile_PIL, DEVICE)
            time_to_predict += time() - dt
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            landslide_prob = prob[:, 1].reshape((tile_size, tile_size))

            # prob_acc[y:y+tile_size, x:x+tile_size] += landslide_prob * weights
            # weight_acc[y:y+tile_size, x:x+tile_size] += weights
            prob_acc[y0:y0+tile_size, x0:x0+tile_size] += landslide_prob
            weight_acc[y0:y0+tile_size, x0:x0+tile_size] += 1

    print("TIME USED BY THE MODEL: ", time_to_predict)
    full_prob = prob_acc / np.maximum(weight_acc, 1e-6)
    final_prob = full_prob[0:H_original, 0:W_original]
    
    final_labels = np.zeros(final_prob.shape)
    final_labels[final_prob >= th] = 1

    src_dest_preds_mask = os.path.splitext(img_path)[0] + f'_preds_mask.{output_format}'
    src_dest_preds_img = os.path.splitext(img_path)[0] + f'_preds_img.{output_format}'

    rgb_labels = np.zeros((final_labels.shape[0], final_labels.shape[1], 3))
    rgb_labels[final_labels == 1] = 255
    if do_save:
        os.makedirs(os.path.dirname(src_dest_preds_mask), exist_ok=True)
        Image.fromarray(final_labels.astype(np.uint8), mode='L').save(src_dest_preds_mask)
        if do_save_mask_as_img:
            Image.fromarray(rgb_labels.astype(np.uint8), mode='RGB').save(src_dest_preds_img)
    
    if do_show:
        plt.imshow(Image.fromarray(rgb_labels.astype(np.uint8), mode="RGB"))

    return final_labels, rgb_labels, final_prob


def predict_batch_array(
    model,
    batch,
    device="cuda",
):
    """
    Parameters
    ----------
    batch_images : np.ndarray
        Shape (B, H, W, 3), RGB images in RAM

    Returns
    -------
    pred_masks : np.ndarray
        Shape (B, H, W)
    upsampled_logits : torch.Tensor
        Shape (B, C, H, W) on CPU
    """

    assert batch.ndim == 4 and batch.shape[-1] == 3

    _, H, W, _ = batch.shape
    batch = batch.permute(0, 3, 1, 2)                  # (B, 3, H, W)
    batch = batch.float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    batch = (batch - mean) / std

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        logits = model(batch).logits  # (B,C,h,w)

        logits = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

    return logits  # keep on GPU


def predict_with_batch(image, model, img_path=None, batch_size=8, tile_size=512, stride=256, th=0.5, do_show=True, do_save=True, do_save_mask_as_img=True):
    if not isinstance(image, Image.Image):
        img_path = image
        image = Image.open(image)
    img_arr = np.array(image)

    img_padded, _, _ = mirror_pad_image(img_arr, tile_size, stride)
    H_original, W_original  = img_arr.shape[:2]
    H, W = img_padded.shape[:2]
    
    # prepare arrays
    prob_acc = torch.zeros((H,W), device=model.device)
    weight_acc = torch.zeros((H,W), device=model.device)

    list_xpos = range(0, W - tile_size + 1, stride)
    list_ypos = range(0, H - tile_size + 1, stride)
    list_positions = list(product(list_xpos, list_ypos))

    batch = torch.zeros((batch_size, tile_size, tile_size, 3), device=model.device)
    initial_poses = []

    for id_sample, (x,y) in enumerate(list_positions):
        x0 = min(x, W - tile_size)
        y0 = min(y, H - tile_size)
        tile = torch.tensor(img_padded[y0:y0 + tile_size, x0:x0 + tile_size, :])
        batch[id_sample % batch_size, ...] = tile
        initial_poses.append((x0, y0))

        # Crop region (handles border tiles automatically)
        if (id_sample > 0 and (id_sample + 1) % batch_size == 0) or id_sample == len(list_positions) - 1:
            logits = predict_batch_array(model, batch, model.device)
            probs = torch.softmax(logits, dim=1)[:, ]

            for i in range(len(initial_poses)):
                xi, yi = initial_poses[i]
                prob_acc[yi:yi+tile_size, xi:xi+tile_size] += probs[i, 1, ...].reshape((tile_size, tile_size))
                weight_acc[yi:yi+tile_size, xi:xi+tile_size] += 1

            batch = torch.zeros((batch_size, tile_size, tile_size, 3), device=model.device)
            initial_poses = []

    final_prob = torch.divide(prob_acc, weight_acc)[0:H_original, 0:W_original].cpu().numpy()
    
    final_labels = np.zeros(final_prob.shape, dtype=np.uint8)
    final_labels[final_prob >= th] = 1

    src_dest_preds_mask = os.path.splitext(img_path)[0] + f'_preds_mask.tif'
    src_dest_preds_img = os.path.splitext(img_path)[0] + f'_preds_img.tif'

    rgb_labels = np.zeros((final_labels.shape[0], final_labels.shape[1], 3), dtype=np.uint8)
    rgb_labels[final_labels == 1] = 255
    if do_save:
        os.makedirs(os.path.dirname(src_dest_preds_mask), exist_ok=True)
        tiff.imwrite(src_dest_preds_mask, final_labels, compression="zstd", compressionargs={"level": 9})
        if do_save_mask_as_img:
            tiff.imwrite(src_dest_preds_img, rgb_labels, compression="zstd", compressionargs={"level": 9})
    
    if do_show:
        plt.imshow(Image.fromarray(rgb_labels.astype(np.uint8), mode="RGB"))

    return final_labels, rgb_labels, final_prob


def produce_with_lower_res(src_img, src_dest, res_frac, do_save=True, do_show=True):
    img = Image.open(src_img)
    res_original = img.size
    low_res = tuple([int(x * res_frac) for x in res_original])

    img_low = img.resize((low_res), resample=Image.BILINEAR)
    src_final = os.path.join(src_dest, os.path.splitext(os.path.basename(src_img))[0] + f'_res_{res_frac}.tif')
    
    
    if do_save:
        tiff.imwrite(src_final, img_low, compression="zstd", compressionargs={"level": 9})
    if do_show:
        plt.imshow(img_low)
    
    return img_low, src_final


def prob_to_rgb(prob_map, cmap_name="viridis"):
    """
    prob_map: (H, W) float32 in [0, 1]
    returns: (H, W, 3) uint8
    """
    cmap = cm.get_cmap(cmap_name)

    # Apply colormap → RGBA in [0,1]
    rgba = cmap(prob_map)

    # Drop alpha channel and convert to uint8
    rgb = (rgba[..., :3] * 255).astype(np.uint8)

    return rgb


# ===========================================
# ==== VECTORIZATION ========================
# ===========================================

def geo_transfert(img_geo, img_target, same_file=True):
    with rasterio.open(img_geo) as src:
        crs = src.crs
        transform = src.transform

    with rasterio.open(img_target) as pred:
        pred_data = pred.read()
        pred_profile = pred.profile
        
    pred_profile.update({
        "crs": crs,
        "transform": transform
    })

    src_new_target = os.path.splitext(img_target)[0] +"_georef.tif" if not same_file else img_target

    with rasterio.open(src_new_target, "w", **pred_profile) as dst:
        dst.write(pred_data)
    return src_new_target


# if __name__ == "__main__":
#     print(datetime.date.today().year)

