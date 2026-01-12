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

if __name__ == "__main__":
    sys.path.append(os.getcwd())

from inference import predict_image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


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


# ===========================================
# ==== REGROUPING SAMPLES ===================
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


def predict(image, model_dir, img_path=None, tile_size=512, stride=256, th=0.5, do_show=True, do_save=True, do_save_mask_as_img=True):
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

    for x in range(0, W - tile_size + 1, stride):# manage if borders reached
        for y in range(0, H - tile_size + 1, stride):
            x0 = min(x, W - tile_size)
            y0 = min(y, H - tile_size)

            # Crop region (handles border tiles automatically)
            tile = img_padded[y0:y0 + tile_size, x0:x0 + tile_size, :]
            tile_PIL = Image.fromarray(tile).convert("RGB")
            _, logits  = predict_image(model, processor, tile_PIL, DEVICE)

            prob = torch.softmax(logits, dim=1).cpu().numpy()
            landslide_prob = prob[:, 1].reshape((tile_size, tile_size))

            # prob_acc[y:y+tile_size, x:x+tile_size] += landslide_prob * weights
            # weight_acc[y:y+tile_size, x:x+tile_size] += weights
            prob_acc[y0:y0+tile_size, x0:x0+tile_size] += landslide_prob
            weight_acc[y0:y0+tile_size, x0:x0+tile_size] += 1

    full_prob = prob_acc / np.maximum(weight_acc, 1e-6)
    final_prob = full_prob[0:H_original, 0:W_original]
    
    final_labels = np.zeros(final_prob.shape)
    final_labels[final_prob >= th] = 1

    src_dest_preds_mask = os.path.splitext(img_path)[0] + '_preds_mask.tif'
    src_dest_preds_img = os.path.splitext(img_path)[0] + '_preds_img.tif'

    rgb_labels = np.zeros((final_labels.shape[0], final_labels.shape[1], 3))
    rgb_labels[final_labels == 1] = 255
    if do_save:
        os.makedirs(os.path.dirname(src_dest_preds_mask), exist_ok=True)
        Image.fromarray(final_labels.astype(np.uint8)).save(src_dest_preds_mask)
        if do_save_mask_as_img:
            Image.fromarray(rgb_labels.astype(np.uint8)).save(src_dest_preds_img)
    
    if do_show:
        plt.imshow(Image.fromarray(rgb_labels.astype(np.uint8)))

    return final_labels, rgb_labels, final_prob


def produce_with_lower_res(src_img, src_dest, res_frac, do_save=True, do_show=True):
    img = Image.open(src_img)
    res_original = img.size
    low_res = tuple([int(x * res_frac) for x in res_original])

    img_low = img.resize((low_res), resample=Image.BILINEAR)
    src_final = os.path.join(src_dest, os.path.splitext(os.path.basename(src_img))[0] + f'_res_{res_frac}.tif')
    
    if do_save:
        img_low.save(src_final)
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
    file_src = os.path.join(dest, f"tile_{E}-{N}_{year}_{suffixe}.tif")
    with open(file_src, 'wb') as handler:
        handler.write(img_data)
    return file_src

    
# def get_tiles_from_canton(cantons, tiles_locs, choice):
#     cantons_dict = {}
#     print("Number of tiles in Switzerland: ", len(EN))
#     estimated_time = round(len(EN)/20/60, 2)
#     estimated_size = round(len(EN) * 0.054, 2)

# if __name__ == "__main__":
#     print(datetime.date.today().year)