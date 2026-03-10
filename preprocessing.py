import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import tifffile as tiff
import rasterio
import cv2
from tqdm import tqdm
import random


from omegaconf import OmegaConf
import warnings

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(42)

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def resize_to(img, to, is_mask=False):
    """
    Resize an image or mask to a square target size.
    Parameters: 
        img (np.ndarray) - input image or mask array; 
        to (int) - target output size (to x to); 
        is_mask (bool) - whether the input is a mask (nearest interpolation).
    Returns: 
        np.ndarray - resized image or mask.
    """

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    interp = cv2.INTER_NEAREST

    if img.ndim == 3:  # (H, W, C)
        H, W, c = img.shape
        out = np.zeros((to, to, c), dtype=img.dtype)
        for i in range(c):
            out[:,:,i] = cv2.resize(img[:,:,i].reshape((H,W)), (to, to), interpolation=interp)
        return out

    else:  # mask (H, W)
        return cv2.resize(img, (to, to), interpolation=interp)


def center_crop(img, crop_size):
    """
    Extract a centered square crop from an image or mask.
    Parameters: 
        img (np.ndarray) - input image or mask array; 
        crop_size (int) - size of the square crop.
    Returns: 
        np.ndarray - cropped image or mask.
    """
    if img.ndim == 3:
        H, W, _ = img.shape
    else:
        H, W = img.shape

    cy, cx = H // 2, W // 2
    half = crop_size // 2

    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half

    if img.ndim == 3:
        return img[y1:y2, x1:x2, :]
    else:
        return img[y1:y2, x1:x2]
    

def label_to_mask(label):
    """
    Convert a multi-channel label image into a binary mask.
    Parameters: 
        label (np.ndarray) - input label image with values in [0-255].
    Returns: 
        np.ndarray - binary mask with values {0,1}.
    """
    label = label[...,0]
    label[label != 0] = 1
    return np.astype(label, np.uint8)


def extract_random_sample(img_arr, mask_arr, shift_based_on_size, tile_size):
    """
    Extract a randomly shifted square sample from an image and its corresponding mask.
    Parameters: 
        img_arr (np.ndarray) - source image array; 
        mask_arr (np.ndarray) - corresponding label array; 
        shift_based_on_size (int) - maximum random shift allowed; 
        tile_size (int) - size of the extracted sample.
    Returns: 
        tuple[np.ndarray, np.ndarray] - sampled image tile and its corresponding mask tile.
    """

    H, W = img_arr.shape[0], img_arr.shape[1]
    starting_point_x = (W - tile_size) // 2
    starting_point_y = (H - tile_size) // 2
    minx = max(-shift_based_on_size, -starting_point_x)
    maxx = min(shift_based_on_size, W - starting_point_x - tile_size)
    miny = max(-shift_based_on_size, -starting_point_y)
    maxy = min(shift_based_on_size, H - starting_point_y - tile_size)
    shift_x = random.randint(minx, maxx)
    shift_y = random.randint(miny, maxy)
        
    x0 = starting_point_x - shift_x
    y0 = starting_point_y - shift_y

    tile = img_arr[y0:y0+tile_size, x0:x0+tile_size, :]
    tile_mask = mask_arr[y0:y0+tile_size, x0:x0+tile_size, :]
    return tile, tile_mask


def preprocessing(args):
    """
    Generate training datasets for the segmentation and/or fusion models from tiled images and masks.
    Parameters: 
        args (OmegaConf) - configuration containing preprocessing, dataset, and sampling parameters.
    Returns: 
        None - writes processed datasets and statistics to disk.
    """

    # General args
    DATASET_TYPE = args.preprocessing.dataset_type
    TILES_LOC = args.preprocessing.tiles_location
    TILES_IMG_SRC = os.path.join(TILES_LOC, 'images')
    TILES_MASKS_SRC = os.path.join(TILES_LOC, 'masks')
    RESULTS_SRC = args.preprocessing.results_location
    SUFFIXE = args.preprocessing.suffixe

    # Segmenter args
    SAMPLE_SIZE_SEGMENT = args.preprocessing.segmenter.sample_size
    BASE_SQUARES = args.preprocessing.segmenter.base_squares
    SCALES = args.preprocessing.segmenter.scales
    CROP_SIZE = int(SAMPLE_SIZE_SEGMENT / min(SCALES))
    NUM_SAMPLES_PER_TILE = args.preprocessing.segmenter.num_samples_per_tile

    # Fusion args
    SAMPLE_SIZE_FUSION = args.preprocessing.fusion.sample_size
    CENTRAL_SQUARE = args.preprocessing.fusion.central_square


    assert os.path.exists(TILES_IMG_SRC)
    assert os.path.exists(TILES_MASKS_SRC)
    assert os.listdir(TILES_IMG_SRC) == os.listdir(TILES_MASKS_SRC)
    assert DATASET_TYPE in ['segmenter', 'fusion', 'both']

    
    os.makedirs(RESULTS_SRC, exist_ok=True)

    list_tiles_img = [x for x in os.listdir(TILES_IMG_SRC) if os.path.splitext(x)[1].lower() in ['.png', '.tif', '.tiff']]


    # --------------------------------------
    # --- CREATION OF FUSION DATASET -------

    if DATASET_TYPE in ['fusion', 'both']:
        print("PRODUCING DATASET FOR FUSION:")
        dataset_base_src = os.path.join(RESULTS_SRC, f"dataset_fusion_{SUFFIXE}")
        src_new_img_src = os.path.join(dataset_base_src, 'images')
        src_new_mask_src = os.path.join(dataset_base_src, 'masks')
        src_new_label_src = os.path.join(dataset_base_src, 'labels')
        os.makedirs(dataset_base_src, exist_ok=True)
        os.makedirs(src_new_img_src, exist_ok=True)
        os.makedirs(src_new_mask_src, exist_ok=True)
        os.makedirs(src_new_label_src, exist_ok=True)
        
        for _, tile_name in tqdm(enumerate(list_tiles_img), total=len(list_tiles_img)):
            src_img = os.path.join(TILES_IMG_SRC, tile_name)
            src_mask = os.path.join(TILES_MASKS_SRC, tile_name)
            src_new_img = os.path.join(src_new_img_src, tile_name)
            src_new_mask = os.path.join(src_new_mask_src, tile_name)
            src_new_label = os.path.join(src_new_label_src, tile_name)

            H, W = rasterio.open(src_img).shape

            shift_x = random.randint(-CENTRAL_SQUARE, CENTRAL_SQUARE) // 2
            shift_y = random.randint(-CENTRAL_SQUARE, CENTRAL_SQUARE) // 2

            starting_point_x = (W - SAMPLE_SIZE_FUSION) // 2
            starting_point_y = (H - SAMPLE_SIZE_FUSION) // 2
            x0 = starting_point_x - shift_x
            y0 = starting_point_y - shift_y

            img = np.moveaxis(rasterio.open(src_img).read(), 0, 2)
            label = np.moveaxis(rasterio.open(src_mask).read(), 0, 2)
            img = img[x0:x0+SAMPLE_SIZE_FUSION, y0:y0+SAMPLE_SIZE_FUSION, :]
            label = label[x0:x0+SAMPLE_SIZE_FUSION, y0:y0+SAMPLE_SIZE_FUSION, :]
            mask = label_to_mask(label)

            tiff.imwrite(src_new_img, img, compression="zstd", compressionargs={"level": 9})
            tiff.imwrite(src_new_mask, mask, compression="zstd", compressionargs={"level": 9})
            tiff.imwrite(src_new_label, label, compression="zstd", compressionargs={"level": 9})


    # --------------------------------------
    # --- CREATION OF SEGMENTER DATASET ----

    if DATASET_TYPE in ['segmenter', 'both']:
        print("PRODUCING DATASET FOR SEGMENTATION:")
        dataset_base_src = os.path.join(RESULTS_SRC, f"dataset_segmenter_{SUFFIXE}")
        os.makedirs(dataset_base_src, exist_ok=True)
        os.makedirs(os.path.join(dataset_base_src, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_base_src, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(dataset_base_src, 'labels'), exist_ok=True)
        dataset_base_src = os.path.join(RESULTS_SRC, f"dataset_segmenter_multi_bases_{SUFFIXE}")
        os.makedirs(dataset_base_src, exist_ok=True)

        count_empties = np.zeros((len(BASE_SQUARES), len(SCALES), 2))
        for id_base, base_size in enumerate(BASE_SQUARES):
            print(f"Base size = {base_size}")
            src_results = os.path.join(dataset_base_src, f"base_size_{base_size}")
            os.makedirs(src_results, exist_ok=True)
            lst_src_scales = [os.path.join(src_results, f'res_{s}') for s in SCALES]
            for scale_dir in lst_src_scales:
                os.makedirs(scale_dir, exist_ok=True)
                os.makedirs(os.path.join(scale_dir, 'images'), exist_ok=True)
                os.makedirs(os.path.join(scale_dir, 'masks'), exist_ok=True)
                os.makedirs(os.path.join(scale_dir, 'labels'), exist_ok=True)

            for _, img_name in tqdm(enumerate(list_tiles_img), total=len(list_tiles_img)):
                src_img = os.path.join(TILES_IMG_SRC, img_name)
                src_mask = os.path.join(TILES_MASKS_SRC, img_name)
                img_arr = np.moveaxis(rasterio.open(src_img).read(), 0, 2)
                mask_arr = np.moveaxis(rasterio.open(src_mask).read(), 0, 2)

                A_img = img_arr.shape[0] * img_arr.shape[1]
                A_samp = (base_size)**2
                num_max_samples = int(round(A_img/A_samp/2, 0))

                for i in range(min(NUM_SAMPLES_PER_TILE, num_max_samples)):
                    samp_img_arr, samp_label_arr = extract_random_sample(img_arr, mask_arr, base_size, CROP_SIZE)
                    for id_scale, scale in enumerate(SCALES):
                        src_img_out = os.path.join(lst_src_scales[id_scale], 'images', os.path.splitext(img_name)[0] + f"_scale_{scale}_{i}.tif")
                        src_label_out = os.path.join(lst_src_scales[id_scale], 'labels', os.path.splitext(img_name)[0] + f"_scale_{scale}_{i}.tif")
                        src_mask_out = os.path.join(lst_src_scales[id_scale], 'masks', os.path.splitext(img_name)[0] + f"_scale_{scale}_{i}.tif")
                        
                        cropped_img_arr = center_crop(samp_img_arr, int(CROP_SIZE * scale))
                        resized_img_arr = resize_to(cropped_img_arr, SAMPLE_SIZE_SEGMENT)

                        cropped_label_arr = center_crop(samp_label_arr, int(CROP_SIZE * scale))
                        resized_label_arr = resize_to(cropped_label_arr, SAMPLE_SIZE_SEGMENT)
                        resized_mask_arr = label_to_mask(resized_label_arr)

                        tiff.imwrite(src_img_out, resized_img_arr, compression="zstd", compressionargs={"level": 9})
                        tiff.imwrite(src_label_out, resized_label_arr, compression="zstd", compressionargs={"level": 9})
                        tiff.imwrite(src_mask_out, resized_mask_arr, compression="zstd", compressionargs={"level": 9})

                        if np.sum(resized_mask_arr[:-1,...] > 0) == 0:
                            count_empties[id_base, id_scale, 0] += 1
                        else:
                            count_empties[id_base, id_scale, 1] += 1

        # Plot occupied
        results_frac_arr = count_empties / len(list_tiles_img) * 100
        fig, ax = plt.subplots(figsize=(10,8))
        ax.plot(results_frac_arr[:,:,1], label=SCALES, linewidth=2)
        ax.set_ylabel('Pourcentage [%]')
        ax.legend()
        ax.set_xticks(range(len(BASE_SQUARES)))
        ax.set_xticklabels(BASE_SQUARES)
        ax.set_xlabel('base squares [px]')
        ax.set_ylabel('fraction [%]')
        ax.set_ylim([0,105])
        ax.grid()
        
        plt.title("Fraction of occupied samples")
        saving_loc = os.path.join(dataset_base_src, 'fraction_of_occupied.png')
        plt.savefig(saving_loc)
        plt.savefig(saving_loc.split('.')[0] + '.eps', format='eps')
        plt.clf()
                

if __name__ == "__main__":
    args = OmegaConf.load("./config/preprocessing.yaml")

    preprocessing(args)
