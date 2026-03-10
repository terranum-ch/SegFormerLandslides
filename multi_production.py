import os
import shutil
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from copy import deepcopy
import subprocess
import tempfile
from itertools import product
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
# import matplotlib
import matplotlib.cm as cm
import pickle
import tifffile as tiff
import rasterio
from utils.production_utils import geo_transfert
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
Image.MAX_IMAGE_PIXELS = None


# def save_image(arr, dest):
#     tiff.imwrite(dest, arr, compression="zstd", compressionargs={"level": 7})


def prob_to_rgb(prob_map, cmap_name="RdYlBu", range=[0,2**16-1]):
    """
    Convert a probability map to an RGB image using a colormap.
    Parameters: 
        prob_map (np.ndarray) - 2D probability map; 
        cmap_name (str) - name of the matplotlib colormap used for color mapping; 
        range (list[int]) - value range used for normalization.
    Returns: 
        np.ndarray - RGB image array of shape (H, W, 3) with dtype uint8.
    """

    cmapRB = cm.get_cmap(cmap_name)
    prob_map = prob_map.astype(np.float32) / 255

    colors_RB = [cmapRB(i) for i in np.linspace(0,1, 100)]
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors_RB, N=10).reversed()
    # Apply colormap → RGBA in [0,1]
    rgba = cmap(prob_map)

    # Drop alpha channel and convert to uint8
    rgb = (rgba[..., :3] * 255).astype(np.uint8)

    return rgb


def postprocess(src_pred):
    """
    Postprocess prediction outputs by generating transparent masks, colorized probability maps, and transferring georeferencing.
    Parameters: 
        src_pred (str) - path to the directory containing prediction outputs.
    Returns: 
        None.
    """

    src_masks = os.path.join(src_pred, 'masks')
    src_probas = os.path.join(src_pred, 'probas')
    os.makedirs(src_masks, exist_ok=True)

    # process masks
    for img in [x for x in os.listdir(os.path.join(src_pred, 'predictions')) if 'img' in x]:
        src_mask = os.path.join(src_masks, img)
        shutil.copyfile(
            os.path.join(src_pred, 'predictions', img),
            src_mask,
        )
        with rasterio.open(src_mask) as src:
            img_arr = np.transpose(np.astype(src.read(),np.uint8), (1,2,0))

        img_arr_transparent = np.zeros((img_arr.shape[0], img_arr.shape[1], 4), dtype=np.uint8)
        img_arr_transparent[...,:3] = img_arr
        mask = img_arr[:,:,0] == 255
        img_arr_transparent[:,:,3][mask] = 255
        
        tiff.imwrite(src_mask.replace('img', 'transparent'), img_arr_transparent, compression="zstd", compressionargs={"level": 9})
        geo_transfert(src_mask, src_mask.replace('img', 'transparent'), True)

    # process probas
    for img in [x for x in os.listdir(src_probas) if 'transparent' not in x]:
        src_proba = os.path.join(src_probas, img)
        # img_arr = np.array(Image.open(src_proba))
        
        with rasterio.open(src_proba) as src:
            img_arr = np.astype(src.read(),np.uint8)

        img_arr = img_arr.reshape(img_arr.shape[1::])

        img_rgb = prob_to_rgb(img_arr)
        img_rgb_transparent = np.zeros((img_arr.shape[0], img_arr.shape[1], 4), dtype=np.uint8)
        img_rgb_transparent[..., :3] = img_rgb
        img_rgb_transparent[..., 3][img_arr >= 0.05 * 255] = 255
        
        src_img_transparent = os.path.splitext(src_proba)[0] + f"_transparent{os.path.splitext(src_proba)[1]}"
        tiff.imwrite(src_img_transparent, img_rgb_transparent, compression="zstd", compressionargs={"level": 9})
        geo_transfert(src_proba, src_img_transparent, True)

    # remove temporary files
    for el in os.listdir(src_pred):
        if el not in ['masks','probas', 'vectors']:
            shutil.rmtree(os.path.join(src_pred, el))


def multi_production(configs, args, verbose=False):
    """
    Run production for multiple configurations by generating temporary config files, launching prediction subprocesses, and postprocessing results.
    Parameters: 
        configs (list[dict]) - list of configuration overrides for each production run; 
        args (OmegaConf) - base configuration used as template; 
        verbose (bool) - whether to print configuration details for each run.
    Returns: 
        None.
    """
    
    src_res_prod = "./results/production"
    if os.path.exists(os.path.join(src_res_prod, 'problematic_confs.pickle')):
        os.remove(os.path.join(src_res_prod, 'problematic_confs.pickle'))
    lst_conf_problematic = []
    for id_conf, conf in tqdm(enumerate(configs), total=len(configs)):
        args_temp = deepcopy(args)
        for key, val in conf.items():
            OmegaConf.update(args_temp, key, val)

        if verbose:
            print("PRODUCING WITH FOLLOWING SET OF PARAMETERS:")
            for key, val in conf.items():
                print("\t", key, ": ", OmegaConf.select(args_temp, key))
            print('---\n')

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode='w', encoding='utf-8') as f:
            OmegaConf.save(args_temp, f)
            cfg_path = f.name

        # res = subprocess.call(
        #     [".venv/Scripts/python.exe", "production.py", f"--config={cfg_path}"], 
        #     stdout=subprocess.DEVNULL,
        #     stderr=None,
        # )
        res = subprocess.run(
            [".venv/Scripts/python.exe", "production_fusion.py", f"--config={cfg_path}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )

        if res.returncode == 0:    # no error during production
            args = OmegaConf.load(cfg_path)
            postprocess(args.predictions.destination)
        else:
            print(f"Error with {id_conf}th conf!")
            lst_conf_problematic.append(args)
        
        if len(lst_conf_problematic) > 0:
            os.makedirs(src_res_prod, exist_ok=True)
            with open(os.path.join(src_res_prod, 'problematic_confs.pickle'), 'wb') as f:
                pickle.dump(lst_conf_problematic, f)
        # break
        
        # print("\n----------------------------------------\n")


if __name__ == "__main__":
    postprocess(r"D:\GitHubProjects\Terranum_repo\LandSlides\segformerlandslides\data\test_finetuning\with_multiple_versions\finetuning\test_6")
    quit()
    src_dest = "data/test_finetuning/with_multiple_versions/finetuning/preds"
    os.makedirs(src_dest, exist_ok=True)
    list_th_preds = [0.05, 0.1, 0.3, 0.5, 0.7]
    # list_th_group = [0.05, 0.1, 0.3, 0.5, 0.7]
    list_stride = [128, 256, 512]
    list_min_cluster_size = [100, 1000, 5000, 10000]


    configs = []

    for config in product(list_th_preds, list_stride, list_min_cluster_size):
        th_pred, stride, min_cluster_size = config

        suffixe = f"stride={stride}_minsize={min_cluster_size}_thpred={th_pred}"

        os.makedirs(os.path.join(src_dest, suffixe), exist_ok=True)

        config = {
            "predictions.destination": os.path.join(src_dest, suffixe),
            "predictions.threshold_preds": th_pred,
            # "predictions.threshold_grouping": th_group,
            "predictions.stride": stride,
            "vectorization.min_cluster_size": min_cluster_size,
        }
        configs.append(config)


    args = OmegaConf.load('./config/production.yaml')

    multi_production(configs, args)