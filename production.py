import os
import shutil
import json
import argparse
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import DBSCAN
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from omegaconf import OmegaConf
from time import time
import torch
import pickle
import tifffile as tiff

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils.production_utils import download_tile, produce_with_lower_res, predict, predict_with_batch, geo_transfert, prob_to_rgb, load_latest_checkpoint

from transformers import SegformerForSemanticSegmentation

# Clearing warnings
Image.MAX_IMAGE_PIXELS = None
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning
)


def tiles_downloading(
        dest_tiles, 
        downloading_mode, 
        canton=None,
        area=None,
        year=None,
        dest_not_empty='add'
        ):
    tiles_to_download = []
    lst_tiles_src = []
    os.makedirs(dest_tiles, exist_ok=True)
    if len(os.listdir(dest_tiles)) > 0:
        if dest_not_empty == 'replace':
            shutil.rmtree(dest_tiles)
            os.makedirs(dest_tiles, exist_ok=True)
        elif dest_not_empty == 'stop':
            raise PermissionError('The destination already contains files. Empty it or change parameter "dest_not_empty"')
        
    # find tiles to download
    tiles_locs = gpd.read_file("utils/resources/tiles_locs/ch.swisstopo.images-swissimage-dop10.metadata.shp")
    if downloading_mode == 'year':
        tiles_locs = tiles_locs.loc[tiles_locs.datenstand == str(year)]
    ids = tiles_locs.id.values
    E = [x.split('_')[0] for x in ids]
    N = [x.split('_')[1] for x in ids]
    EN = [[int(x), int(y)] for x,y in zip(E,N)]

    if downloading_mode in ['canton', 'area']:

        # find area of interest
        (Emin, Emax, Nmin, Nmax) = (0,0,0,0)
        if downloading_mode == 'canton':
            cantons = gpd.read_file('utils/resources/swissboundaries/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp')
            if canton not in cantons.NAME.values:
                raise AttributeError(f"The given canton's name is not correct. Please choos between the following: \n {cantons.NAME.values}")
            
            canton_polygons = cantons[cantons.NAME == canton]
            Emin = int(canton_polygons.bounds.minx.values[0] // 1000)
            Emax = int((canton_polygons.bounds.maxx.values[0] + 1) // 1000)
            Nmin = int(canton_polygons.bounds.miny.values[0] // 1000)
            Nmax = int((canton_polygons.bounds.maxy.values[0] + 1) // 1000)
        elif downloading_mode == 'area':
            Emin = int(area.Emin)
            Emax = int(area.Emax)
            Nmin = int(area.Nmin)
            Nmax = int(area.Nmax)

        tiles_to_download = [x for x in EN if Emin <= x[0] <= Emax and Nmin <= x[1] <= Nmax]

        # Gives information about tiles to be downloaded
        text = f"""
    ({Emin},{Nmax+1}) --- ({Emax+1},{Nmax+1})
        |               |
        |               |
        |               |
    ({Emin},{Nmin}) --- ({Emax+1},{Nmin})
    """
        if downloading_mode == 'canton':
            print(f"Processing canton {canton} with following area ({len(tiles_to_download)} tiles):")
        else:
            print(f"Processing following area ({len(tiles_to_download)} tiles):")
        print(text)
    elif downloading_mode in ['year', 'full']:
        
        tiles_to_download = EN

        # Gives information about tiles to be downloaded
        if downloading_mode == 'year':
            print(f"Processing data of the year {year} ({len(tiles_to_download)} tiles):")
        else:
            print(f"Processing all of Switzerland ({len(tiles_to_download)})")
    else:
        raise AttributeError("downloader.mode is not a valid value!")
        
    # download tiles
    for _, tile in tqdm(enumerate(tiles_to_download), total=len(tiles_to_download), desc="Downloading"):
        tile_src = download_tile(tile[0], tile[1], dest_tiles)
        if tile_src != None:
            lst_tiles_src.append(tile_src)

    return lst_tiles_src


def prediction(
        src_img, 
        src_inter, 
        src_dest_preds, 
        src_dest_probas, 
        resolutions, 
        model_dir, 
        batch_size=8,
        tile_size=512, 
        stride=256, 
        threshold_proba= 0.5, 
        threshold_grouping=0.5,
        output_format='tif',):
    
    # predict at each resolution
    images = []
    preds = []
    masks = []
    probas = []

    # load model
    ckpt_path = load_latest_checkpoint(model_dir)
    model = SegformerForSemanticSegmentation.from_pretrained(ckpt_path)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(DEVICE)
    model.eval()

    # time0 = time()
    for res in resolutions:
        res_img, src_res_img = produce_with_lower_res(src_img, src_inter, res, do_show=False)
        pred_mask, preds_img, proba_img = predict_with_batch(
            image=res_img, 
            model=model, 
            img_path=src_res_img,
            batch_size=batch_size,
            tile_size=tile_size,
            stride=stride,
            th=threshold_proba, 
            do_show=False,
            )
        images.append(res_img)
        preds.append(preds_img)
        masks.append(pred_mask)
        probas.append(proba_img)
    # print("TIME FOR THE PREDICTION LOOP: ", time() - time0)
    # time0 = time()
    # merge different resolutions into one final
    original_img = Image.open(src_img)

    W, H = original_img.size

    #   _creation of final preds
    final_product = np.zeros((H, W), dtype=np.float32)

    for mask in masks:
        rescaled_mask = Image.fromarray(mask).resize((W, H), Image.NEAREST)
        final_product += rescaled_mask

    final_product /= len(masks)
    final_product[final_product >= threshold_grouping] = 1
    final_product[final_product < threshold_grouping] = 0
    final_product = final_product.astype(np.uint8)

    final_product_rgb = np.zeros((W, H, 3))
    final_product_rgb[final_product == 1] = 255

    src_final_preds_mask = os.path.join(src_dest_preds, os.path.splitext(os.path.basename(src_img))[0] + f'_mask.tif')
    src_final_preds_img = os.path.join(src_dest_preds, os.path.splitext(os.path.basename(src_img))[0] + f'_img.tif')

    # Image.fromarray(final_product.astype(np.uint8), mode='L').save(src_final_preds_mask, compression='zstd')
    # Image.fromarray(final_product_rgb.astype(np.uint8), mode='RGB').save(src_final_preds_img, compression='zstd')
    tiff.imwrite(src_final_preds_mask, final_product.astype(np.uint8), compression="zstd", compressionargs={"level": 7})
    tiff.imwrite(src_final_preds_img, final_product_rgb.astype(np.uint8), compression="zstd", compressionargs={"level": 7})

    del final_product_rgb
    del final_product

    #   _creation of final probas
    final_probas = np.zeros((W,H), dtype=np.float32)
    # final_probas_rgb = np.zeros((W,H,4))

    for proba in probas:
        rescaled_proba = Image.fromarray(proba).resize((W, H), Image.NEAREST)
        final_probas += rescaled_proba

    final_probas /= len(probas)
    # final_probas_rgb = prob_to_rgb(final_probas)

    final_probas = np.clip(final_probas, 0, 1)
    final_probas = (final_probas * 255).astype(np.uint8)

    src_final_probas_mask = os.path.join(src_dest_probas, os.path.splitext(os.path.basename(src_img))[0] + f'_probas.tif')
    # src_final_probas_mask_pickle = os.path.join(src_dest_probas, os.path.splitext(os.path.basename(src_img))[0] + f'_probas.pickle')
    # src_final_probas_img = os.path.join(src_dest_probas, os.path.splitext(os.path.basename(src_img))[0] + f'_img.tif')

    # Image.fromarray(final_probas).save(src_final_probas_mask)
    # Image.fromarray(probas_uint16, mode='I;16').save(os.path.splitext(src_final_probas_mask)[0] + '.png')
    tiff.imwrite(src_final_probas_mask, final_probas, compression="zstd", compressionargs={"level": 7})
    # with open(src_final_probas_mask_pickle, 'wb') as f:
    #     pickle.dump(final_probas, f)

    # Image.fromarray(final_probas_rgb.astype(np.uint8), mode='RGB').save(src_final_probas_img)
    # print("TIME FOR THE REST: ", time() - time0)
    return src_final_preds_mask, src_final_preds_img, src_final_probas_mask#, src_final_probas_img


def clustering(src_img, src_dest, eps, min_samples, min_cluster_size,  color_palette, output_format='png'):
    # extract coordinates of landslides
    img_arr = np.array(Image.open(src_img))
    pos_ls = np.argwhere(img_arr)

    mask_clusters = np.zeros(img_arr.shape)
    rgb_clusters = np.zeros((mask_clusters.shape[0], mask_clusters.shape[1], 4))

    if len(pos_ls) > 0:
        # create cluster map
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pos_ls)
        cluster_labels = clustering.labels_
        lst_clusters = set(cluster_labels)


        # unpack coordinates
        rows = pos_ls[:, 0]
        cols = pos_ls[:, 1]

        mask_clusters[rows, cols] = cluster_labels
        mask_clusters[mask_clusters == -1] = 0

        # saving clusters
        distinct_colors_rgb8 = [(x, y, z, 255) for [x,y,z] in color_palette]

        # for _, cluster in tqdm(enumerate(lst_clusters), total=len(lst_clusters)):
        for cluster in lst_clusters:
            if np.sum(mask_clusters == cluster) < min_cluster_size:
                rgb_clusters[mask_clusters == cluster] = [255, 255, 255, 0]
                mask_clusters[mask_clusters == cluster] = 0
            else:
                id_color = cluster % len(distinct_colors_rgb8)
                rgb_clusters[mask_clusters == cluster] = distinct_colors_rgb8[id_color]

    # if np.sum(mask_clusters) == 0:
    #     return
    
    # Background in white
    rgb_clusters[mask_clusters == 0] = (255,255,255, 0)

    # save results
    src_mask = os.path.join(src_dest, os.path.splitext(os.path.basename(src_img))[0] + f'clusters_eps_{eps}_min_samp_{min_samples}_mask.{output_format}')
    src_img = os.path.join(src_dest, os.path.splitext(os.path.basename(src_img))[0] + f'clusters_eps_{eps}_min_samp_{min_samples}_img.{output_format}')
    Image.fromarray(mask_clusters.astype(np.uint16), mode='I;16').save(src_mask)
    Image.fromarray(rgb_clusters.astype(np.uint8), mode='RGB').save(src_img)
    tiff.imwrite(src_mask, mask_clusters.astype(np.uint16), compression="zstd", compressionargs={"level": 7})
    tiff.imwrite(src_img, rgb_clusters.astype(np.uint8), compression="zstd", compressionargs={"level": 7})

    return src_mask, src_img


def vectorize(src_target, src_dest):
    with rasterio.open(src_target) as src:
        mask = src.read(1)
        if np.sum(mask) == 0:
            return
        transform = src.transform
        crs = src.crs

        # Extract polygons AND their raster values
        records = [
            {"geometry": shape(geom), "raster_val": value}
            for geom, value in shapes(mask, transform=transform)
            if value != 0 # optional: ignore background
        ]

    # Build georeferenced GeoDataFrame
    gdf = gpd.GeoDataFrame(records, crs=crs)

    # Save to GeoPackage
    src_polygons = os.path.join(src_dest, os.path.splitext(os.path.basename(src_target))[0] + '_landslides.gpkg')
    gdf.to_file(src_polygons, driver="GPKG")
    
    return src_polygons


def production(args):
    # return
    # raise ValueError("Test")
    start_time = time()

    # Load parameters
    DEST_ORIGINAL_TILES = args.downloader.destination
    DEST_NOT_EMPTY = args.downloader.dest_not_empty
    SKIP_AUTO_DOWNLOADING = args.downloader.skip_auto_downloading
    DOWNLOADING_MODE = args.downloader.mode
    CANTON = args.downloader.canton
    AREA = args.downloader.area
    YEAR = args.downloader.year
    DEST_PREDS = args.predictions.destination
    MODEL_DIR = args.predictions.model_dir
    BATCH_SIZE = args.predictions.batch_size
    OUTPUT_FORMAT = args.predictions.output_format
    THRESHOLD_PREDS = args.predictions.threshold_preds
    THRESHOLD_GROUPING = args.predictions.threshold_grouping
    TILE_SIZE = args.predictions.tile_size
    STRIDE = args.predictions.stride
    RESOLUTIONS = args.predictions.resolutions
    DO_RM_INTERMED_FILES = args.predictions.do_rm_intermed_files

    DEST_PREDS = DEST_ORIGINAL_TILES if DEST_PREDS.lower() == 'default' else DEST_PREDS
    dest_preds_dir = os.path.join(DEST_PREDS, 'predictions')
    dest_probas_dir = os.path.join(DEST_PREDS, 'probas')
    dest_clusters_dir = os.path.join(DEST_PREDS, 'clusters')
    dest_vectors_dir = os.path.join(DEST_PREDS, 'vectors')
    dest_originals_dir = os.path.join(DEST_PREDS, 'originals')
    dest_inter_dir = os.path.join(DEST_PREDS, 'inter')

    # === TILES DOWNLOADING ===
    # =========================
    os.makedirs(DEST_ORIGINAL_TILES, exist_ok=True)
    if not SKIP_AUTO_DOWNLOADING:
        lst_tiles_src = tiles_downloading(
            dest_tiles=dest_originals_dir,
            downloading_mode=DOWNLOADING_MODE,
            canton=CANTON,
            area=AREA,
            year=YEAR,
            dest_not_empty=DEST_NOT_EMPTY
        )
    else:
        img_exts = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        lst_tiles_src = [os.path.join(DEST_ORIGINAL_TILES, x) for x in os.listdir(DEST_ORIGINAL_TILES) if os.path.splitext(x)[1].lower() in img_exts]

    if len(lst_tiles_src) == 0:
        print("NO TILE TO PROCESS!")
        return

    os.makedirs(dest_preds_dir, exist_ok=True)
    os.makedirs(dest_probas_dir, exist_ok=True)
    os.makedirs(dest_clusters_dir, exist_ok=True)
    os.makedirs(dest_vectors_dir, exist_ok=True)
    os.makedirs(dest_inter_dir, exist_ok=True)

    for _, src_img in tqdm(enumerate(lst_tiles_src), total=len(lst_tiles_src), desc="Processing tiles"):
        
        # === PREDICTIONS =====
        # =====================
        # time_start = time()
        src_pred_mask, src_pred_img, src_proba_mask = prediction(
            src_img=src_img,
            src_inter=dest_inter_dir,
            src_dest_preds=dest_preds_dir, 
            src_dest_probas=dest_probas_dir,
            resolutions=RESOLUTIONS, 
            model_dir=MODEL_DIR, 
            batch_size=BATCH_SIZE,
            tile_size=TILE_SIZE,
            stride=STRIDE,
            threshold_proba=THRESHOLD_PREDS, 
            threshold_grouping=THRESHOLD_GROUPING,
            output_format=OUTPUT_FORMAT,
            )
        # print("TIME TO PREDICT: ", time() - time_start)
        # time_start = time()

        # === VECTORIZATION ===
        # =====================
        EPS = args.vectorization.dbscan_eps
        MIN_SAMPLES = args.vectorization.dbscan_min_samples
        MIN_CLUSTER_SIZE = args.vectorization.min_cluster_size
        SRC_COLOR_PALETTE = args.vectorization.src_color_palette
        with open(SRC_COLOR_PALETTE, 'r') as f:
            color_palette = json.load(f)

        res_clustering = clustering(
            src_img=src_pred_mask,
            src_dest=dest_clusters_dir,
            eps= EPS, 
            min_samples=MIN_SAMPLES, 
            min_cluster_size=MIN_CLUSTER_SIZE,
            color_palette=color_palette,
            output_format=OUTPUT_FORMAT,
            )
        
        # test if no cluster found
        if res_clustering == None or OUTPUT_FORMAT.lower() not in ['tif', 'tiff']:
            continue
        else:
            src_clusters_mask, src_clusters_img = res_clustering
            geo_transfert(src_img, src_pred_mask, True)
            geo_transfert(src_img, src_pred_img, True)
            geo_transfert(src_img, src_proba_mask, True)
            # geo_transfert(src_img, src_proba_img, True)
            geo_transfert(src_img, src_clusters_mask, True)
            geo_transfert(src_img, src_clusters_img, True)

            vectorize(src_clusters_mask, dest_vectors_dir)

        # print("TIME TO VECTORIZE: ", time() - time_start)
        # time_start = time()
    if DO_RM_INTERMED_FILES:
        shutil.rmtree(dest_inter_dir)

    # Show duration of process
    delta_time_loop = time() - start_time
    hours = int(delta_time_loop // 3600)
    min = int((delta_time_loop - 3600 * hours) // 60)
    sec = int(delta_time_loop - 3600 * hours - 60 * min)
    print(f"\n==== FINISH! {len(lst_tiles_src)} tiles processed in {hours}:{min}:{sec} ====\n")


if __name__ == "__main__":
    # conf = OmegaConf.load('config/production.yaml')
    # production(conf)

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    args = parser.parse_args()
    cfg_path = args.config

    if cfg_path != "":
        print("- Producing from argument - ")
        args = OmegaConf.load(cfg_path)
    else:
        print("- Producing from yaml file - ")
        args = OmegaConf.load('config/production.yaml')

    # print(args)
    production(args)

