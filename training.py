import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import numpy as np
import pandas as pd
import gc
import json
import argparse
import torch
from torch.utils.data import Subset, random_split
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    SegformerConfig,
)
import albumentations as A

import rasterio
import shutil
import tifffile as tiff
from PIL import Image
from tqdm import tqdm
from time import time
from datetime import datetime
from omegaconf import OmegaConf

# from utils.dataset import SegmentationDataset
from utils.dataset_fusion import SegmentationDataset#, SegFusionDataset, DatasetProxy
from utils.trainer import TrainValMetricsTrainer, MultiScaleFusionModel, collate_with_filename, logits_to_preds
from utils.metrics import compute_metrics
from utils.callbacks import MetricsCallback, SavesCurrentStateCallback
from utils.visualization import show_iou_per_class, show_loss_pa, show_mean_iou_dice, show_confusion_matrix

import logging
from contextlib import contextmanager
import warnings
from rasterio.errors import NotGeoreferencedWarning

# If batch size too big, fails instead of slowing down
torch.cuda.set_per_process_memory_fraction(0.95)
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

@contextmanager
def mute_logging(level=logging.CRITICAL):
    previous_level = logging.root.manager.disable
    logging.disable(level)
    try:
        yield
    finally:
        logging.disable(previous_level)


def get_best_checkpoint(training_dir):
    lst_checkpoints = [x for x in os.listdir(training_dir) if 'checkpoint' in x and not 'last' in x and os.path.isdir(os.path.join(training_dir, x))]

    lst_steps = [int(x.split('-')[-1]) for x in lst_checkpoints]
    best_step = np.max(lst_steps)
    best_checkpoint = [x for x in lst_checkpoints if str(best_step) in x][0]

    return os.path.join(training_dir, best_checkpoint)


def training(args):
    OUTPUT_DIR = rf"{args.train.output_dir}"
    OUTPUT_SUFFIXE = args.train.output_suffixe
    VAL_SPLIT = args.train.val_split
    NUM_EPOCHS = args.train.num_epochs
    NUM_WORKERS = args.train.num_workers
    BATCH_SIZE = args.train.batch_size
    LEARNING_RATE = args.train.learning_rate
    WEIGHT_DECAY = args.train.weight_decay
    PRETRAINED_MODEL = args.train.pretrained_model
    IS_TRAINED = args.train.is_trained
    
    SCALES = args.train.scales
    FROM_PRETRAIN = args.train.from_pretrain
    PRETRAIN_DIR = args.train.pretrain_dir

    RESUME_FROM_EXISTING = args.train.resume_from_existing
    EXISTING_DIR_TO_RESUME_FROM = os.path.join(args.train.existing_dir, 'last_checkpoint') if RESUME_FROM_EXISTING else None

    DATASET_DIR_SEG = args.dataset.segmenter.dataset_dir
    TRAINING_SET_DIR = args.dataset.segmenter.trainset_dir
    VALIDATION_SET_DIR = args.dataset.segmenter.valset_dir
    DATASET_MODE = args.dataset.segmenter.mode
    DATASET_DIR_FUS = args.dataset.fusion.dataset_dir

    DATASET_DIR = DATASET_DIR_SEG if IS_TRAINED == 'segmenter' else DATASET_DIR_FUS

    try:
        assert FROM_PRETRAIN + RESUME_FROM_EXISTING < 2
    except:
        raise AttributeError("PARAMETERS 'train.from_pretrain' and 'train.resume_from_existing' can not be both set to True!!!")

    DO_DATA_AUGMENTATION = args.train.do_data_augmentation
    DO_SAVE_BEST_PREDS = args.train.do_save_best_preds

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create architecture
    RESULTS_DIR = os.path.join(OUTPUT_DIR, datetime.now().strftime(r"%Y%m%d_%H%M%S_") + f'{IS_TRAINED}_' + f"{NUM_EPOCHS}_epochs_" + OUTPUT_SUFFIXE)
    LOG_DIR = os.path.join(RESULTS_DIR, 'logs')
    CONFMAT_DIR = os.path.join(LOG_DIR, "confmats")
    BESTPREDS_DIR = os.path.join(LOG_DIR, "best_preds")
    IMG_DIR = os.path.join(RESULTS_DIR, 'images')
    LAST_CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'last_checkpoint')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(CONFMAT_DIR, exist_ok=True)

    # save config
    with open(os.path.join(RESULTS_DIR, 'config.json'), 'w') as f:
        OmegaConf.save(args, f)

    time_start = time()

    # Load model + processor
    with mute_logging():
        processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", use_fast=True) if IS_TRAINED == 'segmenter' else None
    
    if FROM_PRETRAIN or IS_TRAINED == 'fusion':
        PRETRAINED_MODEL = get_best_checkpoint(PRETRAIN_DIR)

    segformer = SegformerForSemanticSegmentation.from_pretrained(
            PRETRAINED_MODEL,
            num_labels=2,
            ignore_mismatched_sizes=True  # <- Important for custom classes
        )
    if IS_TRAINED == 'segmenter':
        model = segformer
    elif IS_TRAINED == 'fusion':
        config = SegformerConfig.from_pretrained(PRETRAINED_MODEL)
        config.scales = [float(s) for s in SCALES]
        
        model = MultiScaleFusionModel(
            segformer=segformer,
            scales=[float(s) for s in SCALES],
        )
    else:
        raise AttributeError('"is_trained" NOT CORRECT in training.yaml')

    
    # Defining a transform for data augmentation
    list_transforms = [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)]
    if args.train.do_da_scaling:
        list_transforms.append(A.Downscale((0.25, 1.0), p=1.0))
    train_transform = A.Compose(list_transforms)

    if DATASET_MODE == 'auto':
        full_dataset_train = SegmentationDataset(
            # mode=IS_TRAINED,
            data_dir= DATASET_DIR,
            # image_dir=os.path.join(TRAINING_SET_DIR, "images"),
            # mask_dir=os.path.join(TRAINING_SET_DIR, "masks"),
            processor=processor,
            transform=None,
            # scales=SCALES,
        )

        full_dataset_val = SegmentationDataset(
            # mode=IS_TRAINED,
            data_dir= DATASET_DIR,
            # image_dir=os.path.join(TRAINING_SET_DIR, "images"),
            # mask_dir=os.path.join(TRAINING_SET_DIR, "masks"),
            processor=processor,
            transform=None,
            # scales=SCALES,
        )

        split_idx = int(len(full_dataset_train) * (1 - VAL_SPLIT))

        train_indices, val_indices = random_split(
            range(len(full_dataset_train)),
            [split_idx, len(full_dataset_train) - split_idx],
            generator=torch.Generator().manual_seed(42)
        )

        train_subset = Subset(full_dataset_train, train_indices.indices)
        val_subset   = Subset(full_dataset_val, val_indices.indices)
    elif DATASET_MODE == 'split':
        train_subset = SegmentationDataset(
            # mode=IS_TRAINED,
            data_dir= TRAINING_SET_DIR,
            # image_dir=os.path.join(TRAINING_SET_DIR, "images"),
            # mask_dir=os.path.join(TRAINING_SET_DIR, "masks"),
            processor=processor,
            transform=None,
            # scales=SCALES,
        )
        val_subset = SegmentationDataset(
            # mode=IS_TRAINED,
            data_dir= VALIDATION_SET_DIR,
            # image_dir=os.path.join(TRAINING_SET_DIR, "images"),
            # mask_dir=os.path.join(TRAINING_SET_DIR, "masks"),
            processor=processor,
            transform=None,
            # scales=SCALES,
        )
    else:
        raise AttributeError('"dataset_mode" NOT CORRECT in dataset.yaml')

    if DO_DATA_AUGMENTATION:
        if isinstance(train_subset, Subset):
            train_subset.dataset.transform = train_transform
        else:
            train_subset.transform = train_transform

    # Training arguments
    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,  # Where checkpoints and logs are saved
        num_train_epochs=NUM_EPOCHS,        # Total number of epochs
        per_device_train_batch_size=BATCH_SIZE,      # Adjust according to your GPU memory
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,                  # Regularization

        # Logging
        logging_dir=LOG_DIR,                # TensorBoard logs
        logging_strategy="steps",           # Log every N steps
        # logging_steps=len(train_subset),    # Adjust for dataset size
        logging_steps=50,    # Adjust for dataset size
        log_level="info",

        # Checkpoints
        save_strategy="epoch",              # Save checkpoint at the end of each epoch
        save_total_limit=3,                 # Keep last 3 checkpoints
        save_steps=None,                    # Not used when saving by epoch

        # Evaluation
        eval_strategy="epoch",              # Evaluate at the end of each epoch
        eval_accumulation_steps=1,
        load_best_model_at_end=True,        # Load checkpoint with best metric
        metric_for_best_model="mean_dice",       # Adjust if using other metrics

        # Others
        fp16=True,                          # Mixed precision (if GPU supports)
        # prediction_loss_only=True,
        gradient_accumulation_steps=1,      # Increase effective batch size if needed
        dataloader_num_workers=NUM_WORKERS,           # Adjust according to CPU cores
        disable_tqdm=False,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = TrainValMetricsTrainer(
        confmat_dir=CONFMAT_DIR,
        model=model,
        args=training_args,
        data_collator=collate_with_filename,
        train_dataset=train_subset,
        eval_dataset= val_subset,
        processing_class=processor,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(MetricsCallback(trainer=trainer, cf_dir=CONFMAT_DIR))
    trainer.add_callback(SavesCurrentStateCallback(trainer=trainer, last_checkpoint_dir=LAST_CHECKPOINT_DIR))

    # Train
    trainer.train(resume_from_checkpoint=EXISTING_DIR_TO_RESUME_FROM)

    # Save final model
    trainer.save_model(os.path.join(RESULTS_DIR, "segformer_trained_model"))
    if processor is not None:
        processor.save_pretrained(os.path.join(RESULTS_DIR, "segformer_trained_model"))

    # Visualization
    state_file = os.path.join(LAST_CHECKPOINT_DIR, "trainer_state.json")
    
    with open(state_file, "r") as f:
        state = json.load(f)
    history = state["log_history"]

    show_loss_pa(history,os.path.join(IMG_DIR, 'loss_pa.png'), False, True)
    show_mean_iou_dice(history,os.path.join(IMG_DIR, 'mean_iou_dice.png'), False, True)
    show_iou_per_class(history,os.path.join(IMG_DIR, 'iou_per_class.png'), False, True)

    # Save best results
    best_step = trainer.state.best_global_step
    best_results = [x for x in history if x['step'] == best_step]
    assert len(best_results) == 2
    dict_best_results = best_results[0]
    for key, val in best_results[1].items():
        if key not in ["epoch", "step"]:
            dict_best_results[key] = val
    with open(os.path.join(RESULTS_DIR, 'best_results.json'), 'w') as f:
        json.dump(dict_best_results, f, indent=2)

    # Save best preds
    if DO_SAVE_BEST_PREDS:
        print("Computing best preds..")
        
        os.makedirs(BESTPREDS_DIR, exist_ok=True)
        src_best_preds_img = os.path.join(BESTPREDS_DIR, "preds")
        src_best_preds_bin = os.path.join(BESTPREDS_DIR, "bin")
        src_best_preds_labels = os.path.join(BESTPREDS_DIR, "labels")
        src_best_preds_originals = os.path.join(BESTPREDS_DIR, "images")
        src_best_preds_probas = os.path.join(BESTPREDS_DIR, "probas")
        os.makedirs(src_best_preds_img, exist_ok=True)
        os.makedirs(src_best_preds_bin, exist_ok=True)
        os.makedirs(src_best_preds_labels, exist_ok=True)
        os.makedirs(src_best_preds_originals, exist_ok=True)
        os.makedirs(src_best_preds_probas, exist_ok=True)
        list_val_sample_names = [val_subset.dataset.images[x] for x in val_subset.indices]

        ckpt_path = os.path.join(RESULTS_DIR, f'checkpoint-{best_step}')
        # best_model = SegformerForSemanticSegmentation.from_pretrained(ckpt_path)
        # model = load_model(IS_TRAINED, ckpt_path, SCALES)
        if IS_TRAINED == 'segmenter':
            model = SegformerForSemanticSegmentation.from_pretrained(ckpt_path)
        else:
            model  = MultiScaleFusionModel.from_pretrained(
                segformer_model_name_or_path=get_best_checkpoint(PRETRAIN_DIR),
                scales=SCALES,
                fusion_checkpoint=ckpt_path,
                num_labels=2,
                device=DEVICE,
                )
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(DEVICE)
        model.eval()
        
        for _, tile_name in tqdm(enumerate(list_val_sample_names), total=len(list_val_sample_names)):
            src_img = os.path.join(DATASET_DIR, 'images', tile_name)
            src_labels = os.path.join(DATASET_DIR, 'labels', tile_name)

            image = torch.tensor(rasterio.open(src_img).read()[:3,...]).to(DEVICE)
            image = image / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3,1,1)
            image = (image - mean) / std

            image = torch.permute(image[torch.newaxis, ...], (0, 1, 2, 3)) if IS_TRAINED=="segmenter" else torch.permute(image[torch.newaxis, ...], (0, 2, 3, 1))
            output = model(image)
            preds = logits_to_preds(logits=output.logits, do_upscale=IS_TRAINED=='segmenter').squeeze(0)
            preds_rgb = np.zeros((preds.shape[0], preds.shape[1], 3), dtype=np.uint8)
            preds_rgb[preds != 0] = 255

            # proba = torch.softmax(output.logits, dim=1)[0,1,...].detach().cpu().numpy()
            proba = torch.nn.functional.interpolate(
                torch.softmax(output.logits, dim=1),
                size=(preds.shape[0], preds.shape[1]),   # (H_lbl, W_lbl)
                mode="bilinear",
                align_corners=False
            )[0,1,...].detach().cpu().numpy()
            # proba_rgb = np.zeros((proba.shape[0], proba.shape[1], 3), dtype=np.uint8)
            # proba_rgb[proba != 0] = 255

            # tiff.imwrite(src_probas_mask, proba_img, compression="zstd", compressionargs={"level": 9})
            # saving images
            Image.fromarray(proba).save(os.path.join(src_best_preds_probas, tile_name))
            Image.fromarray(preds_rgb).save(os.path.join(src_best_preds_img, tile_name))
            tiff.imwrite(os.path.join(src_best_preds_bin, tile_name), preds, compression="zstd", compressionargs={"level": 9})
            shutil.copyfile(src_img, os.path.join(src_best_preds_originals, tile_name))
            shutil.copyfile(src_labels, os.path.join(src_best_preds_labels, tile_name))

    # Save best confidence matrix
    src_best_cm = os.path.join(CONFMAT_DIR, 'values', f"confusion_matrix_ep_{int(dict_best_results['epoch']-1)}.csv")
    if os.path.exists(src_best_cm):
        conf_mat = pd.read_csv(src_best_cm, sep=';', index_col=0).values
        sum_for_recall = np.sum(conf_mat, axis=1).reshape(-1, 1)
        sum_for_precision = np.sum(conf_mat, axis=0).reshape(1, -1)
        show_confusion_matrix(os.path.join(IMG_DIR, 'confusion_matrix.png'), conf_mat, ['Background', 'Landslide'])
        show_confusion_matrix(os.path.join(IMG_DIR, 'confusion_matrix_recall.png'), conf_mat / sum_for_recall, ['Background', 'Landslide'], "Confusion Matrix - Producer accuracy")
        show_confusion_matrix(os.path.join(IMG_DIR, 'confusion_matrix_precision.png'), conf_mat / sum_for_precision, ['Background', 'Landslide'], "Confusion Matrix - User accuracy")
    else:
        print("CONFMAT NOT CREATED FOR BEST EPOCH")
        print("following does not exist:")
        print(src_best_cm)

    # Show duration of process
    delta_time_loop = time() - time_start
    hours = int(delta_time_loop // 3600)
    min = int((delta_time_loop - 3600 * hours) // 60)
    sec = int(delta_time_loop - 3600 * hours - 60 * min)
    print(f"---\n\n==== Training completed in {hours}:{min}:{sec} ====\n")

    # Free memory
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

    return RESULTS_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    args = parser.parse_args()
    cfg_path = args.config

    print(cfg_path)

    if cfg_path != "":
        print("- Training from argument - ")
        args = OmegaConf.load(cfg_path)
    else:
        print("- Training from yaml file - ")
        conf_train = OmegaConf.load('./config/training.yaml')
        conf_dataset = OmegaConf.load('./config/dataset.yaml')

        args= OmegaConf.merge({"train":conf_train, "dataset":conf_dataset})

    training(args)
