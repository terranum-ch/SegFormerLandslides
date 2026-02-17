import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import rasterio
import cv2
import torch

def resize_to_512(img, is_mask=False):
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

    if img.ndim == 3:  # (C, H, W)
        c, _, _ = img.shape
        out = np.zeros((c, 512, 512), dtype=img.dtype)
        for i in range(c):
            out[i] = cv2.resize(img[i], (512, 512), interpolation=interp)
        return out

    else:  # mask (H, W)
        return cv2.resize(img, (512, 512), interpolation=interp)


def center_crop(img, crop_size):
    """
    img: (C, H, W) or (H, W)
    crop_size: int
    """
    if img.ndim == 3:
        _, H, W = img.shape
    else:
        H, W = img.shape

    cy, cx = H // 2, W // 2
    half = crop_size // 2

    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half

    if img.ndim == 3:
        return img[:, y1:y2, x1:x2]
    else:
        return img[y1:y2, x1:x2]
    

def get_multiscale_patch(img_2048, mask_2048=None, scale=1.0):
    """
    img_2048 : (C, 2048, 2048)
    mask_2048: (2048, 2048)
    scale : int  (1, 2, 4)

    Returns
    -------
    img_512, mask_512
    """

    base = 512
    crop_size = int(base / scale)  # 512, 1024, 2048

    img_crop = center_crop(img_2048, crop_size)
    img_512 = resize_to_512(img_crop, is_mask=False)

    mask_512 = None
    if isinstance(mask_2048, np.ndarray):
        mask_crop = center_crop(mask_2048, crop_size)
        mask_512 = resize_to_512(mask_crop, is_mask=True)

    return img_512, mask_512


class SegFusionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, scales, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.scales = scales
        self.transform = transform

        self.images = sorted(os.listdir(image_dir))
        self.masks  = sorted(os.listdir(mask_dir))

        assert len(self.images) == len(self.masks), "Image/mask count mismatch"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path)).astype("int64")

        # Apply augmentation
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        imgs = []

        for s in self.scales:
            img_s, _ = get_multiscale_patch(np.moveaxis(image, 2, 0), mask, s)
            imgs.append(np.moveaxis(img_s, 0, 2))

        # stack for model
        imgs = np.stack(imgs)      # (K, C, 512, 512)
        mask = center_crop(mask, 512)

        # normalize imgs:
        imgs = imgs.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])[None, None, None, :]
        std  = np.array([0.229, 0.224, 0.225])[None, None, None, :]
        imgs = (imgs - mean) / std

        # HF returns tensors with extra batch dim, we remove manually
        inputs = {
            "multspec_img": torch.from_numpy(imgs).float(),
            'labels': torch.from_numpy(mask).long(),
            'filename': self.images[idx]
        }

        return inputs


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.transform = transform

        self.images = sorted(os.listdir(image_dir))
        self.masks  = sorted(os.listdir(mask_dir))

        assert len(self.images) == len(self.masks), "Image/mask count mismatch"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # image = np.array(Image.open(img_path).convert("RGB"))
        # mask = np.array(Image.open(mask_path)).astype("int64")
        image = np.moveaxis(rasterio.open(img_path).read(), 0, 2)[..., :3]
        mask = np.moveaxis(rasterio.open(mask_path).read(), 0, 2)

        # Apply augmentation
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if self.processor is not None:
            inputs = self.processor(images=image, segmentation_maps=mask, return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].squeeze(0)
            inputs["labels"] = inputs["labels"].squeeze(0)
            inputs['filename'] = self.images[idx]
        else:
            imgs = image.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])[None, None, :]
            std  = np.array([0.229, 0.224, 0.225])[None, None, :]
            imgs = (imgs - mean) / std

            # HF returns tensors with extra batch dim, we remove manually
            inputs = {
                "pixel_values": torch.from_numpy(imgs).float(),
                'labels': torch.from_numpy(mask).squeeze(-1).long(),
                'filename': self.images[idx]
            }

        
        return inputs
    

class DatasetProxy:
    def __init__(
        self,
        mode,  # "segmenter" or "fusion"
        image_dir,
        mask_dir,
        processor,
        transform=None,
        scales=(1.0, 0.75, 0.5, 0.25),
    ):
        self.mode = mode
        mode = 'segmenter'
        if mode == "fusion":
            self.dataset = SegFusionDataset(
                image_dir=image_dir,
                mask_dir=mask_dir,
                transform=transform,
                scales=scales,
            )
        else:
            self.dataset = SegmentationDataset(
                image_dir=image_dir,
                mask_dir=mask_dir,
                processor=processor,
                transform=transform,
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    

if __name__ == "__main__":
    std  = np.array([0.229, 0.224, 0.225])[None, None, None, :]
    test = np.zeros((4,512,512,3))
    print(std.shape)
    # std = std[np.newaxis, ...]
    # print(std.shape)
    # print(np.vstack([std]*4).shape)
    test = test / std
    quit()





    from transformers import AutoImageProcessor
    import matplotlib.pyplot as plt
    from math import ceil

    src_test = r"D:\Terranum_SD\99_Data\Landslide\data\Bern_glissements_spontane_shpfiles\2048_samples\dummy_test\tiles"
    PRETRAINED_MODEL = "results/training/20260122_225022_100_epochs_Longxihe_Bern_v2_focal_dice_losses_with_900_false_pos_from_scratch_da_scale"
    PRETRAINED_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
    scales = [1.0, 0.75, 0.5, 0.25]
    processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL, use_fast=True)
    dataset = DatasetProxy(
        mode='fusion',
        image_dir=os.path.join(src_test, 'images'),
        mask_dir=os.path.join(src_test, 'masks'),
        processor=processor,
        transform=None,
        scales=scales,
    )
    first_batch = dataset[0]

    num_figs = len(scales)
    num_cols = ceil(num_figs**0.5)
    num_rows = ceil(num_figs / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    axs = axs.flatten()
    for id_ax in range(len(axs)):
        img = np.array(first_batch['multspec_img'])
        img_arr = Image.fromarray(img[id_ax,...])
        axs[id_ax].imshow(img_arr)
        axs[id_ax].set_title(f"scale = {scales[id_ax]}")
    plt.show()
    plt.close()

    print()