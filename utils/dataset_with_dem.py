import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, dem_dir, processor, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.dem_dir = dem_dir
        self.processor = processor
        self.transform = transform

        self.images = sorted(os.listdir(image_dir))
        self.masks  = sorted(os.listdir(mask_dir))
        self.dem = sorted(os.listdir(dem_dir))
        # print(len(self.images))
        # print(len(self.masks))
        # print(len(self.images))
        assert len(self.images) == len(self.masks) == len(self.dem), "Image/mask count mismatch"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        dem_path = os.path.join(self.dem_dir, self.dem[idx])

        mask = np.array(Image.open(mask_path)).astype(np.uint8)
        image = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
        image[:,:,:3] = np.array(Image.open(img_path).convert('RGB'))
        image[:,:,3] = np.array(Image.open(dem_path))

        # Apply augmentation
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # inputs = self.processor(images=image, segmentation_maps=mask, return_tensors="pt")

        # Normalize RGB
        image[..., :3] /= 255.0

        # Normalize DEM (example)
        dem_mean = np.mean(image[..., 3])
        dem_std = np.std(image[..., 3])
        image[..., 3] = (image[..., 3] - dem_mean) / (dem_std + 1e-6)

        pixel_values = torch.from_numpy(image).permute(2, 0, 1)  # C,H,W
        labels = torch.from_numpy(mask).long()

        # HF returns tensors with extra batch dim, we remove manually
        # inputs["pixel_values"] = inputs["pixel_values"].squeeze(0)
        # inputs["labels"] = inputs["labels"].squeeze(0)
        # inputs['pixel_values'] = pixel_values
        # inputs['labels'] = labels
        # inputs['filename'] = self.images[idx]
        inputs = {
            'pixel_values': pixel_values,
            'labels': labels,
            'filename': self.images[idx]
        }
        return inputs
    

if __name__ == "__main__":
    pass