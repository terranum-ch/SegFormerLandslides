import os
import numpy as np
import pandas as pd
from math import prod
import time
import psutil
from tqdm import tqdm
from transformers import Trainer, SegformerForSemanticSegmentation
import torch
from torch.utils.data import default_collate
from torch.utils.data import Subset
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers.models.segformer.modeling_segformer import (
    SegformerPreTrainedModel,
    SegformerModel,
    SegformerDecodeHead,
    SemanticSegmenterOutput,
)
from utils.visualization import show_confusion_matrix

from PIL import Image
from .metrics import compute_metrics, compute_cm_from_dict

# for evaluation_loop overwrite
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import EvalLoopContainer, find_batch_size, IterableDatasetShard
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput, denumpify_detensorize, has_length
from transformers.utils import logging
from safetensors.torch import load_file

logger = logging.get_logger(__name__)


def collate_with_filename(batch):
    
    # Normal collate for tensor fields
    batch_collated = default_collate([{k: v for k, v in sample.items() if k != "filename"} 
                                      for sample in batch])

    # Keep filenames as a simple list
    batch_collated["filename"] = [sample["filename"] for sample in batch]

    return batch_collated


def logits_to_preds(logits, do_upscale=True):
    # Resize predictions to match label size
    if do_upscale:
        logits = torch.nn.functional.interpolate(
            torch.tensor(logits),
            size=(logits.shape[-2]*4, logits.shape[-1]*4),   # (H_lbl, W_lbl)
            mode="bilinear",
            align_corners=False
        )

    # Final predicted class mask
    return logits.argmax(dim=1).cpu().numpy()  # (batch_size, H_lbl, W_lbl)
    
def dice_loss(logits, targets, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    targets_onehot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2)

    intersection = (probs * targets_onehot).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    ignore_index: int = 255,
):
    """
    logits: (B, 2, H, W)
    targets: (B, H, W)
    """

    # 1) Convert logits into probabilities (softmax)
    probs = torch.softmax(logits, dim=1)

    # 2) Pick probability of the class actually present
    targets_expanded = F.one_hot(targets, num_classes=2).permute(0, 3, 1, 2)

    p_t = (probs * targets_expanded).sum(dim=1) + 1e-8

    # 3) Focal term
    focal_term = (1 - p_t) ** gamma

    # 4) Cross entropy on probabilities
    ce_loss = - torch.log(p_t)

    # 5) Apply alpha weighting
    loss = alpha * focal_term * ce_loss

    # 6) Mask ignored pixels
    if ignore_index is not None:
        mask = (targets != ignore_index).float()
        loss = loss * mask

    return loss.mean()


class TrainValMetricsTrainer(Trainer):
    def __init__(self, confmat_dir, label_smoothing=0.1, loss_weights='auto', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.training_metrics = []
        self.training_losses = []
        self.confmat_dir = confmat_dir
        self.label_smoothing = label_smoothing

        self.cf_dir_img = os.path.join(confmat_dir, "images")
        self.cf_dir_val = os.path.join(confmat_dir, "values")

        os.makedirs(confmat_dir, exist_ok=True)
        os.makedirs(self.cf_dir_img, exist_ok=True)
        os.makedirs(self.cf_dir_val, exist_ok=True)

        # compute weights for loss
        if loss_weights == 'auto':
            dataset = self.train_dataset.dataset if isinstance(self.train_dataset, Subset) else self.train_dataset
            count_ls = 0
            count_bck = 0
            print("Computing weights...")
            for samp_id in tqdm(range(len(dataset)), total=len(dataset)):
                inputs = dataset[samp_id]
                labels = inputs['labels']
                count_ls += torch.sum(labels == 1)
                count_bck += torch.sum(labels == 0)
            
            tot = count_ls + count_bck
            self.class_weights = torch.Tensor([tot/count_bck, tot/count_ls]).to('cuda:0')
        else:
            self.class_weights = torch.Tensor(loss_weights).to('cuda')
        print(f"Weights:\n\tBackground: {self.class_weights[0]}\n\tLandslide: {self.class_weights[1]}")

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        inputs = self._prepare_inputs(inputs)
        labels = inputs["labels"]

        # ---- Compute loss AND get outputs ----
        loss, outputs = self.compute_loss(
            model, inputs, return_outputs=True
        )

        # ---- Standard HF backward ----
        self.accelerator.backward(loss)

        # ---- Compute training metrics safely ----
        with torch.no_grad():
            logits = outputs.logits
            if isinstance(model, SegformerForSemanticSegmentation):

                logits = torch.nn.functional.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
            # else:   # if model is fusion
            #     logits = outputs[0].logits
        
            # logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0].logits # (B, C, H, W)
            # logits = torch.nn.functional.interpolate(
            #     logits,
            #     size=labels.shape[-2:],
            #     mode="bilinear",
            #     align_corners=False
            # )

            preds = logits.detach().cpu().numpy()
            labs = labels.detach().cpu().numpy()

            metrics = compute_metrics({
                "predictions": preds,
                "label_ids": labs
            })

        # ---- Log properly (this writes into trainer_state.json) ----
        metrics["train_loss"] = loss.detach().cpu().item()
        self.training_metrics.append(metrics)
        self.training_losses.append(loss.cpu().detach().numpy())

        return loss.detach()

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        model = self._wrap_model(self.model, training=False)
        model.eval()

        total_loss = 0.0
        n_samples = 0

        metrics = {}
        confmat = np.zeros((2,2), dtype=np.uint64)
        for step, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs = self._prepare_inputs(inputs)
            if self.state.epoch == 0:
                self.preds_filenames.append(inputs['filename'])
            with torch.no_grad():
                loss, logits, labels = self.prediction_step(
                    model,
                    inputs,
                    prediction_loss_only,
                    ignore_keys=ignore_keys,
                )
            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            n_samples += batch_size

            # Convert logits to predictions
            preds = logits_to_preds(logits, do_upscale=logits.shape[-1] < labels.shape[-1])
            labels = labels.cpu()
            if step == 0:
                metrics = {key: val for key, val in self.compute_metrics(
                    EvalPrediction(predictions=logits.cpu(), label_ids=labels.cpu())
                ).items()}
            else:
                for key, val in self.compute_metrics(EvalPrediction(predictions=logits.cpu(), label_ids=labels)).items():
                    metrics[key] += val * batch_size

            dict_conf_mat = {x: (preds[x,...], labels[x,...]) for x in range(preds.shape[0])}
            confmat += compute_cm_from_dict(dict_conf_mat)

            # 🔥 IMPORTANT: free memory
            del logits, preds, labels
            torch.cuda.empty_cache()

        # Final metrics
        metrics = {f"{metric_key_prefix}_{key}": float(val / n_samples) for key,val in metrics.items()}
        metrics[f"{metric_key_prefix}_loss"] = float(total_loss / n_samples)

        # Save confusion matrix
        show_confusion_matrix(
            saving_loc=os.path.join(self.cf_dir_img, f"confusion_matrix_ep_{int(self.state.epoch - 1)}.jpg"),
            conf_mat=confmat,
            class_labels=['Background', 'Landslide'],
            )
        pd.DataFrame(confmat, index=[0,1], columns=[0,1]).to_csv(os.path.join(self.cf_dir_val, f"confusion_matrix_ep_{int(self.state.epoch - 1)}.csv"), sep=';')

        return type(
            "EvalLoopOutput",
            (),
            {
                "predictions": None,
                "label_ids": None,
                "metrics": metrics,
                "num_samples": n_samples,
            },
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # # if hasattr(inputs, 'labels'):
        # labels = inputs.pop("labels")
        # try:
        #     filenames = inputs.pop("filename")
        # except:
        #     pass

        # outputs = model(**inputs)
        # logits = outputs.logits if hasattr(outputs, 'logits') else outputs # (B, C, H, W)
        
        # # Resize logits to match labels
        # logits = torch.nn.functional.interpolate(
        #     logits,
        #     size=labels.shape[-2:],
        #     mode="bilinear",
        #     align_corners=False
        # )

        # # ce_loss = F.cross_entropy(
        # #     logits,
        # #     labels,
        # #     weight=self.class_weights,
        # #     ignore_index=255  # very important for segmentation
        # # )

        # B, C, H, W = logits.shape

        # # --------------------------
        # # Label smoothing
        # # --------------------------
        # epsilon = 0.05  # smoothing factor

        # # Convert labels to one-hot
        # labels_one_hot = torch.nn.functional.one_hot(
        #     labels.long(),
        #     num_classes=C
        # )  # (B, H, W, C)

        # labels_one_hot = labels_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # # Apply smoothing
        # labels_smooth = labels_one_hot * (1 - epsilon) + epsilon / C

        # # --------------------------
        # # Loss computation
        # # ---
        # f_loss = focal_loss(logits, labels_smooth)
        # d_loss = dice_loss(logits, labels_smooth)
        # loss = f_loss + 0.5 * d_loss

        # return (loss, outputs) if return_outputs else loss

        labels = inputs.pop("labels")

        try:
            filenames = inputs.pop("filename")
        except:
            pass

        outputs = model(**inputs)
        if isinstance(model, SegformerForSemanticSegmentation):
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        else:   # if model is fusion
            outputs = outputs[0]
            logits = outputs.logits

        ce_loss = torch.nn.functional.cross_entropy(
            logits,
            labels.long(),
            weight=self.class_weights,
            label_smoothing=self.label_smoothing
        )
        # f_loss = focal_loss(logits, labels)
        d_loss = dice_loss(logits, labels)

        # loss = f_loss + 0.5 * d_loss
        loss = ce_loss + 0.5 * d_loss

        return (loss, outputs) if return_outputs else loss


class ScaleAttention(nn.Module):
    def __init__(self, n_scales, n_classes):
        super().__init__()
        self.n_scales = n_scales
        self.n_classes = n_classes

        in_ch = n_scales * n_classes

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, n_scales, 1)
        )

    def forward(self, stacked_logits):
        # stacked_logits: [B, K*C, H, W]
        B, KC, H, W = stacked_logits.shape
        K = self.n_scales
        C = self.n_classes

        # Compute weights per scale
        weights = torch.sigmoid(self.net(stacked_logits))  # [B, K, H, W]   # new
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)   # new

        # Reshape logits to separate scales and classes
        logits = stacked_logits.view(B, K, C, H, W)

        # Apply weights to all classes of each scale
        fused = (weights.unsqueeze(2) * logits).sum(dim=1)  # [B, C, H, W]

        return fused, weights
        


    
def sliding_window_inference(model, image, window=512, stride=512, device="cuda"):
    """
    image: [1, 3, H, W]
    returns: stitched logits [1, C, H, W]
    """
    overlap = window - stride
    B, _, H, W = image.shape
    C = model.config.num_labels

    output = torch.zeros((B, C, H, W), device=device)
    counter = torch.zeros((B, 1, H, W), device=device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):

            y1 = min(y + window, H)
            x1 = min(x + window, W)

            patch = image[:, :, y:y1, x:x1]

            # pad if needed
            pad_h = window - patch.shape[2]
            pad_w = window - patch.shape[3]

            if pad_h > 0 or pad_w > 0:
                patch = F.pad(patch, (0, pad_w, 0, pad_h))

            with torch.no_grad():
                # with torch.amp.autocast(device_type='cuda'):
                logits = model(pixel_values=patch).logits

                logits = F.interpolate(
                        logits,
                        size=(window, window),
                        mode="bilinear",
                        align_corners=False
                    )
            logits = logits[:, :, :y1-y, :x1-x]

            y2 = y + int(overlap/2) if y != 0 else 0
            x2 = x + int(overlap/2) if x != 0 else 0
            y3 = y + window - int(overlap/2)
            x3 = x + window - int(overlap/2)
            x0_log = int(overlap/2) if x2 != 0 else 0
            y0_log = int(overlap/2) if y2 != 0 else 0
            output[:, :, y2:y3, x2:x3] += logits[:, :, y0_log:window - int(overlap/2), x0_log:window - int(overlap/2)]

            # output[:, :, y:y1, x:x1] += logits
            # counter[:, :, y:y1, x:x1] += 1

    return output# / counter


def multiscale_logits(model, image_2048, scales, device="cuda"):
    """
    image_2048: [1, 3, 2048, 2048]
    returns: list of aligned logits [1, C, 2048, 2048] per scale
    """

    logits_per_scale = []

    for s in scales:

        # Resize scene
        H_scaled = int(2048 * s)
        W_scaled = int(2048 * s)

        img_scaled = F.interpolate(
            image_2048,
            size=(H_scaled, W_scaled),
            mode="bilinear",
            align_corners=False
        )

        # Sliding window inference
        logits_scaled = sliding_window_inference(
            model,
            img_scaled,
            window=512,
            stride=512,
            device=device
        )

        # Resize logits back to 2048
        logits_resized = F.interpolate(
            logits_scaled,
            size=(2048, 2048),
            mode="bilinear",
            align_corners=False
        )

        # Normalize logits per scale (important)
        logits_resized = logits_resized / (
            logits_resized.std(dim=(2,3), keepdim=True) + 1e-6
        )

        logits_per_scale.append(logits_resized)

    return logits_per_scale

# class ScaleAttention(nn.Module):
    def __init__(self, n_scales, n_classes):
        super().__init__()
        self.n_scales = n_scales
        self.n_classes = n_classes

        in_ch = n_scales * n_classes

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> [B, 32, 1, 1]
            nn.Conv2d(32, n_scales * n_classes, 1)  # -> [B, K*C, 1, 1]
        )

    def forward(self, stacked_logits):
        """
        stacked_logits: [B, K*C, H, W]
        """

        B, KC, H, W = stacked_logits.shape
        K = self.n_scales
        C = self.n_classes

        # === Compute class-aware raw weights ===
        raw_weights = self.net(stacked_logits)  # [B, K*C, 1, 1]

        # reshape -> [B, K, C, 1, 1]
        weights = raw_weights.view(B, K, C, 1, 1)

        # Sigmoid gating
        weights = torch.sigmoid(weights)

        # Normalize across scales PER CLASS
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        # === Separate logits ===
        logits = stacked_logits.view(B, K, C, H, W)

        # === Weighted fusion per class ===
        fused = (weights * logits).sum(dim=1)  # [B, C, H, W]

        return fused, weights


class MultiScaleFusionModel(nn.Module):
    def __init__(self, segformer, scales, device=None):
        super().__init__()

        self.segformer = segformer.to(device)
        self.segformer.eval()  # frozen

        self.scales = scales
        self.fusion = ScaleAttention(
            n_classes= 2,
            n_scales=len(self.scales),
        )

        self.device=device

    def forward(self, pixel_values, labels=None, return_weights=False):

        B, H, W, C = pixel_values.shape
        device = pixel_values.device

        logits_per_scale = []

        with torch.no_grad():  # freeze segmentation
            for s in self.scales:

                # 1️⃣ Resize full scene
                Hs = int(H * s)
                Ws = int(W * s)

                img_scaled = F.interpolate(
                    torch.moveaxis(pixel_values,3,1),
                    size=(Hs, Ws),
                    mode="bilinear",
                    align_corners=False
                )

                # 2️⃣ Sliding window inference
                logits_scaled = sliding_window_inference(
                    self.segformer,
                    img_scaled,
                    window=512,
                    stride=256,
                    device=device
                )

                # 3️⃣ Resize logits back to original resolution
                logits_resized = F.interpolate(
                    logits_scaled,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False
                )

                # 4️⃣ Normalize per scale
                logits_resized = logits_resized / (
                    logits_resized.std(dim=(2,3), keepdim=True) + 1e-6
                )

                logits_per_scale.append(logits_resized)

        # 5️⃣ Stack
        stacked_logits = torch.cat(logits_per_scale, dim=1)

        # 6️⃣ Fuse
        fused_logits, weights = self.fusion(stacked_logits)

        # print(f"Weights: {weights.mean(dim=(0,2,3))}")
        # print("RAM used: ", psutil.virtual_memory().percent, "%, \nVRAM used: ", round(torch.cuda.memory_allocated()/10**9, 2), "Go")
        # print('---')

        output = SemanticSegmenterOutput(
            loss=None,
            logits=fused_logits,
            hidden_states=None,
            attentions=None,
            )
        
        if return_weights == False:
            weights = None

        return output, weights

    @classmethod
    def from_pretrained(
        cls,
        segformer_model_name_or_path,
        scales,
        fusion_checkpoint=None,
        num_labels=2,
        device=None,
        **kwargs
    ):
        """
        Load a pretrained SegFormer and wrap it with the fusion module.

        Args:
            segformer_model_name_or_path (str):
                HuggingFace model name or local checkpoint path.
            scales (list[float]):
                List of scale factors.
            fusion_checkpoint (str, optional):
                Path to saved fusion model weights.
            num_labels (int):
                Number of segmentation classes.
            map_location (str or torch.device):
                Device for loading checkpoints.
            **kwargs:
                Extra args forwarded to HF from_pretrained.

        Returns:
            MultiScaleFusionModel
        """

        # 1️⃣ Load pretrained SegFormer
        segformer = SegformerForSemanticSegmentation.from_pretrained(
            segformer_model_name_or_path,
            num_labels=num_labels,
            # ignore_mismatched_sizes=True,
        )

        # 2️⃣ Build fusion model wrapper
        model = cls(
            segformer=segformer,
            scales=scales,
            device=device,
        )

        # 3️⃣ Optionally load fusion weights
        if fusion_checkpoint is not None:
            ckpt_path = os.path.join(fusion_checkpoint, "model.safetensors")

            state_dict = load_file(ckpt_path, device=device)
            fusion_state_dict = {
                k: v for k, v in state_dict.items()
                if k.startswith("fusion")
            }

            model.load_state_dict(fusion_state_dict, strict=False)
        return model.to(device)


if __name__ == "__main__":
    print("WRONG SCRIPT PAL")

