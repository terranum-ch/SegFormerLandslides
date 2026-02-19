import numpy as np
import os
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
    def __init__(self, confmat_dir, confmat_buffer_size=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Buffers for storing batch results during an epoch
        self.training_metrics = []
        self.training_losses = []
        self.confmat_dir = confmat_dir
        self.confmat = np.zeros((2,2), dtype=np.uint64)
        self.eval_preds = {}

        # # compute weights for loss
        # dataset = self.train_dataset.dataset if isinstance(self.train_dataset, Subset) else self.train_dataset
        # count_ls = 0
        # count_bck = 0
        # print("Computing weights...")
        # for samp_id in tqdm(range(len(dataset)), total=len(dataset)):
        #     inputs = dataset[samp_id]
        #     labels = inputs['labels']
        #     count_ls += torch.sum(labels == 1)
        #     count_bck += torch.sum(labels == 0)
        
        # tot = count_ls + count_bck
        # self.class_weights = torch.Tensor([tot/count_bck, tot/count_ls]).to('cuda:0')
        # print(f"Weights:\n\tBackground: {self.class_weights[0]}\n\tLandslide: {self.class_weights[1]}")


    @staticmethod
    def logits_to_preds(logits):
        # Resize predictions to match label size
        if 2048 not in logits.shape:
            logits = torch.nn.functional.interpolate(
                torch.tensor(logits),
                size=(logits.shape[-2]*4, logits.shape[-1]*4),   # (H_lbl, W_lbl)
                mode="bilinear",
                align_corners=False
            )

        # Final predicted class mask
        return logits.argmax(dim=1).cpu().numpy()  # (batch_size, H_lbl, W_lbl)
    
    # def training_step(self, model, inputs, num_items_in_batch=None):
    #     # Standard HF forward pass
    #     #   remove the filename for training
    #     inputs = {k: v for k, v in inputs.items() if k not in ["filename"]}
    #     outputs = model(**inputs)
    #     loss = outputs.loss

    #     # Compute logits without affecting backward
    #     with torch.no_grad():
    #         logits = outputs.logits
    #         labels = inputs["labels"]

    #     logits = logits.detach().cpu().numpy()
    #     labels = labels.detach().cpu().numpy().astype(np.uint8)
    #     # preds = self.logits_to_preds(logits).astype(np.uint8)

    #     # Save them for end-of-epoch metrics
    #     dict_for_metrics = {'predictions': logits, "label_ids": labels}
    #     # dict_conf_mat = {x: (preds[x,...], labels[x,...]) for x in range(preds.shape[0])}
    #     # self.confmat += compute_cm_from_dict(dict_conf_mat)

    #     metrics = compute_metrics(dict_for_metrics)
    #     self.training_metrics.append(metrics)
    #     self.training_losses.append(loss.cpu().detach().numpy())

    #     # Standard HF backward
    #     self.accelerator.backward(loss)
    #     return loss.detach()

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
            # logits = outputs.logits
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs # (B, C, H, W)
            # labels = inputs["labels"]

            logits = torch.nn.functional.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            preds = logits.detach().cpu().numpy()
            labs = labels.detach().cpu().numpy()

            metrics = compute_metrics({
                "predictions": preds,
                "label_ids": labs
            })

        # ---- Log properly (this writes into trainer_state.json) ----
        # metrics = {f"train_{k}": v for k, v in metrics.items()}
        metrics["train_loss"] = loss.detach().cpu().item()
        self.training_metrics.append(metrics)
        self.training_losses.append(loss.cpu().detach().numpy())
        # self.log(metrics)

        return loss.detach()

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        import torch
        from torch.nn import functional as F

        model = self._wrap_model(self.model, training=False)
        model.eval()

        total_loss = 0.0
        n_samples = 0
    #         # # -----------------------------
    #         # # ------- custom lines --------
    #         # # -----------------------------

    #         # # Prediction step

    #         # filenames = inputs['filename']
    #         # inputs = {k:v for k,v in inputs.items() if k != 'filename'}
    #         # losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

    #         # #   split batches to add each sample
    #         # preds = self.logits_to_preds(logits)
    #         # for sample_id in range(preds.shape[0]):
    #         #     self.eval_preds[filenames[sample_id]] = preds[sample_id, ...]
    #         # labels_cpu = labels.cpu().detach().clone()
    #         # dict_conf_mat = {x: (preds[x,...], labels_cpu[x,...]) for x in range(preds.shape[0])}
    #         # self.confmat += compute_cm_from_dict(dict_conf_mat)
                
    #         # # -----------------------------
    #         # # -----------------------------
        metrics = {}
        for step, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs = self._prepare_inputs(inputs)

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
            preds = torch.argmax(logits, dim=1).cpu()
            labels = labels.cpu()
            if step == 0:
                metrics = {key: val for key, val in self.compute_metrics(
                    EvalPrediction(predictions=logits.cpu(), label_ids=labels.cpu())
                ).items()}
            else:
                for key, val in self.compute_metrics(EvalPrediction(predictions=logits.cpu(), label_ids=labels)).items():
                    metrics[key] += val * batch_size

            dict_conf_mat = {x: (preds[x,...], labels[x,...]) for x in range(preds.shape[0])}
            self.confmat += compute_cm_from_dict(dict_conf_mat)

            # # Flatten
            # preds = preds.view(-1)
            # labels = labels.view(-1)

            # # Remove ignore index if any (e.g. 255)
            # mask = labels != 255
            # preds = preds[mask]
            # labels = labels[mask]

            # 🔥 IMPORTANT: free memory
            del logits, preds, labels
            torch.cuda.empty_cache()

        # Final metrics
        metrics = {f"{metric_key_prefix}_{key}": float(val / n_samples) for key,val in metrics.items()}
        metrics[f"{metric_key_prefix}_loss"] = float(total_loss / n_samples)

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
        # if hasattr(inputs, 'labels'):
        labels = inputs.pop("labels")
        try:
            filenames = inputs.pop("filename")
        except:
            pass

        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs # (B, C, H, W)
        
        # Resize logits to match labels
        logits = torch.nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # ce_loss = F.cross_entropy(
        #     logits,
        #     labels,
        #     weight=self.class_weights,
        #     ignore_index=255  # very important for segmentation
        # )
        f_loss = focal_loss(logits, labels)

        d_loss = dice_loss(logits, labels)
        # print("Weighted CE loss: ", ce_loss)
        # print("Dice loss: ", d_loss)
        # print("Focal loss: ", f_loss)
        # loss = ce_loss + 0.5 * d_loss
        loss = f_loss + 0.5 * d_loss

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
        # weights = torch.softmax(self.net(stacked_logits), dim=1)  # [B, K, H, W]
        weights = torch.sigmoid(self.net(stacked_logits))  # [B, K, H, W]   # new
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)   # new

        # Reshape logits to separate scales and classes
        logits = stacked_logits.view(B, K, C, H, W)

        # Apply weights to all classes of each scale
        fused = (weights.unsqueeze(2) * logits).sum(dim=1)  # [B, C, H, W]

        return fused, weights
        

# class MultiScaleSegformer(SegformerPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)

#         self.segformer = SegformerModel(config)
#         self.decode_head = SegformerDecodeHead(config)

#         # Freeze backbone
#         for p in self.segformer.parameters():
#             p.requires_grad = False
#         for p in self.decode_head.parameters():
#             p.requires_grad = False

#         self.scales = [float(s) for s in config.scales]
#         self.n_scales = len(self.scales)
#         self.n_classes = config.num_labels

#         self.fusion = ScaleAttention(self.n_scales, self.n_classes)

#         self.post_init()

#     def forward(
#         self,
#         pixel_values=None,          # unused but required by Trainer
#         multspec_img=None,         # [B, K, C, 512, 512]
#         labels=None,
#         filename=None,
#         return_dict=None,
#         return_weights=False,
#         **kwargs
#     ):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         _, K, H, W, _ = multspec_img.shape
#         logits_per_scale = []

#         # === Run SegFormer on each provided scale ===
#         for k in range(K):
#             xk = multspec_img[:, k]  # [B, C, 512, 512]
#             xk = torch.moveaxis(xk, 3, 1)

#             with torch.no_grad():
#                 outputs = self.segformer(
#                     xk,
#                     output_hidden_states=True,
#                     return_dict=True,
#                 )
#                 logits_k = self.decode_head(outputs.hidden_states)

#             logits_k = F.interpolate(
#                 logits_k, size=(H, W), mode="bilinear", align_corners=False
#             )
#             logits_k = logits_k / (logits_k.std(dim=(2,3), keepdim=True) + 1e-6)    # new
#             logits_per_scale.append(logits_k)

#         # === Fuse ===
#         stacked = torch.cat(logits_per_scale, dim=1)  # [B, K*C, H, W]
#         fused_logits, weights = self.fusion(stacked)  # [B, C, H, W]

#         # SegFormer outputs at 1/4 res
#         logits = F.interpolate(
#             fused_logits,
#             scale_factor=0.25,
#             mode="bilinear",
#             align_corners=False
#         )
#         if return_weights:
#             return SemanticSegmenterOutput(
#                 loss=None,
#                 logits=logits,
#                 hidden_states=None,
#                 attentions=None,
#             ), weights
#         else:
#             return SemanticSegmenterOutput(
#                 loss=None,
#                 logits=logits,
#                 hidden_states=None,
#                 attentions=None,
#             )


# ==================================================================


# class ScaleAttention(nn.Module):
#     def __init__(self, n_scales, n_classes):
#         super().__init__()
#         self.n_scales = n_scales
#         self.n_classes = n_classes

#         in_ch = n_scales * n_classes

#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, 32, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d(1),   # -> [B, 32, 1, 1]
#             nn.Conv2d(32, n_scales, 1) # -> [B, K, 1, 1]
#         )

#     def forward(self, stacked_logits):
#         """
#         stacked_logits: [B, K*C, H, W]
#         """

#         B, KC, H, W = stacked_logits.shape
#         K = self.n_scales
#         C = self.n_classes

#         # === Compute raw weights ===
#         weights = torch.sigmoid(self.net(stacked_logits))  # [B, K, 1, 1]

#         # Normalize across scales (no competition pressure)
#         weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

#         # === Separate logits ===
#         logits = stacked_logits.view(B, K, C, H, W)

#         # === Weighted fusion ===
#         fused = (weights.unsqueeze(2) * logits).sum(dim=1)  # [B, C, H, W]

#         return fused, weights
    

# class MultiScaleSegformer(SegformerPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)

#         self.segformer = SegformerModel(config)
#         self.decode_head = SegformerDecodeHead(config)

#         # Freeze backbone
#         for p in self.segformer.parameters():
#             p.requires_grad = False
#         for p in self.decode_head.parameters():
#             p.requires_grad = False

#         self.scales = [float(s) for s in config.scales]
#         self.n_scales = len(self.scales)
#         self.n_classes = config.num_labels

#         self.fusion = ScaleAttention(self.n_scales, self.n_classes)

#         self.post_init()

#     def forward(
#         self,
#         pixel_values=None,          # unused but required by Trainer
#         multspec_img=None,          # [B, K, H, W, C]
#         labels=None,
#         filename=None,
#         return_dict=None,
#         return_weights=False,
#         **kwargs
#     ):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         B, K, H, W, C = multspec_img.shape
#         logits_per_scale = []

#         # === Run frozen SegFormer per scale ===
#         for k in range(K):

#             # [B, H, W, C] -> [B, C, H, W]
#             xk = multspec_img[:, k].permute(0, 3, 1, 2)

#             with torch.no_grad():
#                 outputs = self.segformer(
#                     xk,
#                     output_hidden_states=True,
#                     return_dict=True,
#                 )
#                 logits_k = self.decode_head(outputs.hidden_states)

#             # Upsample to common resolution (H, W)
#             logits_k = F.interpolate(
#                 logits_k,
#                 size=(H, W),
#                 mode="bilinear",
#                 align_corners=False,
#             )

#             # 🔥 IMPORTANT: Normalize logits per scale
#             logits_k = logits_k / (logits_k.std(dim=(2,3), keepdim=True) + 1e-6)

#             logits_per_scale.append(logits_k)

#         # === Stack logits ===
#         stacked = torch.cat(logits_per_scale, dim=1)  # [B, K*C, H, W]

#         # === Fuse ===
#         fused_logits, weights = self.fusion(stacked)  # [B, C, H, W]

#         # No extra scaling here — keep native resolution
#         logits = fused_logits

#         print(f"Weights: {weights.mean(dim=0)}")

#         if return_weights:
#             return (
#                 SemanticSegmenterOutput(
#                     loss=None,
#                     logits=logits,
#                     hidden_states=None,
#                     attentions=None,
#                 ),
#                 weights
#             )
#         else:
#             return SemanticSegmenterOutput(
#                 loss=None,
#                 logits=logits,
#                 hidden_states=None,
#                 attentions=None,
#             )



# ==================================================================



# class ScaleAttention(nn.Module):
#     def __init__(self, n_scales, n_classes):
#         super().__init__()
#         self.n_scales = n_scales
#         self.n_classes = n_classes

#         in_ch = n_scales * n_classes

#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, 32, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d(1),  # -> [B, 32, 1, 1]
#             nn.Conv2d(32, n_scales * n_classes, 1)  # -> [B, K*C, 1, 1]
#         )

#     def forward(self, stacked_logits):
#         """
#         stacked_logits: [B, K*C, H, W]
#         """

#         B, KC, H, W = stacked_logits.shape
#         K = self.n_scales
#         C = self.n_classes

#         # === Compute class-aware raw weights ===
#         raw_weights = self.net(stacked_logits)  # [B, K*C, 1, 1]

#         # reshape -> [B, K, C, 1, 1]
#         weights = raw_weights.view(B, K, C, 1, 1)

#         # Sigmoid gating
#         weights = torch.sigmoid(weights)

#         # Normalize across scales PER CLASS
#         weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

#         # === Separate logits ===
#         logits = stacked_logits.view(B, K, C, H, W)

#         # === Weighted fusion per class ===
#         fused = (weights * logits).sum(dim=1)  # [B, C, H, W]

#         return fused, weights


# class MultiScaleSegformer(SegformerPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)

#         self.segformer = SegformerModel(config)
#         self.decode_head = SegformerDecodeHead(config)

#         # Freeze backbone
#         for p in self.segformer.parameters():
#             p.requires_grad = False
#         for p in self.decode_head.parameters():
#             p.requires_grad = False

#         self.scales = [float(s) for s in config.scales]
#         self.n_scales = len(self.scales)
#         self.n_classes = config.num_labels

#         self.fusion = ScaleAttention(self.n_scales, self.n_classes)

#         self.post_init()

#     def forward(
#         self,
#         pixel_values=None,
#         multspec_img=None,  # [B, K, H, W, C]
#         labels=None,
#         filename=None,
#         return_dict=None,
#         return_weights=False,
#         **kwargs
#     ):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         B, K, H, W, C = multspec_img.shape
#         logits_per_scale = []

#         for k in range(K):

#             xk = multspec_img[:, k].permute(0, 3, 1, 2)

#             with torch.no_grad():
#                 outputs = self.segformer(
#                     xk,
#                     output_hidden_states=True,
#                     return_dict=True,
#                 )
#                 logits_k = self.decode_head(outputs.hidden_states)

#             logits_k = F.interpolate(
#                 logits_k,
#                 size=(H, W),
#                 mode="bilinear",
#                 align_corners=False,
#             )

#             # 🔥 Normalize per scale (prevents dominance)
#             logits_k = logits_k / (logits_k.std(dim=(2,3), keepdim=True) + 1e-6)

#             logits_per_scale.append(logits_k)

#         stacked = torch.cat(logits_per_scale, dim=1)  # [B, K*C, H, W]

#         fused_logits, weights = self.fusion(stacked)

#         logits = fused_logits  # keep your 0.25 scaling if needed elsewhere
    
#         print(f"Weights: {weights.mean(dim=0).squeeze(-1).squeeze(-1)}")
#         print('---')
        
#         if return_weights:
#             return (
#                 SemanticSegmenterOutput(
#                     loss=None,
#                     logits=logits,
#                     hidden_states=None,
#                     attentions=None,
#                 ),
#                 weights
#             )
#         else:
#             return SemanticSegmenterOutput(
#                 loss=None,
#                 logits=logits,
#                 hidden_states=None,
#                 attentions=None,
#             )


# ==================================================================

    
def sliding_window_inference(model, image, window=512, stride=512, device="cuda"):
    """
    image: [1, 3, H, W]
    returns: stitched logits [1, C, H, W]
    """

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

            output[:, :, y:y1, x:x1] += logits
            counter[:, :, y:y1, x:x1] += 1

    return output / counter


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
        # for p in self.segformer.parameters():
        #     p.requires_grad = False

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

                # temp_final = np.moveaxis(torch.softmax(logits_resized[0,...], dim=1).cpu().numpy(), 0, 2)
                # temp_final = temp_final[...,1]
                # temp_final = (np.clip(temp_final, 0, 1) * 255).astype(np.uint8)
                # temp_final[temp_final > 0.5] = 1
                # Image.fromarray(temp_final).save(
                #     os.path.join(r"D:\GitHubProjects\Terranum_repo\LandSlides\segformerlandslides\data\test\subimages", f'image_{s}.tif')
                # )

                logits_per_scale.append(logits_resized)

        # quit()
        # 5️⃣ Stack
        stacked_logits = torch.cat(logits_per_scale, dim=1)

        # 6️⃣ Fuse
        fused_logits, weights = self.fusion(stacked_logits)

        # print(f"Weights: {weights.mean(dim=0).squeeze(-1).squeeze(-1)}")
        print(f"Weights: {weights.mean(dim=(0,2,3))}")
        print(psutil.virtual_memory().percent)
        print('---')

        if return_weights:
            return (
                SemanticSegmenterOutput(
                    loss=None,
                    logits=fused_logits,
                    hidden_states=None,
                    attentions=None,
                ),
                weights
            )
        else:
            return SemanticSegmenterOutput(
                loss=None,
                logits=fused_logits,
                hidden_states=None,
                attentions=None,
            )
    # def forward(self, pixel_values, labels=None, return_weights=False):

    #     B, H, W, C = pixel_values.shape
    #     device = pixel_values.device

    #     scale_descriptors = []
    #     logits_cache = []

    #     with torch.no_grad():

    #         for s in self.scales:

    #             Hs = int(H * s)
    #             Ws = int(W * s)

    #             img_scaled = F.interpolate(
    #                 pixel_values.permute(0,3,1,2),
    #                 size=(Hs, Ws),
    #                 mode="bilinear",
    #                 align_corners=False,
    #             )

    #             logits_scaled = sliding_window_inference(
    #                 self.segformer,
    #                 img_scaled,
    #                 window=512,
    #                 stride=512,
    #                 device=device
    #             )

    #             logits_resized = F.interpolate(
    #                 logits_scaled,
    #                 size=(H, W),
    #                 mode="bilinear",
    #                 align_corners=False
    #             )

    #             logits_resized = logits_resized / (
    #                 logits_resized.std(dim=(2,3), keepdim=True) + 1e-6
    #             )

    #             # Store small descriptor only
    #             desc = F.adaptive_avg_pool2d(logits_resized, 1)
    #             scale_descriptors.append(desc)

    #             # Temporarily store logits for fusion
    #             logits_cache.append(logits_resized)

    #     # Compute weights from descriptors (tiny tensor)
    #     scale_descriptors = torch.cat(scale_descriptors, dim=1)  # [B, K*C, 1, 1]
    #     weights = torch.sigmoid(self.fusion.net(scale_descriptors))
    #     weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

    #     # Fuse incrementally
    #     fused_logits = 0
    #     for k in range(len(self.scales)):
    #         fused_logits = fused_logits + weights[:, k:k+1] * logits_cache[k]

    #     del logits_cache  # free memory immediately

    #     print(f"Weights: {weights.mean(dim=0).squeeze()}")

    #     if return_weights:
    #         return (
    #             SemanticSegmenterOutput(
    #                 loss=None,
    #                 logits=fused_logits,
    #                 hidden_states=None,
    #                 attentions=None,
    #             ),
    #             weights
    #         )
    #     else:
    #         return SemanticSegmenterOutput(
    #             loss=None,
    #             logits=fused_logits,
    #             hidden_states=None,
    #             attentions=None,
    #         )


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
            # state_dict = torch.load(
            #     os.path.join(fusion_checkpoint, 'model.safetensors'), 
            #     map_location=map_location, 
            #     weights_only=False,
            #     )
            # model.load_state_dict(state_dict, strict=False)
            ckpt_path = os.path.join(fusion_checkpoint, "model.safetensors")

            state_dict = load_file(ckpt_path, device=device)
            fusion_state_dict = {
                k: v for k, v in state_dict.items()
                if k.startswith("fusion")
            }

# model.load_state_dict(fusion_state_dict, strict=False)
            model.load_state_dict(fusion_state_dict, strict=False)
        return model.to(device)


if __name__ == "__main__":
    print("WRONG SCRIPT PAL")

