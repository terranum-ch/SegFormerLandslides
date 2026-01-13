import torch
import numpy as np
from math import prod
import time
from tqdm import tqdm
from transformers import Trainer
from torch.utils.data import default_collate
from torch.utils.data import Subset
import torch.nn.functional as F

from .metrics import compute_metrics, compute_cm_from_dict


# for evaluation_loop overwrite
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import EvalLoopContainer, find_batch_size, IterableDatasetShard
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput, denumpify_detensorize, has_length
from transformers.utils import logging
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

        # compute weights for loss
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
        print(f"Weights:\n\tBackground: {self.class_weights[0]}\n\tLandslide: {self.class_weights[1]}")


    @staticmethod
    def logits_to_preds(logits, height=512, width=512):
        # Resize predictions to match label size
        logits = torch.nn.functional.interpolate(
            torch.tensor(logits),
            size=(logits.shape[-2]*4, logits.shape[-1]*4),   # (H_lbl, W_lbl)
            mode="bilinear",
            align_corners=False
        )

        # Final predicted class mask
        return logits.argmax(dim=1).cpu().numpy()  # (batch_size, H_lbl, W_lbl)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        # Standard HF forward pass
        #   remove the filename for training
        inputs = {k: v for k, v in inputs.items() if k not in ["filename"]}
        outputs = model(**inputs)
        loss = outputs.loss

        # Compute logits without affecting backward
        with torch.no_grad():
            logits = outputs.logits
            labels = inputs["labels"]

        logits = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy().astype(np.uint8)
        preds = self.logits_to_preds(logits).astype(np.uint8)

        # Save them for end-of-epoch metrics
        dict_for_metrics = {'predictions': logits, "label_ids": labels}
        # dict_conf_mat = {x: (preds[x,...], labels[x,...]) for x in range(preds.shape[0])}
        # self.confmat += compute_cm_from_dict(dict_conf_mat)

        metrics = compute_metrics(dict_for_metrics)
        self.training_metrics.append(metrics)
        self.training_losses.append(loss.cpu().detach().numpy())

        # Standard HF backward
        self.accelerator.backward(loss)
        return loss.detach()
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only = None, ignore_keys = None, metric_key_prefix = "eval"):
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8" and not self.args.torch_compile)
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        if hasattr(model, "eval") and callable(model.eval):
            model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # -----------------------------
            # ------- custom lines --------
            # -----------------------------

            # Prediction step
            #   remove filename from the inputs before giving it to the model
            # filename_in = 'filename' in inputs.keys()
            # if filename_in: 

            filenames = inputs['filename']
            inputs = {k:v for k,v in inputs.items() if k != 'filename'}
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            #   split batches to add each sample
            # if filename_in:
            preds = self.logits_to_preds(logits)
            for sample_id in range(preds.shape[0]):
                self.eval_preds[filenames[sample_id]] = preds[sample_id, ...]
            labels_cpu = labels.cpu().detach().clone()
            dict_conf_mat = {x: (preds[x,...], labels_cpu[x,...]) for x in range(preds.shape[0])}
            self.confmat += compute_cm_from_dict(dict_conf_mat)
                
            # -----------------------------
            # -----------------------------
            
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            # if is_torch_xla_available():
            #     xm.mark_step()

            # Update containers
            if losses is not None:
                losses = self.gather_function(losses.repeat(batch_size))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function(inputs_decode)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function(logits)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function(labels)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, **batch_kwargs),
                        compute_result=is_last_step,
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels, **eval_set_kwargs)
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits  # (B, C, H, W)

        # Resize logits to match labels
        logits = torch.nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        ce_loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            ignore_index=255  # very important for segmentation
        )
        f_loss = focal_loss(logits, labels)

        d_loss = dice_loss(logits, labels)
        # print("Weighted CE loss: ", ce_loss)
        # print("Dice loss: ", d_loss)
        # print("Focal loss: ", f_loss)
        # loss = ce_loss + 0.5 * d_loss
        loss = f_loss + 0.5 * d_loss

        return (loss, outputs) if return_outputs else loss
    

if __name__ == "__main__":
    print("WRONG SCRIPT PAL")

