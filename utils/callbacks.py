import os
import shutil
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import TrainerCallback
from .visualization import show_confusion_matrix


class MetricsCallback(TrainerCallback):
    def __init__(self, trainer, cf_dir):
        self.trainer = trainer   # keep a reference
        self.cf_dir = cf_dir
        self.cf_dir_img = os.path.join(cf_dir, "images")
        self.cf_dir_val = os.path.join(cf_dir, "values")

        os.makedirs(cf_dir, exist_ok=True)
        os.makedirs(self.cf_dir_img, exist_ok=True)
        os.makedirs(self.cf_dir_val, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        # Nothing to do if lists are empty
        if not self.trainer.training_metrics:
            return

        # Concatenate stored batch outputs from the epoch
        new_metrics = {metric: [float(lst[metric]) for lst in self.trainer.training_metrics] for metric in self.trainer.training_metrics[0].keys()}
        
        training_log = {"train_loss": float(np.mean(self.trainer.training_losses))}
        
        # Log mean value of batches for epoch
        for key, val in new_metrics.items():
            training_log[f"train_{key}"] = float(np.mean(val))

        # Create confusion matrix
        show_confusion_matrix(
            saving_loc=os.path.join(self.cf_dir_img, f"confusion_matrix_ep_{int(state.epoch - 1)}.jpg"),
            conf_mat=self.trainer.confmat,
            class_labels=['Background', 'Landslide'],
            )
        pd.DataFrame(self.trainer.confmat, index=[0,1], columns=[0,1]).to_csv(os.path.join(self.cf_dir_val, f"confusion_matrix_ep_{int(state.epoch - 1)}.csv"), sep=';')

        # Clear buffers for next epoch
        self.trainer.training_metrics.clear()
        self.trainer.confmat = np.zeros((2,2), dtype=np.uint64)

        # Log training metric (it will appear in trainer_state.json)
        self.trainer.log(training_log)


class SaveBestPredictionsCallback(TrainerCallback):
    def __init__(self, trainer, save_dir, dataset_dir):
        self.trainer = trainer
        self.save_dir = save_dir
        self.dataset_dir = dataset_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_metric = None   # track manually
        self.first_time_saving = True

    @staticmethod
    def save_tif_from_array(src, filename, arr):
        src_preds = os.path.join(src, "preds")
        src_bin = os.path.join(src, "bin")
        os.makedirs(src_preds, exist_ok=True)
        os.makedirs(src_bin, exist_ok=True)

        src_image_file = os.path.join(src_preds, filename)
        src_mask_file = os.path.join(src_bin, filename)

        # Saving binary mask
        pil_mask = Image.fromarray(arr.astype(np.uint8))
        pil_mask.save(src_mask_file)

        # Saving rgb version of mask
        rgb_mask = np.zeros((arr.shape[0], arr.shape[1], 3))
        rgb_mask[arr == 1] = 255
        pil_rgb_mask = Image.fromarray(rgb_mask.astype(np.uint8))
        pil_rgb_mask.save(src_image_file)

    def on_evaluate(self, args, state, control, **kwargs):
        # Save evaluation predictions if best epoch and then clear it
        if state.best_metric == None or state.stateful_callbacks['TrainerControl']['args']['should_save']:
            for filename, preds in self.trainer.eval_preds.items():
                self.save_tif_from_array(self.save_dir, filename, preds)

                # saving also images and labels if first time
                if self.first_time_saving:
                    src_images = os.path.join(self.save_dir, 'images')
                    src_labels = os.path.join(self.save_dir, 'labels')

                    os.makedirs(src_images, exist_ok=True)
                    os.makedirs(src_labels, exist_ok=True)

                    shutil.copyfile(
                        os.path.join(self.dataset_dir, 'images', filename), 
                        os.path.join(src_images, filename),
                        )
                    shutil.copyfile(
                        os.path.join(self.dataset_dir, 'labels', filename), 
                        os.path.join(src_labels, filename),
                        )
                    
        self.first_time_saving = False
        self.trainer.eval_preds.clear()


class SavesCurrentStateCallback(TrainerCallback):
    def __init__(self, last_checkpoint_dir, trainer):
        self.trainer = trainer
        self.checkpoint_dir = last_checkpoint_dir

    def on_evaluate(self, args, state, control, **kwargs):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.trainer.save_model(self.checkpoint_dir, _internal_call=True)
        self.trainer.state.save_to_json(os.path.join(self.checkpoint_dir, 'trainer_state.json'))
        self.trainer._save_rng_state(self.checkpoint_dir)
        torch.save(self.trainer.optimizer.state_dict(), os.path.join(self.checkpoint_dir, "optimizer.pt"))
        torch.save(self.trainer.accelerator.scaler.state_dict(), os.path.join(self.checkpoint_dir, "scaler.pt"))
        torch.save(self.trainer.lr_scheduler.state_dict(), os.path.join(self.checkpoint_dir, "scheduler.pt"))


class TrainMetricsCallback(TrainerCallback):
    """
    Collect logits and labels during training without interfering with
    HuggingFace training_step, AMP, DDP, or compute_loss.

    Computes metrics on-the-fly for monitoring only.
    """

    def __init__(self, trainer, compute_metrics_fn):
        self.trainer = trainer
        self.compute_metrics = compute_metrics_fn
        self.reset_buffers()

    def reset_buffers(self):
        self.preds = []
        self.labels = []

    def on_step_end(self, args, state, control, **kwargs):
        # trainer = kwargs["trainer"]

        # HF stores last model outputs here
        if not hasattr(self.trainer, "_last_outputs"):
            return

        outputs = self.trainer._last_outputs
        inputs = self.trainer._last_inputs

        if outputs is None or inputs is None:
            return

        logits = outputs.logits.detach().cpu()
        labels = inputs["labels"].detach().cpu()

        self.preds.append(logits)
        self.labels.append(labels)

    def on_epoch_end(self, args, state, control, **kwargs):
        if len(self.preds) == 0:
            return

        preds = torch.cat(self.preds).numpy()
        labels = torch.cat(self.labels).numpy()

        metrics = self.compute_metrics({
            "predictions": preds,
            "label_ids": labels
        })

        print(f"\n🟢 Training metrics at epoch {state.epoch:.1f}: {metrics}\n")

        self.reset_buffers()
