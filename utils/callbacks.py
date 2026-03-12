import os
import numpy as np
import torch
from transformers import TrainerCallback


class MetricsCallback(TrainerCallback):
    """
    HuggingFace Trainer callback used to aggregate and log training metrics at the end of each epoch.
    Parameters: 
        trainer (Trainer) - reference to the trainer instance storing batch-level training metrics; 
        cf_dir (str) - directory where confusion matrices or related evaluation artifacts may be stored.
    Returns: 
        TrainerCallback - callback that computes epoch-level training metrics and logs them to the trainer state.
    """

    def __init__(self, trainer, cf_dir):
        self.trainer = trainer   # keep a reference

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

        # Clear buffers for next epoch
        self.trainer.training_metrics.clear()

        # Log training metric (it will appear in trainer_state.json)
        self.trainer.log(training_log)


class SavesCurrentStateCallback(TrainerCallback):
    """
    HuggingFace Trainer callback used to save the latest training state after each epoch.
    Parameters: 
        last_checkpoint_dir (str) - directory where the latest checkpoint state will be saved; 
        trainer (Trainer) - reference to the trainer instance whose state will be stored.
    Returns: 
        TrainerCallback - callback that saves the model, optimizer, scheduler, scaler, and trainer state.
    """

    def __init__(self, last_checkpoint_dir, trainer):
        self.trainer = trainer
        self.checkpoint_dir = last_checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, **kwargs):
        self.trainer.state.save_to_json(os.path.join(self.checkpoint_dir, 'trainer_state.json'))

    def on_epoch_end(self, args, state, control, **kwargs):
        self.trainer.save_model(self.checkpoint_dir, _internal_call=True)
        self.trainer._save_rng_state(self.checkpoint_dir)
        torch.save(self.trainer.optimizer.state_dict(), os.path.join(self.checkpoint_dir, "optimizer.pt"))
        torch.save(self.trainer.accelerator.scaler.state_dict(), os.path.join(self.checkpoint_dir, "scaler.pt"))
        torch.save(self.trainer.lr_scheduler.state_dict(), os.path.join(self.checkpoint_dir, "scheduler.pt"))

