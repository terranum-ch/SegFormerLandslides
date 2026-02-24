import os
import numpy as np
import torch
from transformers import TrainerCallback


class MetricsCallback(TrainerCallback):
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
