from omegaconf import OmegaConf
from copy import deepcopy
import subprocess
import tempfile


def multi_training(trainings, args):
    """
    Runs multiple training experiments by modifying configuration parameters
    and launching training.py in a subprocess for each configuration.

    parameters:
        trainings (list[dict]) : list of parameter overrides for each run
        args (OmegaConf) : base configuration

    returns:
        None
    """

    for run in trainings:
        print("TRAINING WITH FOLLOWING SET OF PARAMETERS:")
        args_temp = deepcopy(args)
        for key, val in run.items():
            OmegaConf.update(args_temp, key, val)
            print("\t", key, ": ", OmegaConf.select(args_temp, key))
        print('---\n')

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode='w', encoding='utf-8') as f:
            OmegaConf.save(args_temp, f)
            cfg_path = f.name

        subprocess.run(
            [".venv/Scripts/python.exe", "training.py", f"--config={cfg_path}"]
        )
        
        print("\n----------------------------------------\n")


if __name__ == "__main__":
    trainings = [
        {
            'train.output_suffixe': 'playground_lbl_smoothing_0.001',
            'train.label_smoothing': 0.001,
        },
        {
            'train.output_suffixe': 'playground_lbl_smoothing_0.01',
        },
        {
            'train.output_suffixe': 'playground_lbl_smoothing_0.1',
            'train.label_smoothing': 0.1,
        },
        {
            'train.output_suffixe': 'landslide_lbl_smoothing_0.01',
            'dataset.segmenter.dataset_dir': "data/dataset_segmentation_landslide",
            'train.loss_weights': 'auto',
        },
        {
            'train.output_suffixe': 'playground_fusion_false_pos_lbl_smoothing_0.01',
            'train.is_trained': 'fusion',
            'train.num_epochs': 20,
            'train.batch_size': 16,
            'train.num_workers': 8,
        },

    ]

    args = OmegaConf.load('./config/training.yaml')

    multi_training(trainings, args)

