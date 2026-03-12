from omegaconf import OmegaConf
import argparse
from training import training

def train_both(conf):
    """
    Sequentially train the segmentation model and the fusion model using the same configuration object.
    Parameters: conf (OmegaConf) – configuration containing training, fusion, and dataset parameters.
    Returns: None – trains both models and saves their checkpoints to disk.
    """
    
    # training segmenter
    print("\n", "="*30)
    print("=== TRAINING SEGMENTER =======")
    print("\n", "="*30)

    conf.train.num_epochs = conf.fusion.segmenter.num_epochs
    conf.train.batch_size = conf.fusion.segmenter.batch_size
    conf.train.num_workers = conf.fusion.segmenter.num_workers
    conf.train.is_trained = 'segmenter'
    conf.dataset.segmenter.dataset_dir = conf.fusion.segmenter.dataset
    segmenter_model_dir = training(conf)

    # training fusion modulus
    print("\n", "="*30)
    print("=== TRAINING FUSION MOD ======")
    print("\n", "="*30)

    conf.train.num_epochs = conf.fusion.fusion.num_epochs
    conf.train.batch_size = conf.fusion.fusion.batch_size
    conf.train.num_workers = conf.fusion.fusion.num_workers
    conf.train.is_trained = 'fusion'
    conf.dataset.fusion.dataset_dir = conf.fusion.fusion.dataset
    conf.train.pretrained_model = segmenter_model_dir
    training(conf)


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
        conf_fusion = OmegaConf.load('./config/training_fusion.yaml')
        conf_train = OmegaConf.load('./config/training.yaml')

        args= OmegaConf.merge({"fusion": conf_fusion, "train":conf_train.train, "dataset":conf_train.dataset})

        # args = OmegaConf.load('./config/training_fusion.yaml')

    train_both(args)
