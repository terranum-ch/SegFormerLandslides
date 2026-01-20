from omegaconf import OmegaConf
from training import training
from copy import deepcopy


def multi_training(trainings, args):
    for run in trainings:
        print("TRAINING WITH FOLLOWING SET OF PARAMETERS:")
        args_temp = deepcopy(args)
        for key, val in run.items():
            OmegaConf.update(args_temp, key, val)
            print("\t", key, ": ", OmegaConf.select(args_temp, key))

        training(args_temp)
        
        print("\n----------------------------------------\n")


if __name__ == "__main__":
    trainings = [
        {
            'dataset.dataset_dir': 'data/dataset_Bern_v2_and_false_pos_0.5',
            'train.output_suffixe': "Bern_v2_focal_dice_losses_with_1800_false_pos_from_scratch",
            'train.resume_from_existing': True,
        },
        {
            'dataset.dataset_dir': 'data/dataset_Bern_v2_and_false_pos_0.25',
            'train.output_suffixe': "Bern_v2_focal_dice_losses_with_900_false_pos_from_scratch",
        },
        {
            'dataset.dataset_dir': 'data/dataset_Bern_v2_and_false_pos_0.25',
            'train.output_suffixe': "Bern_v2_focal_dice_losses_with_900_false_pos_from_scratch_da_scaling",
            'train.do_da_scaling': True,
        },
    ]

    conf_train = OmegaConf.load('./config/training.yaml')
    conf_dataset = OmegaConf.load('./config/dataset.yaml')

    args= OmegaConf.merge({"train":conf_train, "dataset":conf_dataset})

    multi_training(trainings, args)

