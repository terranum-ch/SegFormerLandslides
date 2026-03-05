import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_loss_pa(history, saving_loc, do_show=False, do_save=True):
    # Lists to fill
    train_loss = []
    train_pa = []
    val_loss = []
    val_pa = []

    # Parse log history
    for entry in history:
        if "train_loss" in entry:
            train_loss.append(entry["train_loss"])

        if "train_pa" in entry:
            train_pa.append(entry["train_pa"])

        if "eval_pa" in entry:
            val_pa.append(entry["eval_pa"])

        if "eval_loss" in entry:
            val_loss.append(entry["eval_loss"])

    # -----------------------
    # Plot 1: Training Loss
    # -----------------------
    plt.figure(figsize=(14,5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train')

    plt.plot(val_loss, label='val')
    plt.title("Loss")
    plt.legend()
    plt.xlabel("Epoch [-]")
    plt.ylabel("Loss [-]")

    # -----------------------
    # Plot 2: Validation Pixel accuracy
    # -----------------------
    plt.subplot(1, 2, 2)
    plt.plot(train_pa, label='train')
    plt.plot(val_pa, label='val')
    plt.title("Pixel Accuracy")
    plt.legend()
    plt.xlabel("Epoch [-]")
    plt.ylabel("PA [-]")
    
    plt.tight_layout()
    if do_save:
        plt.savefig(saving_loc)
        plt.savefig(os.path.splitext(saving_loc)[0] + '.eps', format='eps')
    
    if do_show:
        plt.show()
    plt.clf()


def show_mean_iou_dice(history, saving_loc, do_show=False, do_save=True):
    # Lists to fill
    train_mdice = []
    train_miou = []
    val_mdice = []
    val_miou = []

    # Parse log history
    for entry in history:
        if "train_loss" in entry:
            train_miou.append(entry["train_mean_iou"])

        if "train_pa" in entry:
            train_mdice.append(entry["train_mean_dice"])

        if "eval_mean_iou" in entry:
            val_miou.append(entry["eval_mean_iou"])

        if "eval_mean_dice" in entry:
            val_mdice.append(entry["eval_mean_dice"])

    # -----------------------
    # Plot 1: Training Loss
    # -----------------------
    plt.figure(figsize=(14,5))
    plt.subplot(1, 2, 1)
    plt.plot(train_miou, label='train')

    plt.plot(val_miou, label='val')
    plt.title("Mean IoU")
    plt.legend()
    plt.xlabel("Epoch [-]")
    plt.ylabel("IoU [-]")

    # -----------------------
    # Plot 2: Validation Pixel accuracy
    # -----------------------
    plt.subplot(1, 2, 2)
    plt.plot(train_mdice, label='train')
    plt.plot(val_mdice, label='val')
    plt.title("Mean Dice")
    plt.legend()
    plt.xlabel("Epoch [-]")
    plt.ylabel("Dice [-]")
    
    plt.tight_layout()

    if do_save:
        plt.savefig(saving_loc)
        plt.savefig(os.path.splitext(saving_loc)[0] + '.eps', format='eps')
    
    if do_show:
        plt.show()
    plt.clf()
    

def show_iou_per_class(history, saving_loc, do_show=False, do_save=True):
    # Per-class IoU (dynamic)
    val_per_class_iou = {}  # {class_id: [iou_epoch1, iou_epoch2, ...]}
    train_per_class_iou = {}  # {class_id: [iou_epoch1, iou_epoch2, ...]}

    # Parse log history
    for entry in history:
        # Per-class IoU
        for key, value in entry.items():
            if key.startswith("eval_iou_class_"):
                cls = int(key.split("_")[-1])
                val_per_class_iou.setdefault(cls, [])
                val_per_class_iou[cls].append(value)
            elif key.startswith("train_iou_class_"):
                cls = int(key.split("_")[-1])
                train_per_class_iou.setdefault(cls, [])
                train_per_class_iou[cls].append(value)
    
    # Show results
    plt.figure(figsize=(14,5))
    plt.subplot(1, 2, 1)
    mapping_class_names = {key: val for key,val in zip(train_per_class_iou.keys(), ['Background', 'Landslide'])}
    for cls, values in train_per_class_iou.items():
        plt.plot(values, label=mapping_class_names[cls])
    plt.title("Training set")
    plt.xlabel("Epoch [-]")
    plt.ylabel("IoU [-]")
    plt.legend()

    plt.subplot(1, 2, 2)
    mapping_class_names = {key: val for key,val in zip(val_per_class_iou.keys(), ['Background', 'Landslide'])}
    for cls, values in val_per_class_iou.items():
        plt.plot(values, label=mapping_class_names[cls])
    plt.title("Validation set")

    plt.xlabel("Epoch [-]")
    plt.ylabel("IoU [-]")
    plt.legend()

    plt.suptitle("Mean IoU per class")
    plt.tight_layout()
    
    if do_save:
        plt.savefig(saving_loc)
        plt.savefig(os.path.splitext(saving_loc)[0] + '.eps', format='eps')
    
    if do_show:
        plt.show()
    plt.clf()


def show_confusion_matrix(saving_loc, conf_mat, class_labels, title="Confusion Matrix", do_show=False, do_save=True):
    """
        plots the confusion matrix as and image
        :param saving_loc : location of saved image
        :param y_true: list of the GT label of the models
        :param y_pred: List of the predicted label of the models
        :param class_labels: List of strings containing the label tags
        :param epoch: number of the epoch of training which provided the results
        :param do_save: saves the image
        :param do_show: shows the image
        :return: None (just plots)
    """
    
    df_conf_mat = pd.DataFrame(conf_mat, index=class_labels, columns=class_labels)

    fig = plt.figure()
    sns.heatmap(df_conf_mat, annot=True, cmap=sns.color_palette("Blues", as_cmap=True))
    ax = plt.gca()
    ax.set_title(title)

    plt.tight_layout()
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    fig.tight_layout()

    if do_save:
        plt.savefig(saving_loc)
        plt.savefig(os.path.splitext(saving_loc)[0] + '.eps', format='eps')

    if do_show:
        plt.show()

    plt.clf()


if __name__ == "__main__":
    # Save best confidence matrix
    src_best_cm = r"D:\GitHubProjects\Terranum_repo\LandSlides\segformerlandslides\results\training\20251217_102843_50_epochs_Bern_v2_da\logs\confmats\values\confusion_matrix_ep_49.csv"
    IMG_DIR = r"D:\GitHubProjects\Terranum_repo\LandSlides\segformerlandslides\results\training\20251217_102843_50_epochs_Bern_v2_da\images"
    if os.path.exists(src_best_cm):
        conf_mat = pd.read_csv(src_best_cm, sep=';', index_col=0).values
        sum_for_recall = np.sum(conf_mat, axis=1).reshape(-1, 1)
        sum_for_precision = np.sum(conf_mat, axis=0).reshape(1, -1)
        show_confusion_matrix(os.path.join(IMG_DIR, 'confusion_matrix.png'), conf_mat, ['Background', 'Landslide'])
        show_confusion_matrix(os.path.join(IMG_DIR, 'confusion_matrix_recall.png'), conf_mat / sum_for_recall, ['Background', 'Landslide'], "Confusion Matrix - Producer accuracy")
        show_confusion_matrix(os.path.join(IMG_DIR, 'confusion_matrix_precision.png'), conf_mat / sum_for_precision, ['Background', 'Landslide'], "Confusion Matrix - User accuracy")
    else:
        print("CONFMAT NOT CREATED FOR BEST EPOCH")
        print("following does not exist:")
        print(src_best_cm)
    quit()
    # src = r"D:\GitHubProjects\Terranum_repo\LandSlides\segformerlandslides\results\training\20251215_080906_50_epochs_Bern_from_scratch\last_checkpoint"

    # # Path to trainer_state.json
    # state_file = os.path.join(src, "trainer_state.json")
    # # print(os.path.basename(src))
    # # quit()
    # IMG_DIR = os.path.join(os.path.dirname(src), 'images')
    # os.makedirs(IMG_DIR, exist_ok=True)
    # with open(state_file, "r") as f:
    #     state = json.load(f)
    # history = state["log_history"]

    # show_loss_pa(history,os.path.join(IMG_DIR, 'loss_pa.png'), False, True)
    # show_mean_iou_dice(history,os.path.join(IMG_DIR, 'mean_iou_dice.png'), False, True)
    # show_iou_per_class(history,os.path.join(IMG_DIR, 'iou_per_class.png'), False, True)

    # last_checkpoint_path = trainer.state.best_model_checkpoint or trainer.state.last_model_checkpoint
    last_checkpoint_path = r"D:\GitHubProjects\Terranum_repo\LandSlides\segformerlandslides\results\training\20251215_162829_50_epochs_Bern_from_pretrained\last_checkpoint"
    IMG_DIR = r"D:\GitHubProjects\Terranum_repo\LandSlides\segformerlandslides\results\training\20251215_162829_50_epochs_Bern_from_pretrained\images"
    state_file = os.path.join(last_checkpoint_path, "trainer_state.json")
    
    with open(state_file, "r") as f:
        state = json.load(f)
    history = state["log_history"]

    do_show = True
    do_save = False
    show_loss_pa(history,os.path.join(IMG_DIR, 'loss_pa.png'), do_show, do_save)
    show_mean_iou_dice(history,os.path.join(IMG_DIR, 'mean_iou_dice.png'), do_show, do_save)
    show_iou_per_class(history,os.path.join(IMG_DIR, 'iou_per_class.png'), do_show, do_save)
