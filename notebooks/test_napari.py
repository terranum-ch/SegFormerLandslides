import napari
import numpy as np
from PIL import Image
from skimage import data
import matplotlib.pyplot as plt
import napari
import os
import numpy as np
from PIL import Image

class FinetuningNapari:
    def __init__(self, src_dest):
        self.src_dest = src_dest

        self.viewer = napari.Viewer()

        self.layer_original = None
        self.layer_mask = None
        self.layer_proba = None

        self.load_first_image()

    def load_first_image(self):
        src_original = os.path.join(self.src_dest, "originals", os.listdir(os.path.join(self.src_dest, "originals"))[0])
        src_mask = src_original.replace("originals", "masks").replace(".tif", "_transparent.tif")
        src_proba = src_original.replace("originals", "probas").replace(".tif", "_transparent.tif")

        self.set_image(src_original, src_mask, src_proba)

    def set_image(self, src_original, src_mask=None, src_proba=None):
        img = np.array(Image.open(src_original))

        if self.layer_original is None:
            self.layer_original = self.viewer.add_image(
                img, name="original", contrast_limits=[0, 255]
            )
        else:
            self.layer_original.data = img

        if src_mask:
            mask = np.array(Image.open(src_mask))
            if self.layer_mask is None:
                self.layer_mask = self.viewer.add_image(
                    mask,
                    name="mask",
                    opacity=0.5,
                    colormap="red",
                )
            else:
                self.layer_mask.data = mask

        if src_proba:
            proba = np.array(Image.open(src_proba))
            if self.layer_proba is None:
                self.layer_proba = self.viewer.add_image(
                    proba,
                    name="proba",
                    opacity=0.5,
                    colormap="blue",
                )
            else:
                self.layer_proba.data = proba



def main():
    # from magicgui import magicgui
    # import os

    # viewer = napari.Viewer()

    # original_layer = viewer.add_image(
    #     np.zeros((10, 10, 3)),
    #     name="Original",
    #     rgb=True
    # )

    # mask_layer = viewer.add_image(
    #     np.zeros((10, 10)),
    #     name="Mask",
    #     opacity=0.6,
    #     visible=True
    # )

    # proba_layer = viewer.add_image(
    #     np.zeros((10, 10)),
    #     name="Probas",
    #     opacity=0.6,
    #     visible=True
    # )

    # src_originals = r"D:\GitHubProjects\Terranum_repo\LandSlides\segformerlandslides\data\test_finetuning\interesting_tiles\big_landslide\originals"
    # src_masks = r"D:\GitHubProjects\Terranum_repo\LandSlides\segformerlandslides\data\test_finetuning\interesting_tiles\big_landslide\masks"
    # src_probas = r"D:\GitHubProjects\Terranum_repo\LandSlides\segformerlandslides\data\test_finetuning\interesting_tiles\big_landslide\probas"
    # dict_images = {
    #             os.path.join(src_originals, x):{
    #                 'mask': os.path.join(src_masks, os.path.splitext(x)[0] + "_transparent.tif"),
    #                 'proba':os.path.join(src_probas, os.path.splitext(x)[0] + "_transparent.tif"),
    #             }
    #             for x in os.listdir(src_originals) }

    # @magicgui(
    #     image_name={
    #         "choices": [os.path.basename(x) for x in dict_images.keys()]
    #     },
    #     proba_opacity={"min": 0, "max": 1, "step": 0.05},
    #     th_pred={"choices": [0.05, 0.1, 0.3, 0.5, 0.7]},
    #     layout="vertical"
    # )

    # def controls(
    #     image_name: str,
    #     show_mask: bool,
    #     show_probas: bool,
    #     proba_opacity: float = 0.6,
    #     th_pred: float = 0.5,
    # ):
    #     # Resolve paths
    #     src_original = os.path.join(src_dest, "originals", image_name)
    #     src_mask = dict_images[src_original]["mask"]
    #     src_proba = dict_images[src_original]["proba"]

    #     # Update data (NO reallocation of layers)
    #     original_layer.data = np.array(Image.open(src_original))
    #     mask_layer.data = np.array(Image.open(src_mask))
    #     proba_layer.data = np.array(Image.open(src_proba))

    #     # Visibility
    #     mask_layer.visible = show_mask
    #     proba_layer.visible = show_probas

    #     # Opacity
    #     proba_layer.opacity = proba_opacity

    # return
    src_dest = r"D:\GitHubProjects\Terranum_repo\LandSlides\segformerlandslides\data\test_finetuning"
    viewer = FinetuningNapari(src_dest)
    napari.run()
    return
    # import napari

    viewer = napari.Viewer()
    viewer.add_image(data.astronaut(), rgb=True)
    # print(data.astronaut().shape)
    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.imshow(data.astronaut())
    # plt.show()
    return

    src_original = r"D:\GitHubProjects\Terranum_repo\LandSlides\segformerlandslides\data\test_finetuning\originals\tile_2580-1160_2023.tif"
    src_mask = r"D:\GitHubProjects\Terranum_repo\LandSlides\segformerlandslides\data\test_finetuning\masks\tile_2580-1160_2023_transparent.tif"
    src_proba = r"D:\GitHubProjects\Terranum_repo\LandSlides\segformerlandslides\data\test_finetuning\probas\tile_2580-1160_2023_transparent.tif"
    viewer = napari.Viewer()

    viewer.add_image(
        np.array(Image.open(src_original)),
        name="original",
    )

    # viewer.add_image(
    #     np.array(Image.open(src_mask)),
    #     name="mask",
    #     opacity=0.5,
    #     colormap="red",
    # )

    # viewer.add_image(
    #     np.array(Image.open(src_proba)),
    #     name="proba",
    #     opacity=0.5,
    #     colormap="blue",
    # )


if __name__ == "__main__":
    main()