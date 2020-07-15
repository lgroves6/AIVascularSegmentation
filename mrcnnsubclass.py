from mrcnnconfig import *
from mrcnnutils import *
import numpy as np
from keras.utils import to_categorical

class VesselConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "vessels"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5




class VesselDataset(Dataset):

    start_idx = 0

    def __init__(self, X_path, Y_path):
        self.X = np.load(X_path)
        self.Y = np.load(Y_path)
        Dataset.__init__(self)

    def load_images(self, start_idx, end_idx):
        self.start_idx = start_idx
        self.add_class("vessels", 1, "carotid")
        self.add_class("vessels", 2, "jugular")
        for i in range(start_idx, end_idx):
            self.add_image("vessels", image_id=i, path=None)

    def load_image(self, image_id):
        print(image_id + self.start_idx)
        image = (self.X[image_id + self.start_idx]*255).astype(np.int32)
        return image
        image = np.tile(np.expand_dims(image, axis=2), (1,1,3)) # Convert grayscale to RBG
        return np.squeeze(image, axis=2)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "vessels":
            return info["vessels"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        mask = self.Y[image_id + self.start_idx].astype(np.int32)
        mask = to_categorical(mask).astype(np.int32)
        mask = np.delete(mask, 0, 2) # Delete background class channel
        return mask.astype(np.bool), np.array([1,2])
