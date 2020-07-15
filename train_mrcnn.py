from mrcnnconfig import *
from mrcnnmodel import *
from mrcnnutils import *
from mrcnnvisualize import *
import numpy as np
import imgaug.augmenters as aug
import argparse
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
    NUM_CLASSES = 3  # background + 2 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor size in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1200

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 8

    # Only accept detections if 95% confidence
    DETECTION_MIN_CONFIDENCE = 0.95

    RPN_NMS_THRESHOLD = 0.7
	
    LEARNING_RATE = 0.001
	
    BACKBONE = "resnet50"
    IMAGE_RESIZE_MODE = "square"
    
    IMAGE_CHANNEL_COUNT = 1

    MEAN_PIXEL = np.array([70.0])
	
    MAX_GT_INSTANCES = 100
    
    WEIGHT_DECAY = 0.0001
    
    LOSS_WEIGHTS = {"rpn_class_loss": 1., "rpn_bbox_loss": 1., "mrcnn_class_loss": 1., "mrcnn_bbox_loss": 1.,"mrcnn_mask_loss": 1.}
	
    PATIENCE = 15




class VesselDataset(Dataset):

    start_idx = 0

    def load_images(self, start_idx, end_idx):
        self.start_idx = start_idx
        self.add_class("vessels", 1, "carotid")
        self.add_class("vessels", 2, "jugular")
        for i in range(start_idx, end_idx):
            self.add_image("vessels", image_id=i, path=None)

    def load_image(self, image_id):
        print(image_id + self.start_idx)
        image = (X[image_id + self.start_idx]*255).astype(np.int32)
        return image
		#image = np.tile(np.expand_dims(image, axis=2), (1,1,3)) # Convert grayscale to RBG
        #return np.squeeze(image, axis=2)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "vessels":
            return info["vessels"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        mask = Y[image_id + self.start_idx].astype(np.int32)
        mask = to_categorical(mask).astype(np.int32)
        mask = np.delete(mask, 0, 2) # Delete background class channel
        return mask.astype(np.bool), np.array([1,2])

print("Before args parsed")
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', '--gpu_count', type=int)
parser.add_argument('-ipg', '--imgs_per_gpu', type=int)
parser.add_argument('-lr', '--learning_rate', type=float)
parser.add_argument('-p', '--patience', type=int)
parser.add_argument('-wd', '--weight_decay', type=float)
parser.add_argument('-spe', '--steps_per_epoch', type=int)
args = parser.parse_args()
print("Args parsed")

# Set up model config. Change defaults if supplied as command args.
config = VesselConfig()
if args.gpu_count:
	config.GPU_COUNT = args.gpu_count
if args.imgs_per_gpu:
	config.GPU_COUNT = args.imgs_per_gpu
if args.learning_rate:
	config.GPU_COUNT = args.learning_rate
if args.patience:
	config.GPU_COUNT = args.patience
if args.weight_decay:
	config.GPU_COUNT = args.weight_decay
if args.steps_per_epoch:
	config.GPU_COUNT = args.steps_per_epoch
config.display()

data_path = '/scratch/bvanberl/data/'
X = np.load(data_path + 'X_crossval.npy')
Y = np.load(data_path + 'Y_crossval.npy')
print("ORIGINAL SHAPES:",X.shape,Y.shape)
X = np.squeeze(X, axis=1)
Y = np.squeeze(Y, axis=1)
X = np.expand_dims(X, axis=3)
Y = np.expand_dims(Y, axis=3)
'''
np.random.seed(0)
permut = np.random.permutation(X.shape[0])
print("RANDOM PERMUTATION: ", permut)
X = X[permut]
Y = Y[permut]
print("X SIZE: ", X.shape)
'''
########################################################
# PARTITION FOR CROSS-VALIDATION
'''
print("******SET A******")
permut = list(range(0,78)) + list(range(611,1865)) + list(range(1955,2215)) + list(range(2314,2439)) # train
permut += list(range(1865,1955)) + list(range(2215,2295)) # val
permut += list(range(78,611)) + list(range(2295,2314)) # test
train_start = 0
train_end = 1717
val_start = 1717
val_end = 1887
test_start = 1887
test_end = 2439

print("******SET B******")
permut = list(range(0,1012)) + list(range(1432,1714)) + list(range(1792,2021)) + list(range(2108,2314)) + list(range(2369,2439)) # train
permut += list(range(1714,1792)) + list(range(2021,2108)) # val
permut += list(range(1012,1432)) + list(range(2314,2369)) # test
train_start = 0
train_end = 1799
val_start = 1799
val_end = 1964
test_start = 1964
test_end = 2439

print("******SET C******")
permut = list(range(78,1432)) + list(range(1714,2108)) + list(range(2215,2369)) # train
permut += list(range(0,78)) + list(range(2108,2215)) # val
permut += list(range(1432,1714)) + list(range(2369,2439)) # test
train_start = 0
train_end = 1902
val_start = 1902
val_end = 2087
test_start = 2087
test_end = 2439
'''
print("******SET D******")
permut = list(range(0,611)) + list(range(1012,1792)) + list(range(1865,1955)) + list(range(2021,2215)) + list(range(2295,2439)) # train
permut += list(range(1792,1865)) + list(range(1955,2021)) # val
permut += list(range(611,1012)) + list(range(2215,2295)) # test
train_start = 0
train_end = 1819
val_start = 1819
val_end = 1958
test_start = 1958
test_end = 2439

print("STATS", len(permut), X.shape[0])
assert len(permut) == X.shape[0]
permut = np.array(permut)
print(permut)
X = X[permut]
Y = Y[permut]
X /= 255.0

#######################################################

X_test = X[test_start:test_end]
Y_test = Y[test_start:test_end]
X_val = X[val_start:val_end]
Y_val = Y[val_start:val_end]
X_train = X[train_start:train_end]
Y_train = Y[train_start:train_end]

np.save("./datasets/X_test", X_test)
np.save("./datasets/Y_test", Y_test)

# Training dataset 
dataset_train = VesselDataset()
dataset_train.load_images(train_start, train_end)
dataset_train.prepare()

# Test dataset 
dataset_test = VesselDataset()
dataset_test.load_images(test_start, test_end)
dataset_test.prepare()


# Validation dataset (User 1)
dataset_val = VesselDataset()
dataset_val.load_images(val_start, val_end)
dataset_val.prepare()

# Perform image augmentation on the training set
augmentation = aug.Sometimes(1.0, [
                    aug.Sometimes(1.0, aug.Affine(
                               scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                               rotate=(-15, 15)))
                ])

# Create model in training mode
model = MaskRCNN(mode="training", config=config, model_dir='/home/bvanberl/vessels/models/setD/')
print(model.keras_model.summary())

# Train the model
model.train(dataset_train, dataset_val, dataset_test, learning_rate=config.LEARNING_RATE,
            epochs=100, layers='all', augmentation=augmentation)
model.keras_model.save('/home/bvanberl/vessels/setD_mask_rcnn.h5')