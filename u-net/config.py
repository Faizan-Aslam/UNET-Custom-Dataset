import torch 
import os

''' this file consists of model's parametes configurations and initial settings'''

#root directory of the data set
DATASET_PATH = os.path.join('dataset', 'train' )

# root directory to training images and masks.
IMAGES_PATH = os.path.join(DATASET_PATH, 'images')
MASKS_PATH = os.path.join(DATASET_PATH, 'masks')


# default test split ratio

TEST_SPLIT = 0.15

# device to be used for training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# memory pinning during data loading:

MEMORY_PIN = True if DEVICE == "cuda" else False

# Defining model parameters i.e. channels in the input, input classes and number of levels in the model.

NUM_OF_CHANNELS = 1
NUM_OF_CLASSES = 1 
NUM_OF_LEVELS = 3

# intializing parameters

LR_INIT = 0.001
EPOCHS =   40
BATCH_SIZE = 64

# define the input image dimensions

INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128

# threshold for confident predictions

THRESHOLD =0.5

#directory to save output results

BASE_OUTPUT = 'output'


# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_pyt.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])