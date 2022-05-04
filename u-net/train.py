import os
from time import time

from cv2 import sort, split
from sklearn import datasets
from zmq import device
from .dataset import SegDataset
from .model import Unet
import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os

# load the image and ask filepaths in sorted manner
imgPaths = sorted(list(paths.list_images(config.IMAGES_PATH)))
maskPaths =  sorted(list(paths.list_images(config.MASKS_PATH)))

# partitioning the data into test and training split
split = train_test_split(imgPaths, maskPaths, test_size=config.TEST_SPLIT, random_state=42)

# unpacking the datasets
(trainImgs, testImgs) = split[:2]
(trainMasks, testMasks) = split[2:]

# write the test images paths to disk so we can use later for testing validating our model

print("saving test paths")

with open(config.TEST_PATHS, "w") as f:
    f.write("\n".join(testImgs))

# define transformations

transforms = transforms.Compose(transforms.ToPILImage(), 
transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH),
transforms.ToTensor
))


# create the train and test datasets.

train = SegDataset(imagePaths= trainImgs, maskPaths= trainMasks, transforms=transforms)
test = SegDataset(imagePaths=testImgs, maskPaths= testMasks, transforms= transforms)

print(f"[INFO] found {len(train)} examples in the training set...")
print(f"[INFO] found {len(test)} examples in the test set...")



# creating the test and train data loaders(creating batches and loading from the disk)

trainLoader = DataLoader(train, shuffle=True, batch_size= config.BATCH_SIZE, pin_memory= config.MEMORY_PIN, num_workers=os.cpu_count())


testLoader = DataLoader(test, shuffle=True, batch_size= config.BATCH_SIZE, pin_memory= config.MEMORY_PIN, num_workers= os.cpu_count())


# initialize the Model created ----> U-Net

unet = Unet().to(config.DEVICE)

# initialize the loss function and optimizer

lossObj  = BCEWithLogitsLoss()
opt = Adam(unet.parameters, lr= config.LR_INIT)

## calculate the steps per epoch-----> he number of steps required to iterate over the entire dataset

trainSteps = len(train) // config.BATCH_SIZE
testSteps = len(test) // config.BATCH_SIZE

# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}


# loop over epochs 

print("[INFO] training the network...")

start = time.time()
for epoch in tqdm(range(config.EPOCHS)):
    # set the model in training
    unet.train()

    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalTestLoss = 0

    # loop over the training set
    for (i, (x,y)) in enumerate(trainLoader):
        (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

        # perform the forward pass by ___cal__ function in nn.Module

        pred = unet(x)
        loss = lossObj(pred, y)

		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters

        opt.zero_grad()
        loss.backward()
        opt.step()

        # add the loss to the total training loss so far
        totalTrainLoss += loss
    with torch.no_grad():
        unet.eval()    
        # loop over the test data
        for (x,y) in testLoader:
            x.to(config.DEVICE), y.to(config.DEVICE)
            # make the predictions and add to the testLoss
            pred= unet(x)
            totalTestLoss += lossObj(pred, y)
    
    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss // trainSteps
    avgTestLoss = totalTestLoss// testSteps





