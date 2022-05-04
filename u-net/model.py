from turtle import forward
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
from . import config

import torch

class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        # storing the convolution and relu layers ..
        self.conv1 = Conv2d(inChannels, outChannels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels)

    def forward(self, x):
        # connecting the layers
        return self.conv2(self.relu(self.conv1(x)))

class Encoder(Module):
    def __init__(self, channels = (3, 16, 32, 64)):
        super().__init__()
        # store the encoder blocks and maxpooling layer
        self.encBlocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)
    
    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        blockOutputs = []
        for block in self.encBlocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        # return the list containing the intermediate outputs
        return blockOutputs
        

class Decoder(Module):
    def __init__(self, channels = (64, 32, 16, 3)):
        super().__init__()
        self.channels =channels
        self.upconvs = ModuleList(ConvTranspose2d(channels[i],channels[i+1],2,2) for i in range(len(channels)-1))

    def forward(self, x, encFeatures):
        #loop through the number of channels
        for i in range(len(self.channels)-1):
            x = self.upconvs[i](x) 
        
        # crop the current features from the encoder blocks
        # concatenate them with the cureent upsampled features,
        # and pass the concatenated output through the decoder block

            encFeat = self.crop(encFeatures[i],x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocs[i](x)

         #return the final decoder output
        return x

    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder 
        # features to match the dimensions
        (_,_, H, W) = x.shape
        encFeatures = CenterCrop((H,W)) (encFeatures)
        # return the cropped features
        return encFeatures


class Unet(Module):
    def __init__(self, encChannels = (2, 16, 32, 64), decChannels = (64, 32, 16), nbClasses = 1, retainDim =True, 
    outsize = (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)

        # initialize the regression head and store the class variables

        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outsize
    
    def forward(self, x):
        # grab the feature from the encoders
        encFeatures = self.encoder(x)

        # pass the encoder features through decoder making sure that their dimensions are suited for concatenation

        decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])

        # pass the decoder features through the regression head to obtain the segmentation mask
        map = self.head(decFeatures)

        # check to see if we are retaining the original output dimensions and if so, then resize the output to match them

        if self.retainDim:
            map = F.interpolate(map, self.outsize)
        
        # return the segmentation map
        return map