from torch.utils.data import Dataset
import cv2

''' Custom dataset class using Pytorch Dataset Class and modifying'''

class SegDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms):
        # storing image/masks paths and transformationa pplied to the images
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms

    def __len__(self):
        #returning the total no of sample available in the tranining dataset.
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # grab image of the particular index
        imagePath = self.imagePaths[idx]
        

        
		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
        img = cv2.imread(imagePath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.maskPaths[idx], 0)

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            img = self.transforms(img)
            mask = self.transforms(mask)

		# return a tuple of the image and its mask
        return (img, mask)
