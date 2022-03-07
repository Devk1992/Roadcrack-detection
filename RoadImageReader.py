from __future__ import division
from __future__ import print_function

import os

from skimage import io
from torch.utils.data import Dataset


class RoadImageReader(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.pics_list = self.getListOfFiles(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.pics_list)

    def __getitem__(self, idx):
        img_name = self.pics_list[idx]
        target = 0 if "uncracked" in self.pics_list[idx] else 1
        print("Image_Name: "+ img_name)
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        sample = {"image": image, "target": target}

        return sample
    
    def getListOfFiles(self,dirName):
        # create a list of file and sub directories 
        # names in the given directory 
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory 
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.getListOfFiles(fullPath)
            else:
                allFiles.append(fullPath)
        return allFiles
