import os
import numpy as np
import torch
import csv
from PIL import Image
import pdb
import re

re_fbase = re.compile('^(.*)\.[jJ][pP][eE]?[gG]')


class OD_Dataset_Predict(object):
    def __init__(self,folders,transforms):
        self.folders = folders
        self.transforms = transforms

        #load names of image files and corresponding annotation files
        self.imgs = list(sorted(os.listdir(folders[0])))

    def __getitem__(self,idx):
        img_path = os.path.join(self.folders[0], self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            target = None
            img, target = self.transforms(img,target)

        return img, self.imgs[idx]

    def __len__(self):
        return len(self.imgs)
