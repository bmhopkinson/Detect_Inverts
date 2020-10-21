import os
import numpy as np
import torch
import csv
from PIL import Image, ExifTags
import pdb
import re
import helpers

re_fbase = re.compile('^(.*)\.[jJ][pP][eE]?[gG]')

class OD_Dataset_Predict(object):
    def __init__(self,folder,transforms):
        self.folder = folder
        self.transforms = transforms

        #load names of image files and corresponding annotation files
        img_fnames = []
        for (dirpath, dirname, files) in os.walk(folder, topdown='True'):
                for name in files:
                    fullpath = os.path.join(dirpath,name)
                    img_fnames.append(fullpath)
        self.imgs = img_fnames

    def __getitem__(self,idx):
        img = Image.open(self.imgs[idx]).convert("RGB")

        if self.transforms is not None:
            target = None
            img, target = self.transforms(img,target)

        return img, self.imgs[idx]

    def __len__(self):
        return len(self.imgs)
