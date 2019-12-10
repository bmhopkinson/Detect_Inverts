import os
import numpy as np
import torch
import csv
from PIL import Image
import pdb

class OD_Dataset_Test(object):
    def __init__(self,folders,transforms):
        self.folders = folders
        self.transforms = transforms

        #load names of image files and corresponding annotation files
        #sort to ensure order matches
        self.imgs = list(sorted(os.listdir(folders[0])))
        self.anns = list(sorted(os.listdir(folders[1])))

    def __getitem__(self,idx):
        img_path = os.path.join(self.folders[0], self.imgs[idx])
        ann_path = os.path.join(self.folders[1], self.anns[idx])
        img = Image.open(img_path).convert("RGB")

        #load annotations
        boxes = []
        labels = []
        with open(ann_path) as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                boxes.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])]) #xmin, ymin, xmax, ymax
                labels.append(int(row[4]) +1) #increment b/c zero is the background class

        num_objs = len(labels)
        boxes  = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["file_name"]  = self.imgs[idx]

#        pdb.set_trace()
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
