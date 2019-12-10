import os
import numpy as np
import torch
import csv
from PIL import Image
import pdb
import re

re_fbase = re.compile('^(.*)\.[jJ][pP][eE]?[gG]')
#re_fbase = re.compile('^(.*).jpg')

def read_anns(ann_path):
    #load annotations
    boxes = []
    labels = []
    with open(ann_path) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            boxes.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])]) #xmin, ymin, xmax, ymax
            labels.append(int(row[4]) +1) #increment b/c zero is the background class
    boxes  = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)

    return boxes, labels

def filter_by_objarea(area, boxes, labels, min_area):
    exceed = area > min_area
    boxes_filt  = boxes[exceed]
    labels_filt = labels[exceed]
    return boxes_filt, labels_filt

def filter_imgs_by_objarea(anns,imgs, folder, min_area):
    keep = []
    for i in anns:
        ann_path = os.path.join(folder, i)
        boxes, labels = read_anns(ann_path)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        exceed = area > min_area
        if sum(exceed) >= 1:
            keep.append(True)
        else:
            keep.append(False)

    anns_filt = [k[1] for k in zip(keep,anns) if k[0] ]
    imgs_filt = [k[1] for k in zip(keep,imgs) if k[0] ]

    return imgs_filt, anns_filt


class OD_Dataset(object):
    def __init__(self,folders,transforms, min_area):
        self.folders = folders
        self.transforms = transforms
        self.min_area = min_area

        #load names of image files and corresponding annotation files
        imgs = list(sorted(os.listdir(folders[0])))
        anns = []
        for img in imgs:
            m = re_fbase.search(img)
            anns.append(m.group(1) + '_objs.txt')

        self.imgs, self.anns = filter_imgs_by_objarea(anns,imgs, self.folders[1], self.min_area)

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


        boxes  = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        boxes, labels = filter_by_objarea(area, boxes, labels, self.min_area)
        num_objs = len(labels)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

#        pdb.set_trace()
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
