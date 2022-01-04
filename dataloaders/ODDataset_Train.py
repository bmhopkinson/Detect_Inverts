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
    if os.path.isfile(ann_path):
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

def filter_imgs_by_objarea(anns, imgs,  min_area):
    keep = []
    for a in anns:
        boxes, labels = read_anns(a)
        if boxes.size(dim=0) == 0:
            continue

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
    def __init__(self, datainfo, transforms, min_area=0, filter_by_area=False):
        self.folders = datainfo['topfolders']
        self.dataf   = datainfo['datafolder']
        self.imgf    = datainfo['imgfolder']
        self.transforms = transforms
        self.min_area = min_area
        self.imgs = []
        self.anns = []

        #load names of image files and corresponding annotation files
        for fld in self.folders:
            img_fld = os.path.join(fld,self.imgf)
            imgs = list(sorted(os.listdir(img_fld)))
            print('num images in train folder: {}'.format(len(imgs)))
            anns = []
            imgs_mod = []
            for img in imgs:
                m = re_fbase.search(img)
                anns.append(os.path.join(fld, self.dataf, m.group(1) + '_objs.txt'))
                imgs_mod.append(os.path.join(fld, self.imgf, img))
                #imgs = map(lambda: p, f: p + f, fld * len(imgs), imgs)

            if filter_by_area:
                imgs_mod, anns = filter_imgs_by_objarea(anns, imgs_mod, self.min_area)
            print('num images in train folder after filtering: {}'.format(len(imgs_mod)))
            self.imgs.extend(imgs_mod)
            self.anns.extend(anns)
        #pdb.set_trace()

    def __getitem__(self,idx):
        #img_path = os.path.join(self.folders[0], self.imgs[idx])
        #ann_path = os.path.join(self.folders[1], self.anns[idx])
        img_path = self.imgs[idx]
        ann_path = self.anns[idx]
        img = Image.open(img_path).convert("RGB")

        #load annotations
        #boxes, labels = read_anns(ann_path)
        boxes = []
        labels = []

        if os.path.isfile(ann_path):
            with open(ann_path) as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    boxes.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])]) #xmin, ymin, xmax, ymax
                    labels.append(int(row[4]) +1) #increment b/c zero is the background class

        else: #no object annotations - just background
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.float32)

        boxes  = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])

        if boxes.size(dim=0) != 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            boxes, labels = filter_by_objarea(area, boxes, labels, self.min_area)
        else:
            area = torch.as_tensor(0, dtype=torch.float32)

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
