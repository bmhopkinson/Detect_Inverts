import csv
import re
import numpy as np
import os
import shutil
from collections import defaultdict

infile = './snails_sapelo2021_round1_vott-csv-export/Marsh_Inverts_1-export.csv'
infolder = 'snails_sapelo2021_round1_vott-csv-export'
outfolder_data= './OD_data/'
outfolder_imgs= './OD_imgs/'

cats = {'snail' : 0,'Dead_Snail' : 1}
keep = {0}
counts = defaultdict(lambda: 0)
n_cats = len(cats)

ADD_DUMMYOBJ_TO_EMPTYIMAGES = True # cannot process images that have no annotations, so optionally add a dummy object to images otherwise lacking annotated objecters
DUMMYOBJ_SIZE = 15
DUMMY_ID = 1
dummy_mod_tag = '_dummy'

ext = '.jpg'
re_string = '^(.*)(?<!{0}){1}'.format(dummy_mod_tag, ext)  #uses negative lookbehind to avoid adding dummy objects repeatedly
print('re_string: {}'.format(re_string))
re_fid = re.compile(re_string)

def log_object(data,new_obj,idx):
     data.append([new_obj[1:5],idx])
     return data

csvfile = open(infile, newline='')
freader = csv.reader(csvfile, delimiter = ',')

#organize data
data = dict()
for felm in freader:
    #print(felm)
    m = re_fid.search(felm[0])
    if m:
      #print("%s %s" % (m.group(1), m.group(2)))
      pic = m.group(1)
      idx = cats[felm[5]]
      if idx in keep:
         if pic in data:
            data[pic] = log_object(data[pic], felm, idx)
         else:
            data[pic] = []#set up dict entry
            data[pic] = log_object(data[pic], felm, idx)
         counts[idx] += 1

    else:
      print("no match")

if ADD_DUMMYOBJ_TO_EMPTYIMAGES:
    from PIL import Image, ImageDraw
    import random



    #first identify images w/out annotations
    empty_imgs = []
    for file in os.listdir(infolder):
        m = re_fid.search(file)
        if m:
            if m.group(1) in data:
                continue
            else:
                empty_imgs.append(m.group(1))

    #add dummy object to image and log it as an object in "data"
    for img in empty_imgs:
        img_path = os.path.join(infolder,img+ext)
        imgPIL = Image.open(img_path)
        width, height = imgPIL.size

        obj_col_center = (width - 2 * DUMMYOBJ_SIZE) * random.uniform(0, 1) + DUMMYOBJ_SIZE
        obj_row_center = (height - 2 * DUMMYOBJ_SIZE) * random.uniform(0, 1) + DUMMYOBJ_SIZE
        felm = []

        xmin = (obj_col_center - DUMMYOBJ_SIZE)
        ymin = (obj_row_center - DUMMYOBJ_SIZE)
        xmax = (obj_col_center + DUMMYOBJ_SIZE)
        ymax = (obj_row_center + DUMMYOBJ_SIZE)

        felm = ['ignored', xmin, ymin, xmax, ymax]
        key = img+dummy_mod_tag
        data[key] = []
        data[key] = log_object(data[key], felm, DUMMY_ID)

        draw = ImageDraw.Draw(imgPIL)
        draw.ellipse((xmin, ymin, xmax, ymax), fill='red', outline='red')
        out_path = os.path.join(infolder,img + dummy_mod_tag + ext )
        imgPIL.save(out_path, format='JPEG', subsampling=0, quality=100)


    print(empty_imgs)
    print(len(empty_imgs))

for c in counts:
    print('category {}, total counts: {}'.format(c, counts[c]))
#write out data to files, also:
#not all images will have objects in them. copy only images with objects present to new directory
for pic in data:
    fout = open(outfolder_data + pic + '_objs.txt','w')
    for obj in data[pic]:
        fout.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(obj[0][0], obj[0][1], obj[0][2],obj[0][3],obj[1]))
    fout.close()
    shutil.copyfile(infolder+'/'+pic+'.jpg', outfolder_imgs+'/'+pic+'.jpg')
