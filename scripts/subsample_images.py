import os
import re
import random
from shutil import copyfile

KEEP_FRAC = 0.15 #fraction of images to keep

path_regex = re.compile('.+?/(.*)$')

imdir = 'output'
outdir = './image_sections_sub/Sapelo_202106_round2'

for (dirpath, dirname, files) in os.walk(imdir, topdown='True'):
    for name in files:
        print(name)
        draw = random.random()
        print(draw)
        if draw < KEEP_FRAC:
          fullpath = os.path.join(dirpath,name)
          m = path_regex.findall(dirpath)
          dirpath_sub = m[0]
          new_dirpath = os.path.join(outdir,dirpath_sub)
          if not os.path.exists(new_dirpath):
             os.makedirs(new_dirpath)

          new_fullpath = os.path.join(new_dirpath,name);
          copyfile(fullpath, new_fullpath)
