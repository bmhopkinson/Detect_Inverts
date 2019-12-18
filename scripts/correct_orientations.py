import re
import os
import piexif
from PIL import Image


re_fbase = re.compile('^(.*)\.[jJ][pP][eE]?[gG]')

base_dir = '../Data/2014'

for (dirpath, dirname, files) in os.walk(base_dir,topdown = 'True'):
    for file in files:
        fullpath = os.path.join(dirpath, file)
        m = re_fbase.search(fullpath)
        if m:
            exif_dict = piexif.load(fullpath)
            if exif_dict["0th"][274] != 1:
                im = Image.open(fullpath)
                print('{}\t{}'.format(fullpath,exif_dict["0th"][274] ))
                exif_dict["0th"][274] = 1
                exif_bytes = piexif.dump(exif_dict)
                im.save(fullpath, exif = exif_bytes)
