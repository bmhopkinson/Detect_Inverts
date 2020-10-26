import os
import re
import cv2
import math
import numpy as np
import multiprocessing as mp
from PIL import Image, ImageDraw, ImageFont, ExifTags

path_regex = re.compile('.+?/(.*)$')

def PIL_to_cv2(img_PIL):
    return cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

def write_image(name, data, imgPIL, params):
    #visualize predctions on images
    m = params['re_fbase'].search(name)
    overlay_pred = Image.new('RGBA',imgPIL.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay_pred)
    n_preds = len(data['scores'])

    for i in range(n_preds):
        draw.rectangle(data['boxes'][i],outline = (0,0,255,127), width=4)
    imgPIL = Image.alpha_composite(imgPIL, overlay_pred).convert("RGB")

    dir = os.path.dirname(name)
    if not os.path.isdir(dir):
        os.makedirs(dir)
    imgPIL.save( m.group(1) + "_preds.jpg","JPEG")




def extract_exif_data(img):
    img_exif = []
    try:
        img_exif = img._getexif()
    except:
        if hasattr(img, 'filename'):
            print('{} has no exif data'.format(img.filename))
        else:
            print('image has no exif data')

    return img_exif

def text_exif_labels(exif):
    labeled = {}
    for (key, val) in exif.items():
        labeled[ExifTags.TAGS.get(key)] = val

    return labeled

def properly_orient_image(image):
    img_exif = extract_exif_data(image)
    if(img_exif):
      img_exif = text_exif_labels(img_exif)
    else:
      return image

    orient_to_angle = {
        1: 0,
        3: 180,
        6: 90,
        8: 270
    }

    if('Orientation' in img_exif):
        orient = img_exif['Orientation']
        image = image.rotate(orient_to_angle[orient], expand = True)

    else:
        if hasattr(img, 'filename'):
          print('no orientation in exif of {}'.format(image.filename))
        else:
          print('no orientation in exif of image')

    return image

#rotate_image() is from https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides/47248339

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height boundsfor name in files
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat




def _section_single_image(im, section_dim):
   sections = []  #image sections
   offsets = []   #x,y offests of sections
   n_wide = section_dim[0]
   n_high = section_dim[1]
   im_height , im_width = im.shape[:2]
   x_b = np.linspace(0,im_width , n_wide +1, dtype='int')
   y_b = np.linspace(0,im_height, n_high +1, dtype='int')

   for i in range(n_high):
     for j in range(n_wide):
        im_sec = im[y_b[i]:y_b[i+1],x_b[j]:x_b[j+1]]
        sections.append(im_sec)
        offsets.append([x_b[j], y_b[i]])

   return sections, offsets


#_section_images is core of splitting image - called by parallel prosesing pool
def  _section_images(sec_data,files, dirpath, params):
#    files = [f for f in files if not re.match(r'^\.',f)] #remove mac hidden files which start with dot
    for name in files:
        fullpath = os.path.join(dirpath,name)
        m = path_regex.findall(dirpath)
        is_imfile = params['re_fbase'].findall(name)

        if(is_imfile):
            dirpath_sub = m[0]
            new_dirpath = os.path.join(params['outfld'],dirpath_sub)
            if not os.path.isdir(new_dirpath):
                os.makedirs(new_dirpath)

            file_base = os.path.splitext(name)[0]

            with Image.open(fullpath) as im:   #PIL image is lazy loading, weird file acces, so best to manage context using "with"
                im_rot = properly_orient_image(im)
                im_rot = PIL_to_cv2(im_rot)

                im_sections, offsets = _section_single_image(im_rot, params['dim'])

                for i in range(len(im_sections)):
                    outfile =  file_base + "_" + str(i) +'.jpg'
                    outpath = os.path.join(new_dirpath, outfile)
                    cv2.imwrite(outpath,im_sections[i])

                sec_data[os.path.join(new_dirpath,file_base)] = [fullpath,offsets]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def section_images(infolder, params):
    n_proc = params['n_proc']
#    pool = mp.Pool(processes = n_proc)
    manager = mp.Manager()
    sec_data = manager.dict()

    for (dirpath, dirname, files) in os.walk(infolder, topdown='True'):
        #send image files to split to n_proc different processes - all data is added to sec_data (manager.dict() - thread safe dict)
        jobs = []
        files = [f for f in files if not re.match(r'^\.',f)] #remove mac hidden files which start with dot
        for chunk in chunks(files,math.ceil(len(files)/n_proc)):
            #pdb.set_trace()
            #pool.apply(_fsec, args = (sec_data,chunk, dirpath, params))  #this didn't work for me - always used a single core
            j = mp.Process(target = _section_images, args = (sec_data,chunk, dirpath, params)) #this works - actually uses multiple cores
            j.start()
            jobs.append(j)

        for j in jobs:
            j.join()

    return sec_data
