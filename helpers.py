import os
import cv2
import numpy as np
import re
import parse
import pdb
import multiprocessing as mp
import time
import math
from PIL import Image, ImageDraw, ImageFont


path_regex = re.compile('.+?/(.*)$')
#jpg_regex  = re.compile('^(.*)\.[jJ][pP][eE]?[gG]')
#temp_regex = re.compile('./tmp(.*)')

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


def write_pred(fname, data, params):
    fout = open(fname,'w')
    for b, l, s in zip(data['boxes'], data['labels'], data['scores']):
        fout.write(params['fmt'].format(l, s, b[0], b[1], b[2], b[3]))
    fout.close()

def _section_image(im, section_dim):
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

#_fsec is core of splitting image - called by parallel prosesing pool
def _fsec(sec_data,files, dirpath, params):

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
            #print(fullpath)
            im = cv2.imread(fullpath)
            height , width = im.shape[:2]

            if width < height:
                im_rot = rotate_image(im, 90);
            else:
                im_rot = im;
            im_sections, offsets = _section_image(im_rot, params['dim'])

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
    n_proc = 8
#    pool = mp.Pool(processes = n_proc)
    manager = mp.Manager()
    sec_data = manager.dict()
    start = time.time()

    for (dirpath, dirname, files) in os.walk(infolder, topdown='True'):
        #send image files to split to n_proc different processes - all data is added to sec_data (manager.dict() - thread safe dict)
        jobs = []
        for chunk in chunks(files,math.ceil(len(files)/n_proc)):
            #pdb.set_trace()
            #pool.apply(_fsec, args = (sec_data,chunk, dirpath, params))  #this didn't work for me - always used a single core
            j = mp.Process(target = _fsec, args = (sec_data,chunk, dirpath, params)) #this works - actually uses multiple cores
            j.start()
            jobs.append(j)

        for j in jobs:
            j.join()

    stop = time.time();
    delta_t = stop - start
    print('done splitting images {:4.2f}'.format(delta_t))
    return sec_data

def _assemble_predictions(im_files, section_data,params):
    for im_file in im_files:
        labels = []
        scores = []
        boxes = []
        section_dim = params['dim']
        fmt_str = params['fmt']
        n_sec = section_dim[0] * section_dim[1]
        offsets = section_data[im_file][1]
        for i in range(n_sec):
            fname = im_file + "_" + str(i) + "_preds.txt"
            f = open(fname,'r')
            for line in f:
                pl = parse.parse(fmt_str,line)
                #pdb.set_trace()
                labels.append(int(pl[0]))
                scores.append(pl[1])
                boxes.append([pl[2] + offsets[i][0], pl[3] + offsets[i][1], pl[4] + offsets[i][0], pl[5] + offsets[i][1] ])

            f.close()
        pred_data = {'labels': labels, 'scores': scores, 'boxes': boxes}  #package data in standard format

        #setup output files and directories
        full_img_file = section_data[im_file][0]
        m = path_regex.search(full_img_file)
        out_img_file = os.path.join('./Preds',m.group(1))
        out_dir = os.path.dirname(out_img_file)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        m2 = params['re_fbase'].search(out_img_file)
        out_preds_file    = m2.group(1) + '_preds.txt'

        #write out predictions
        write_pred(out_preds_file, pred_data, params)

        #mark predictions on images
        full_img = Image.open(full_img_file).convert("RGBA")  #NEED TO TEST WHETHER TO ROTATE IMAGE
        write_image(out_img_file, pred_data, full_img, params)

def assemble_predictions(section_data, params):
    n_proc = 8
    print('assembling predictions')
    start = time.time()
    jobs = []
    for chunk in chunks(section_data.keys(),math.ceil(len(section_data.keys())/n_proc)):
        j = mp.Process(target = _assemble_predictions, args = (chunk, section_data, params)) #this works - actually uses multiple cores
        j.start()
        jobs.append(j)

    for j in jobs:
        j.join()

    stop = time.time()
    delta_t = stop - start
    print('done assembling predictions {:4.2f}'.format(delta_t))
