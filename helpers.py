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
    m = params[2].search(name)
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


def write_pred(name, data, params):
    m = params[2].search(name)
    fn_out =  m.group(1) + '_preds.txt'
    fout = open(fn_out,'w')
    for b, l, s in zip(data['boxes'], data['labels'], data['scores']):
        fout.write(params[1].format(l, s, b[0], b[1], b[2], b[3]))
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
    sector_dim = params[0]
    for name in files:
        fullpath = os.path.join(dirpath,name)

        m = path_regex.findall(dirpath)
        is_imfile = params[2].findall(name)
        if(is_imfile):
            dirpath_sub = m[0]
            new_dirpath = os.path.join(params[-1],dirpath_sub)
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
            im_sections, offsets = _section_image(im_rot, sector_dim)

            for i in range(len(im_sections)):
                outfile =  file_base + "_" + str(i) +'.jpg'
                outpath = os.path.join(new_dirpath, outfile)
                cv2.imwrite(outpath,im_sections[i])

            sec_data[os.path.join(new_dirpath,file_base)] = [fullpath,offsets]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def section_images(infolder, outdir, params):
    n_proc = 8
#    pool = mp.Pool(processes = n_proc)
    manager = mp.Manager()
    sec_data = manager.dict()
    params.append(outdir)
    print('splitting images')
    start = time.time()
    for (dirpath, dirname, files) in os.walk(infolder, topdown='True'):
        jobs = []
        for chunk in chunks(files,math.ceil(len(files)/n_proc)):
            #pdb.set_trace()
            #pool.apply(_fsec, args = (sec_data,chunk, dirpath, params))  #this didn't work for me - always used a single core
            p = mp.Process(target = _fsec, args = (sec_data,chunk, dirpath, params))
            p.start()
            jobs.append(p)
            print('started new job')

        for j in jobs:
            j.join()

    stop = time.time();
    delta = stop - start
    print('done splitting images {}'.format(delta))
    return sec_data

def assemble_predictions(section_data, params):
    for im_file in section_data:
        labels = []
        scores = []
        boxes = []
        section_dim = params[0]
        fmt_str = params[1]
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

        #setup output files and directories
        full_img_file = section_data[im_file][0]
        m = path_regex.search(full_img_file)
        out_img_file = os.path.join('./Preds',m.group(1))
        out_dir = os.path.dirname(out_img_file)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        #WORKING AROUD HERE - NOT ALL out_preds_file WENT TO THE RIGTH PLACE
        m2 = params[2].search(out_img_file)
        out_preds_file    = m2.group(1) + '_preds.txt'

        fout = open(out_preds_file,"w")
        for j in range(len(labels)):
            fout.write(fmt_str.format(labels[j], scores[j],boxes[j][0], boxes[j][1], boxes[j][2], boxes[j][3]))

        pred_data = {'labels': labels, 'scores': scores, 'boxes': boxes}

        full_img = Image.open(full_img_file).convert("RGBA")  #NEED TO TEST WHETHER TO ROTATE IMAGE


        write_image(out_img_file, pred_data, full_img, params)
