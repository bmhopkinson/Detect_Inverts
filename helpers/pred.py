import os
import re
import parse
import pdb
import helpers.image
import multiprocessing as mp
import math
from PIL import Image


path_regex = re.compile('.+?/(.*)$')
#jpg_regex  = re.compile('^(.*)\.[jJ][pP][eE]?[gG]')
#temp_regex = re.compile('./tmp(.*)')


def write_pred(fname, data, params):
    fout = open(fname,'w')
    for b, l, s in zip(data['boxes'], data['labels'], data['scores']):
        fout.write(params['fmt'].format(l, s, b[0], b[1], b[2], b[3]))
    fout.close()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


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
        if params['write_imgs']:
            full_img = Image.open(full_img_file)
            full_img = helpers.image.properly_orient_image(full_img)
            full_img = full_img.convert("RGBA")
            helpers.image.write_image(out_img_file, pred_data, full_img, params)

def assemble_predictions(section_data, params):
    n_proc = params['n_proc']
    jobs = []
    for chunk in chunks(section_data.keys(),math.ceil(len(section_data.keys())/n_proc)):
        j = mp.Process(target = _assemble_predictions, args = (chunk, section_data, params)) #this works - actually uses multiple cores
        j.start()
        jobs.append(j)

    for j in jobs:
        j.join()
