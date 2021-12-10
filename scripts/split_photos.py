import os
import cv2
import numpy as np
import re

IMG_WIDTH  = 4096
IMG_HEIGHT = 2160
N_WIDE = 6
N_HIGH = 4
N_SECTIONS = N_WIDE * N_HIGH
x_b = np.linspace(0,IMG_WIDTH , N_WIDE +1, dtype='int')
y_b = np.linspace(0,IMG_HEIGHT, N_HIGH +1, dtype='int')

path_regex = re.compile('.+?/(.*)$')
jpg_regex  = re.compile('.*.[jJ][pP][eE]*[gG]$')

def section_image(im):
   sections = []
   n_sec = 0;
   for i in range(N_HIGH):
     for j in range(N_WIDE):
        n_sec = n_sec+1

        im_sec = im[y_b[i]:y_b[i+1],x_b[j]:x_b[j+1]]
        sections.append(im_sec)

   return sections

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

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


imdir = 'Sapelo_202106/round_2'
outdir = './output'
for (dirpath, dirname, files) in os.walk(imdir, topdown='True'):
    for name in files:
        fullpath = os.path.join(dirpath,name)

        m = path_regex.findall(dirpath)
        is_imfile = jpg_regex.findall(name)
        if(is_imfile):
            dirpath_sub = m[0]
            new_dirpath = os.path.join(outdir,dirpath_sub)
            if not os.path.isdir(new_dirpath):
                os.makedirs(new_dirpath)

            final_dir = os.path.basename(dirpath)

            file_base = os.path.splitext(name)[0]
            print(fullpath)
            im = cv2.imread(fullpath)
            height , width = im.shape[:2]

            if width < height:
                im_rot = rotate_image(im, 90);
            else:
                im_rot = im;
            im_sections = section_image(im_rot)
            for i in range(N_SECTIONS):
                outfile = final_dir  + "_" + file_base + "_" + str(i) +'.jpg'
                outpath = os.path.join(new_dirpath, outfile)
                cv2.imwrite(outpath,im_sections[i])


#imfile ='./images/2014/Row1_1_2748to2797/DSC_2791.jpg'




#h, w, c = im.shape
#print('im width: %d, im height %d\n' % (w, h))
