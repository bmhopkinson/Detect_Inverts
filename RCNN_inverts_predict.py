import numpy as np
import torch
from ODDataset_Predict import OD_Dataset_Predict
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import utils
import transforms as T
import pdb
import os
import shutil
import re
import helpers
import time

re_fbase = re.compile('^(.*)\.[jJ][pP][eE]?[gG]')

num_classes = 2
score_threshold = 0.80
OUTPUT_IMAGES = True
img_input_folder = './Data/2014'  #each directory (and its subdirectories) within this folder is processed as a unit
#img_input_folder = './Data/Snails_pred_wholetest'
section_dim = [7, 6]  #columns, rows to split input image into
pred_format = "{}\t{:4.3f}\t{:5.1f}\t{:5.1f}\t{:5.1f}\t{:5.1f}\n"

params = {'dim':section_dim, 'fmt': pred_format,'re_fbase': re_fbase, 'n_proc' : 8}

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def setup_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def make_predictions(model, data_loader, device):
    with torch.no_grad():
        for it, sample in enumerate(data_loader):
            images = list(image.to(device) for image in sample[0])
            names = sample[1]
            preds = model(images)
            #pdb.set_trace()

            for name, pred, img in zip(names, preds, images):
                pdata = {}
                pdata['boxes']  = pred['boxes'].to('cpu').numpy()
                pdata['labels'] = pred['labels'].to('cpu').numpy()
                pdata['scores'] = pred['scores'].to('cpu').numpy()

                #filter then write predictions to file, image
                pdata_filt = {'scores':[], 'labels':[], 'boxes':[] }
                for b, l, s in zip(pdata['boxes'], pdata['labels'], pdata['scores']):
                    if s > score_threshold:
                        pdata_filt['scores'].append(s)
                        pdata_filt['labels'].append(l)
                        pdata_filt['boxes'].append(b)

                m = params['re_fbase'].search(name)
                fn_out =  m.group(1) + '_preds.txt'
                helpers.write_pred(fn_out,pdata_filt,params)
                if OUTPUT_IMAGES:
                    img = img.to('cpu')
                    img = torchvision.transforms.ToPILImage()(img).convert("RGBA")
                    helpers.write_image(name, pdata_filt,img, params)

def main():

    #set up model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_state_file = 'faster_rcnn_snails.pt'
    model = setup_model(num_classes)
    model.load_state_dict(torch.load(model_state_file))
    model.eval()
    model.to(device)

    dir_units = [entry for entry in os.listdir(img_input_folder) if os.path.isdir(os.path.join(img_input_folder, entry))]
    for unit in dir_units:
        print('working on {}'.format(unit))
        start = time.time()

        base_folder = os.path.join(img_input_folder,unit)
        tmp_folder = './tmp'  #start with a clean tmp folder
        if os.path.isdir(tmp_folder):
            shutil.rmtree(tmp_folder)
            os.mkdir(tmp_folder)
        else:
            os.mkdir(tmp_folder)
        params['outfld'] = tmp_folder
        section_data = helpers.section_images(base_folder,  params)

        # setup datasets and dataloaders
        dataset = OD_Dataset_Predict(tmp_folder,get_transform(train=False))

        data_loader = torch.utils.data.DataLoader(dataset, batch_size = 4,
                shuffle=False, num_workers = 4, collate_fn= utils.collate_fn)

        make_predictions(model, data_loader, device)

        helpers.assemble_predictions(section_data, params)

        stop = time.time();
        delta_t = stop - start
        print('finished in {:4.2f} s'.format(delta_t))



if __name__ == '__main__':
    main()
