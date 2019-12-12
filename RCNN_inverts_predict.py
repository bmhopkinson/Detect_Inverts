import numpy as np
import torch
from ODDataset_Predict import OD_Dataset_Predict
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import utils
import transforms as T
import pdb
from PIL import Image, ImageDraw, ImageFont
import re

re_fbase = re.compile('^(.*)\.[jJ][pP][eE]?[gG]')

num_classes = 2
score_threshold = 0.70
OUTPUT_IMAGES = True

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

def write_pred(name, data):
    m = re_fbase.search(name)
    fn_out = './output/preds/' + m.group(1) + '_preds.txt'
    fout = open(fn_out,'w')
    for b, l, s in zip(data['boxes'], data['labels'], data['scores']):
        if s > score_threshold:
            fout.write("{}\t{:4.3f}\t{:5.1f}\t{:5.1f}\t{:5.1f}\t{:5.1f}\n".format(l, s, b[0], b[1], b[2], b[3]))
    fout.close()

def write_image(name, data, img):
    #visualize predctions on images
    m = re_fbase.search(name)
    imgPIL = torchvision.transforms.ToPILImage()(img).convert("RGBA")
    overlay_pred = Image.new('RGBA',imgPIL.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay_pred)
    n_preds = len(data['scores'])

    for i in range(n_preds):
        if data['scores'][i] > score_threshold:
            draw.rectangle(data['boxes'][i],outline = (0,0,255,127), width=3)
    imgPIL = Image.alpha_composite(imgPIL, overlay_pred).convert("RGB")
    imgPIL.save("./output/"+ m.group(1) + "_preds.jpg","JPEG")

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #set up model
    model_state_file = 'faster_rcnn_snails.pt'
    model = setup_model(num_classes)
    model.load_state_dict(torch.load(model_state_file))
    model.eval()
    model.to(device)

    # setup datasets and dataloaders
    folder = ['./Data/Snails_pred_test']
    dataset = OD_Dataset_Predict(folder,get_transform(train=False))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 2,
            shuffle=False, num_workers = 4, collate_fn= utils.collate_fn)

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

                #write predictions to file, image
                write_pred(name,pdata)
                if OUTPUT_IMAGES:
                    img = img.to('cpu')
                    write_image(name, pdata,img)

if __name__ == '__main__':
    main()
