import numpy as np
import torch
from ODDataset_Train import OD_Dataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import pdb
from PIL import Image, ImageDraw, ImageFont

num_classes = 2
score_threshold = 0.70
min_area = 1 #minimum object size in pixels^2
logfile_name = "logfile_test.txt"

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


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #set up model
    model_state_file = 'faster_rcnn_snails.pt'
    num_classes = 2
    model = setup_model(num_classes)
    model.load_state_dict(torch.load(model_state_file))
    model.eval()
    model.to(device)

    # setup datasets and dataloaders
    folder = ['./Data/Snails_2_BH/OD_imgs_test','./Data/Snails_2_BH/OD_data_test']
    dataset = OD_Dataset(folder,get_transform(train=False), min_area)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1,
            shuffle=False, num_workers = 4, collate_fn= utils.collate_fn)

    logfile = open(logfile_name,"w")
    evaluate(model, data_loader, logfile, device)
    with torch.no_grad():
        for it, sample in enumerate(data_loader):
            img = sample[0][0]
            target = sample[1][0]
            pred = model([img.to(device)])
            #pdb.set_trace()
            boxes  = pred[0]['boxes'].to('cpu').numpy()
            labels = pred[0]['labels'].to('cpu').numpy()
            scores = pred[0]['scores'].to('cpu').numpy()
            imgPIL = torchvision.transforms.ToPILImage()(img).convert("RGBA")
            overlay_pred = Image.new('RGBA',imgPIL.size, (0,0,0,0))
            draw = ImageDraw.Draw(overlay_pred)
            n_preds = len(scores)
            for i in range(n_preds):
                if scores[i] > score_threshold:
                    draw.rectangle(boxes[i],outline = (0,0,255,127), width=3)
            imgPIL = Image.alpha_composite(imgPIL, overlay_pred)

            overlay_true = Image.new('RGBA',imgPIL.size, (0,0,0,0))
            draw = ImageDraw.Draw(overlay_true)
            true_boxes = target["boxes"].numpy()
            for box in true_boxes:
                draw.rectangle(box, outline = (255,0,0,127), width=3)
            imgPIL = Image.alpha_composite(imgPIL, overlay_true).convert("RGB")

            imgPIL.save("./output/"+ dataset.imgs[target["image_id"]] + "_preds.jpg","JPEG")


if __name__ == '__main__':
    main()
