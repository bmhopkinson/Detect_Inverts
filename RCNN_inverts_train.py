import engine
import utils
from dataloaders.ODDataset_Train import OD_Dataset
import dataloaders.transforms as T

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import evaluate


num_classes = 2
min_area = 1 #minimum object size in pixels^2
num_epochs = 10
batch_size_train = 4
batch_size_val = 1  # only works w/ batchsize 1 right now
model_save_path = "faster_rcnn_snails_202106_2.pt"
tensorboard_path = "runs/FRCNN_snails_1"

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

    writer = SummaryWriter(tensorboard_path)

    # setup datasets and dataloaders
    #train_datainfo = {'topfolders' : ['./Data/Snails_2_BH', './Data/Snails_3_2015'], 'datafolder' :'OD_data_train', 'imgfolder' : 'OD_imgs_train' }
    #val_datainfo   = {'topfolders' : ['./Data/Snails_2_BH', './Data/Snails_3_2015'], 'datafolder' :'OD_data_val',   'imgfolder' : 'OD_imgs_val'   }
    train_datainfo = {'topfolders' : ['./Data/Snails_Sapelo_202106'], 'datafolder' :'OD_data_train', 'imgfolder' : 'OD_imgs_train' }
    val_datainfo   = {'topfolders' : ['./Data/Snails_Sapelo_202106'], 'datafolder' :'OD_data_val', 'imgfolder' : 'OD_imgs_val' }

    dataset_train = OD_Dataset(train_datainfo, get_transform(train=True) , min_area )
    dataset_val   = OD_Dataset(val_datainfo  , get_transform(train=False), min_area )
    #print("length of val dataset {}".format(len(dataset_val)))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size_train, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size_val, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # setup model
    model = setup_model(num_classes)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    #optimizer = torch.optim.Adam(params, lr = 1E-5, weight_decay=0.0005 )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for X epochs
    logfile = open("logfile_training_size_>1_SGD.txt",'w')

    trainer = engine.Trainer(model, optimizer, device, logfile, writer, print_freq=20)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        trainer.train_one_epoch(data_loader_train)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_val, logfile, device, writer, epoch)

    torch.save(model.state_dict(), model_save_path)
    print("That's it!")


if __name__ == "__main__":
    main()
