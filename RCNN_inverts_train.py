import numpy as np
import torch
from ODDataset_Train import OD_Dataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T

num_classes = 2
min_area = 1 #minimum object size in pixels^2

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

    # setup datasets and dataloaders
    folder_train = ['./Data/Snails_2_BH/OD_imgs_train','./Data/Snails_2_BH/OD_data_train']
    folder_val   = ['./Data/Snails_2_BH/OD_imgs_val'  ,'./Data/Snails_2_BH/OD_data_val'  ]
    dataset_train = OD_Dataset(folder_train,get_transform(train=True) , min_area )
    dataset_val   = OD_Dataset(folder_val  ,get_transform(train=False), min_area )
    #print("length of val dataset {}".format(len(dataset_val)))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=4,
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
    num_epochs = 5
    logfile = open("logfile_training_size_>1_SGD.txt",'w')

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, logfile, print_freq=20)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_val, logfile, device=device)

    torch.save(model.state_dict(),"faster_rcnn_snails.pt")
    print("That's it!")


if __name__ == "__main__":
    main()
