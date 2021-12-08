import torch
import cv2
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

config_file = "detectron2_configs.yaml"
model_path = "./model_archive/frcnn_detectron_2_lr_005.pth"
cfg = get_cfg()
cfg.merge_from_file(config_file)
cfg.MODEL.WEIGHTS = model_path
#model = build_model(cfg)  # returns a torch.nn.Module
#DetectionCheckpointer(model).load(model_path)

im = cv2.imread("./Data/Snails_3_2015/OD_imgs_test/Row36_P-_0754to0928_DSC_0756_10.jpg")
predictor = DefaultPredictor(cfg)
#model.eval()
with torch.no_grad():
    preds = predictor(im)
    print(preds)
    v = Visualizer(im[:, :, ::-1])
    out = v.draw_instance_predictions(preds["instances"].to("cpu"))
    cv2.imshow('preds', out.get_image()[:, :, ::-1])
    cv2.waitKey(0)