# import some common libraries
import numpy as np
import os, cv2, re, csv
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

rm_ann_re = re.compile('(.*)_objs.txt$')

def get_snails_dicts(ann_dir, img_dir):

    ann_files = [fname for fname in os.listdir(ann_dir) if os.path.splitext(fname)[1] == '.txt']

    dataset_dicts = []
    for idx, fname in enumerate(ann_files):
        record = {}

        m = rm_ann_re.search(fname)
        if m:
            img_name_base = m[1]
        else:
            continue

        filename = os.path.join(img_dir, img_name_base+".jpg")
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        with open(os.path.join(ann_dir, fname), "r") as anns:
            reader = csv.reader(anns, delimiter='\t')
            for row in reader:
                bbox = [row[0], row[1], row[2], row[3]]  # xmin, ymin, xmax, ymax
                bbox = [int(np.rint(float(x))) for x in bbox]
                obj = {
                    "bbox" : bbox,
                    "bbox_mode" : BoxMode.XYXY_ABS,
                    "category_id" : int(row[4]),
                }
                objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

anns_paths = ["./Data/Snails_3_2015/OD_data_train", "./Data/Snails_2_BH/OD_data_train", "./Data/Snails_3_2015/OD_data_val"]
img_paths = ["./Data/Snails_3_2015/OD_imgs_train", "./Data/Snails_2_BH/OD_imgs_train", "./Data/Snails_3_2015/OD_imgs_val"]
for d, ann_path, img_path in zip(["train_1", "train_2", "val"] , anns_paths, img_paths):
    f = lambda a=ann_path, i = img_path: get_snails_dicts(a, i)
    DatasetCatalog.register("snails_" + d, f )
    MetadataCatalog.get("snails_" + d).set(thing_classes=["snails"])
snails_metadata = MetadataCatalog.get("snails_train")

dataset_dicts = get_snails_dicts(anns_paths[0], img_paths[0])

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=snails_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow('train', out.get_image()[:, :, ::-1])
    cv2.waitKey(1000)


from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("snails_train_1", "snails_train_2",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 8
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()