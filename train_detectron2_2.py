#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.
This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os, csv
import sys
import re
import cv2
import numpy as np
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch

from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    print_csv_format,
)

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

logger = logging.getLogger("detectron2")
rm_ann_re = re.compile('(.*)_objs.txt$')

anns_paths_train = ["./Data/Snails_3_2015/OD_data_train", "./Data/Snails_2_BH/OD_data_train"]
img_paths_train = ["./Data/Snails_3_2015/OD_imgs_train", "./Data/Snails_2_BH/OD_imgs_train"]

anns_paths_val = ["./Data/Snails_3_2015/OD_data_val", "./Data/Snails_2_BH/OD_data_val"]
img_paths_val = ["./Data/Snails_3_2015/OD_imgs_val", "./Data/Snails_2_BH/OD_imgs_val"]


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = COCOEvaluator(dataset_name, output_dir="./output")
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

import time
def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    data_loader = build_detection_train_loader(cfg)

    logger.info("Starting training from iteration {}".format(start_iter))
    min_loss = 1000 #sys.float_info.max for some reason this caused a problem
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            if losses < min_loss:
                min_loss = losses
                checkpointer.save('best_detection_model')

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg



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
                    "bbox_mode" : detectron2.structures.BoxMode.XYXY_ABS,
                    "category_id" : int(row[4]),
                }
                objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )


    anns_paths = [*anns_paths_train, *anns_paths_val]
    img_paths = [*img_paths_train, *img_paths_val]
    dnames = [*(['snails_train']*len(anns_paths_train)), *(['snails_val']*len(anns_paths_val))]
    dnames = [name+"_"+str(i) for i, name in enumerate(dnames)]

    for dname, ann_path, img_path in zip(dnames, anns_paths, img_paths):
        f = lambda a=ann_path, i=img_path: get_snails_dicts(a, i)
        DatasetCatalog.register(dname, f)
        MetadataCatalog.get(dname).set(thing_classes=["snails"])

    cfg.DATASETS.TRAIN = tuple(dnames[0:len(anns_paths_train)])
    cfg.DATASETS.TEST = tuple(dnames[len(anns_paths_train):])

    do_train(cfg, model,  resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )