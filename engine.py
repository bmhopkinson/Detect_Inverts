import math
import sys
import time
import torch
import pdb
import numpy as np
import copy
from collections import defaultdict

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, logfile, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(logfile, delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def filter_results(coco_evaluator, score_thresh):
    #currently only works for a single object class
    coco_eval = coco_evaluator.coco_eval["bbox"]
    #coco_eval.evaluate()
    #clear matches below threshold, then evaluate precision, recall
    #numpy version
    #fitler results
    evalFilt = defaultdict(dict)
    p = coco_eval.params
    for idx, img in enumerate(coco_eval.evalImgs):
        #pdb.set_trace()
        dtMatches = np.copy(img['dtMatches'])
        nT = len(dtMatches)
        iclr = [  s < score_thresh  for s in img['dtScores']]
        iclr_np = np.asarray(iclr, dtype=bool)
        dtMatches[:,iclr_np] = -1.0  #indicates to ignore

        gtMatches = np.zeros(img['gtMatches'].shape)
        gtMatches_filt = []
        gtIds = img['gtIds']
        #pdb.set_trace()
        for i, gtM in enumerate(np.copy(img['gtMatches'])):
            #gtM_fr = [0.0 if iclr[int(m)] else m for m in gtM]
            gtM_fr = []
            for m in gtM:
                m = int(m)
                if m == 0:
                    gtM_fr.append(0.0)
                else:
                    gtM_fr.append(0.0 if iclr[m-1] else float(m))
            gtMatches_filt.append(np.asarray(gtM_fr,dtype=np.float64))
            gtMatches[i,] = np.asarray(gtM_fr,dtype=np.float64)

        aRngidx = p.areaRng.index(img['aRng'])
        evalFilt[img['image_id']][p.areaRngLbl[aRngidx]] = {'dtMatches': dtMatches,'gtMatches': gtMatches}
        #pdb.set_trace()

    #determine performance metrics on filtered results
    TP = np.zeros(shape=(10,),dtype=int)
    TP2= np.zeros(shape=(10,),dtype=int)
    FP = np.zeros(shape=(10,),dtype=int)
    FN = np.zeros(shape=(10,),dtype=int)

    for img in evalFilt:
        for rng in evalFilt[img]:
            if rng == p.areaRngLbl[0]:  #all size objects
                TP = TP + np.sum(evalFilt[img][rng]['dtMatches'] > 0.0, axis = 1 )
                FP = FP + np.sum(np.absolute(evalFilt[img][rng]['dtMatches']) < 0.1, axis = 1 )
                TP2= TP2+ np.sum(evalFilt[img][rng]['gtMatches'] > 0.0, axis = 1 )
                FN = FN + np.sum(evalFilt[img][rng]['gtMatches'] < 0.1, axis = 1 )
    #print("TP: {}, TP2: {}, FP: {} , FN: {}".format(TP, TP2, FP, FN))

    #computer precision and recall for IoU = 0.5
    pr = TP[0]/(TP[0] + FP[0])
    rc = TP[0]/(TP[0] + FN[0])
    return {'TP': TP[0], 'FP': FP[0], 'FN': FN[0], 'PR': pr, 'RC': rc}

@torch.no_grad()
def evaluate(model, data_loader, logfile, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(logfile, delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    #pdb.set_trace()
    #eval_trials(coco_evaluator)
    score_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for thresh in score_thresholds:
        res_filt = filter_results(coco_evaluator, thresh)
        print_str = "Thresh: {:4.3f}, TP: {}, FP: {} , FN: {}, PR: {:4.3f}, RC: {:4.3f}".format(thresh, res_filt['TP'],res_filt['FP'],res_filt['FN'],res_filt['PR'],res_filt['RC'])
        print(print_str)
        logfile.write(print_str + '\n')

    torch.set_num_threads(n_threads)
    return coco_evaluator
