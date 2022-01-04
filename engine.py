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

class Trainer:
    def __init__(self, model, optimizer, device, logfile, writer, print_freq):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logfile = logfile
        self.writer = writer
        self.print_freq = print_freq
        self.epoch = 0

    def train_one_epoch(self, data_loader ):
        self.model.train()
        device = self.device

        metric_logger = utils.MetricLogger(self.logfile, delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(self.epoch)

        lr_scheduler = None
        if self.epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = utils.warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)

        running_loss = 0.0
        for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, self.print_freq, header)):
            images = list(image.to(device) for image in images)

        #    print(targets)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            running_loss = running_loss + losses

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()


            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

            if i % self.print_freq == 0:
                 self.writer.add_scalar('training_loss_iter', loss_value, self.epoch*len(data_loader) + i)

        self.writer.add_scalars('loss_epoch', {'train': running_loss / i}, self.epoch)
        self.epoch = self.epoch + 1



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

def rec_dd(): #recursive default dict
    return defaultdict(rec_dd)

def filter_dtMatches_by_score(score_thresh, scores, dtMatches):
    dtMatches = np.copy(dtMatches)
    iclr = [s < score_thresh for s in scores]
    iclr_np = np.asarray(iclr, dtype=bool)
    dtMatches[:, iclr_np] = -1.0  # indicates to ignore
    return dtMatches, iclr

def filter_gtMatches_by_score(iclr, dtIds, _gtMatches):
    _gtMatches = np.copy(_gtMatches)
    dtids_to_idx = {int(dtid): idx for idx, dtid in enumerate(dtIds)}  # map from detected id to index of iclr (ids are not always == to idx)

    gtMatches = np.zeros(_gtMatches.shape)
    for i, gtM in enumerate(_gtMatches):
        # gtM_fr = [0.0 if iclr[int(m)] else m for m in gtM]
        gtM_fr = []
        for dt_idx in gtM:
            dt_idx = int(dt_idx)
            if dt_idx == 0:  # no match
                gtM_fr.append(0.0)
            elif dt_idx == -1: # ignore
                gtM_fr.append(-1.0)
            else:
                gtM_fr.append(0.0 if iclr[dtids_to_idx[dt_idx]] else float(dt_idx))

        gtMatches[i,] = np.asarray(gtM_fr, dtype=np.float64)

        return gtMatches


def filter_anns_by_area(area_thresh, img, ann_data):
    #identify anns (ground truth) objects that fall below area_thresh
    iclr = []
    for col, gtid in enumerate(img['gtIds']):
        if ann_data[gtid]['area']  < area_thresh:
            iclr.append({'col': col, 'id': gtid})

    dtMatches_to_remove = []
    gtMatches = np.copy(img['gtMatches'])
    dtMatches = np.copy(img['dtMatches'])
    dtids_to_idx = {dtid: idx for idx, dtid in enumerate(img['dtIds'])}  # map from detected id to index of iclr (ids are not always == to idx)
    dtids_to_idx[0] = -1    # zero key indicates there was no match to a detection, mark with -1

    #record dtMatches corresponding to gtMatches that fall below area threshold
    dtMatches_to_remove_by_gtMatch = []
    for elm in iclr:
        dtids = list(gtMatches[:, elm['col']].astype(int))
        dtidxs = [dtids_to_idx[i] for i in dtids]
        dtMatches_to_remove_by_gtMatch.append(dtidxs)

    #remove corresponding dtMatches
    for dts in dtMatches_to_remove_by_gtMatch:
        for i, j in enumerate(dts):
            if j >= 0:
                dtMatches[i, j] = -1.0

    #remove gtMatches columns corresponding to anns that fell below area_thresh
    ignore_cols = [elm['col'] for elm in iclr]
    gtMatches[:, ignore_cols] = -1.0

    return dtMatches, gtMatches



def filter_results(coco_evaluator, score_thresh, area_thresh):
    coco_eval = coco_evaluator.coco_eval["bbox"]
    ann_data = coco_eval.cocoGt.anns #dictionary, key: gt Annotation id, value dict of data including 'bbox' and 'area'
    #clear matches below threshold, then evaluate precision, recall

    evalFilt = rec_dd()
    p = coco_eval.params
    for idx, img in enumerate(coco_eval.evalImgs):
        #pdb.set_trace()
        if img is None:  #no objects were detected
            continue
        category = img['category_id']
        dtMatches, gtMatches = filter_anns_by_area(area_thresh, img, ann_data)
        # dtMatches, iclr = filter_dtMatches_by_score(score_thresh, img['dtScores'], dtMatches)
        # gtMatches = filter_gtMatches_by_score(iclr, img['dtIds'], gtMatches)

        dtMatches, iclr = filter_dtMatches_by_score(score_thresh, img['dtScores'], dtMatches)
        gtMatches = filter_gtMatches_by_score(iclr, img['dtIds'], gtMatches)

        aRngidx = p.areaRng.index(img['aRng'])
        evalFilt[category][img['image_id']][p.areaRngLbl[aRngidx]] = {'dtMatches': dtMatches,'gtMatches': gtMatches}
        #pdb.set_trace()

    #determine performance metrics on filtered results
    metrics = []
    for category in evalFilt:

        TP = np.zeros(shape=(10,),dtype=int)
        TP2= np.zeros(shape=(10,),dtype=int)
        FP = np.zeros(shape=(10,),dtype=int)
        FN = np.zeros(shape=(10,),dtype=int)

        for img in evalFilt[category]:
            for rng in evalFilt[category][img]:
                data_elm = evalFilt[category][img][rng]
                if rng == p.areaRngLbl[0]:  #all size objects
                    TP = TP + np.sum(data_elm['dtMatches'] > 0.0, axis=1)
                    FP = FP + np.sum(np.absolute(data_elm['dtMatches']) < 0.1, axis=1)  # 0.0 indicates no match was made to a ground truth (-1.0 indicates ignore, other positive value is idx of gt match)
                    TP2= TP2+ np.sum(data_elm['gtMatches'] > 0.0, axis=1)
                    FN = FN + np.sum(np.absolute(data_elm['gtMatches']) < 0.1, axis=1)
        #print("TP: {}, TP2: {}, FP: {} , FN: {}".format(TP, TP2, FP, FN))

        #computer precision and recall for IoU = 0.5
        pr = TP[0]/(TP[0] + FP[0])
        rc = TP[0]/(TP[0] + FN[0])

        metrics.append({'category': category, 'TP': TP[0], 'FP': FP[0], 'FN': FN[0], 'PR': pr, 'RC': rc})

    return metrics

@torch.no_grad()
def evaluate(model, data_loader, logfile, device, writer=None, epoch=0):
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

    score_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    area_thresh = 500 #currently just allow a single area threshold in pixels^2
    for thresh in score_thresholds:
        res_filt = filter_results(coco_evaluator, thresh, area_thresh)

        for cat_data in res_filt:
            print_str = "Thresh: {:4.3f}, category {}, TP: {}, FP: {} , FN: {}, PR: {:4.3f}, RC: {:4.3f}".format(thresh, cat_data['category'], \
                                                cat_data['TP'], cat_data['FP'], cat_data['FN'], cat_data['PR'], cat_data['RC'])
            print(print_str)
            logfile.write(print_str + '\n')

            if writer:
                writer.add_scalars('{}_precision'.format(cat_data['category']), {str(thresh): cat_data['PR']}, epoch)
                writer.add_scalars('{}_recall'.format(cat_data['category']), {str(thresh): cat_data['RC']}, epoch)

    torch.set_num_threads(n_threads)
    return coco_evaluator
