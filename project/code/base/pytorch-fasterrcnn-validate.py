import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

from wheat.wheat_dataset import WheatDataset
from wheat.albumentations_utils import get_train_transform, get_valid_transform
from wheat.validation_utils import calculate_image_precision
from wheat.metric_validation import Metric
from wheat.averager import Averager
from wheat.utils import collate_fn

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt

# Global Wheat Detection - 
DIR_INPUT = '/root/global-wheat-detection'
DIR_TRAIN = "{0}/train".format(DIR_INPUT)
DIR_TEST = "{0}/test".format(DIR_INPUT)

print("Printing shape of training dataset")
train_df = pd.read_csv("{0}/train.csv".format(DIR_INPUT))
train_df.shape

# Bounding Boxes
train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)
train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)

# Split to train and validation set
image_ids = train_df['image_id'].unique()
valid_ids = image_ids[-665:]
train_ids = image_ids[:-665]
valid_df = train_df[train_df['image_id'].isin(valid_ids)]
train_df = train_df[train_df['image_id'].isin(train_ids)]

valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_valid_transform())

WEIGHTS_FILE = "{0}/models/fasterrcnn_resnet50_fpn_epoch_60.pth".format(DIR_INPUT)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load(WEIGHTS_FILE))
model.eval()
x = model.to(device)
valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=8,
    collate_fn=collate_fn
)

validation_image_precisions = []
iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]
for images, targets, image_ids in valid_data_loader:
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    outputs = model(images)
    all_gt_boxes = list()
    for i, image in enumerate(targets):
        gt_boxes = targets[i]['boxes'].data.cpu().numpy()
        # coco format
        gt_boxes[:, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_boxes[:, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]
        all_gt_boxes.append(gt_boxes)

    all_boxes = list()
    for i, image in enumerate(images):
        boxes = outputs[i]['boxes'].data.cpu().numpy()
        scores = outputs[i]['scores'].data.cpu().numpy()
        # coco format
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        preds_sorted_idx = np.argsort(scores)[::-1]
        preds_sorted = boxes[preds_sorted_idx]
        all_boxes.append(preds_sorted)
    assert len(all_boxes) == len(all_gt_boxes)
    for idx in range(0, len(all_boxes)):
        preds_sorted = all_boxes[idx]
        gt_boxes = all_gt_boxes[idx]
        image_precision = calculate_image_precision(preds_sorted, gt_boxes, thresholds=iou_thresholds, form='coco')
        validation_image_precisions.append(image_precision)
        print("The average precision of the sample image: {0:.4f}".format(image_precision))

print("Validation IOU: {0:.4f}".format(np.mean(validation_image_precisions)))
