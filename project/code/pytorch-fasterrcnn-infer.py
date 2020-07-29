import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

from wheat.wheat_test_dataset import WheatTestDataset
from wheat.albumentations_utils import get_train_transform, get_valid_transform, get_test_transform
from wheat.averager import Averager
from wheat.utils import collate_fn, format_prediction_string

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

DIR_INPUT = '/root/global-wheat-detection'
DIR_TRAIN = "{0}/train".format(DIR_INPUT)
DIR_TEST = "{0}/test".format(DIR_INPUT)

WEIGHTS_FILE = "{0}/models/fasterrcnn_resnet50_fpn.pth".format(DIR_INPUT)

test_df = pd.read_csv("{0}/sample_submission.csv".format(DIR_INPUT))

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the trained weights
model.load_state_dict(torch.load(WEIGHTS_FILE))
model.eval()

x = model.to(device)


test_dataset = WheatTestDataset(test_df, DIR_TEST, get_test_transform())

test_data_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    drop_last=False,
    collate_fn=collate_fn
)

detection_threshold = 0.5
results = []

for images, image_ids in test_data_loader:
    images = list(image.to(device) for image in images)
    outputs = model(images)
    for i, image in enumerate(images):
        boxes = outputs[i]['boxes'].data.cpu().numpy()
        scores = outputs[i]['scores'].data.cpu().numpy()
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]
        image_id = image_ids[i]
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        result = { 'image_id': image_id, 'PredictionString': format_prediction_string(boxes, scores)}
        results.append(result)

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.to_csv('new_submission.csv', index=False)
