import pandas as pd
import numpy as np
import cv2
import os
import re
import sys
from PIL import Image

from wheat.wheat_dataset import WheatDataset
from wheat.albumentations_utils import get_valid_transform, get_train_transform
from wheat.validation_utils import calculate_image_precision
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

import argparse

def get_aug_train_transform():
    return A.Compose([
        A.OneOf([
#           A.OneOf([
#               A.RandomBrightnessContrast(p=1),    
#               A.RandomGamma(p=1),    
#               #A.CLAHE(p=1), 
#           ], p=1),
#           A.OneOf([
#               #A.CLAHE(p=1),
#               A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1),
#           ], p=1),
           A.OneOf([
               A.ChannelShuffle(p=1),
           ], p=1),
        ], p=1),
        A.OneOf([
           A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1),
           A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=1),
           A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=1),
           #A.RandomShadow(num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.5),
           A.RandomFog(fog_coef_lower=0.7, fog_coef_upper=0.8, alpha_coef=0.1, p=0.1)
        ], p=1),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

parser = argparse.ArgumentParser(description='Faster RCNN Trainer')
parser.add_argument('pretrained_model_path', default=None,  help='Pretrained Model Path')
parser.add_argument('epochs', default=None, type=int, help='Number of epochs')
parser.add_argument('lr', default=None, type=float, help='Learning Rate')
parser.add_argument('momentum', default=None, type=float, help='Momentum')
parser.add_argument('decay', default=None, type=float, help='Weight Decay')

args = parser.parse_args()
lr1 = float(args.lr)
num_epochs = int(args.epochs)
momentum1 = float(args.momentum)
decay1 = float(args.decay)

lrs = str(lr1).replace(".", "")
momentums = str(momentum1).replace(".", "")
decays = str(decay1).replace(".", "")
print("lr={0}, num_epochs={1}, momentum={2}, decay={3}".format(lr1, num_epochs, momentum1, decay1))

# Global Wheat Detection - 
DIR_INPUT = '/root/global-wheat-detection'
DIR_TRAIN = "{0}/train".format(DIR_INPUT)
DIR_TEST = "{0}/test".format(DIR_INPUT)

MODEL_DIR = "/root/w251/models"
WEIGHTS_FILE = "{0}/fasterrcnn_resnet50_fpn_aug_add_epoch_{1}_{2}_{3}_{4}.pth".format(MODEL_DIR, num_epochs, lrs, momentums, decays)
INPUT_WEIGHTS_FILE = args.pretrained_model_path

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
print("Num Image IDS : {0}".format(len(image_ids)))
valid_ids = image_ids[-665:]
print("Num Validation IDS : {0}".format(len(valid_ids)))
train_ids = image_ids[:-665]
print("Num Train IDS : {0}".format(len(train_ids)))
valid_df = train_df[train_df['image_id'].isin(valid_ids)]
train_df = train_df[train_df['image_id'].isin(train_ids)]

if os.path.exists(WEIGHTS_FILE):
  print("Output model file {0} already exists".format(WEIGHTS_FILE))
  sys.exit(-1)

# load a model; pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

if INPUT_WEIGHTS_FILE != "null":
  model.load_state_dict(torch.load(INPUT_WEIGHTS_FILE))

train_dataset = WheatDataset(train_df, DIR_TRAIN, get_aug_train_transform(), default_transform=get_train_transform(), append_transformed=True)
valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_valid_transform(), default_transform=get_valid_transform())

# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
print("lr={0}, num_epochs={1}, momentum={2}, decay={3}".format(lr1, num_epochs, momentum1, decay1))
optimizer = torch.optim.SGD(params, lr=lr1, momentum=momentum1, weight_decay=decay1)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lr_scheduler = None

loss_hist = Averager()

for epoch in range(num_epochs):
    loss_hist.reset()
    num_images = 0  
    itr = 0
    for images, targets, image_ids in train_data_loader:
        num_images = num_images + 1
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % 5 == 0:
            print("Epoch : #{0}, Num Images Processed: #{1} loss: {2}".format(epoch, str(num_images * 16), loss_value))
        itr += 1

    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()
    print("Epoch #{0} Training loss: {1}".format(epoch,loss_hist.value))   

torch.save(model.state_dict(), WEIGHTS_FILE)
