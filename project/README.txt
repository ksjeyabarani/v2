Presentation: https://docs.google.com/presentation/d/1crP0JSKXDCP9dK7Wn3RCbLKor1wlv4zDUQKtUmJgNxg/edit?usp=sharing
White Paper: https://docs.google.com/document/d/19w7EXLD5FOPENaDHvXu_WBm8rTEf7ehLiUdW5Hs7qVQ/edit?usp=sharing

Ran the training and validation in v100 ubuntu boxes. The training was done with 60 epochs. Hyper-parameters used are defaults as in the code.

* Setup:
   Run on python3
** Code setup
   1. cd && git clone https://github.com/ksjeyabarani/v2.git
   2. cd v2 && git checkout w251_prj
   3. pip3 install virtualenv
   4. virtualenv venv
   5. source venv/bin/activate
   6. pip install -r project/requirements.txt

** Kaggle dataset
    1. cd && kaggle competitions download -c global-wheat-detection
    2. Alternatively download data from https://www.kaggle.com/c/global-wheat-detection/data
    3. cd && tar -zxvf global-wheat-detection.zip


### Models are not checked-in.  Please skip this step
 To run validation on checked-in models
    1. mkdir -p /root/global-wheat-detection/models/
    2. cp models/fasterrcnn_resnet50_fpn_epoch_60.pth /root/global-wheat-detection/models/fasterrcnn_resnet50_fpn_epoch_60.pth
    3. cd code/base && python  pytorch-fasterrcnn-validate.py
#######

### Baseline Models
  Here we use the pretrained resnet50 model and finetune with the train dataset. No augmentations is performed here

* To train for baseline:
    cd code/base &&  python pytorch-fasterrcnn-train.py <num_epochs> <learning_rate> <momentum> <weight_decay>
   A model file qualified by above hyper parameters will be stored in model/ directory with name : fasterrcnn_resnet50_fpn_base_epoch_<num_epochs>_<learning_rate>_<momentum>_<decay>.pth

* To run validation for baseline:
    cd code && python  pytorch-fasterrcnn-validate.py  <train_num_epochs> <train_learning_rate> <train_momentum> <train_weight_decay>
  The training parameters are needed to be provided to this script inorder to identify the corresponding model and run evaluation on them.

### Augmented Models
  Here we use the pretrained resnet50 model or the models from previous step and further finetune with augmented train dataset.

* To train for baseline:
    cd code/aug &&  python pytorch-fasterrcnn-train.py <base_model_loc> <num_epochs> <learning_rate> <momentum> <weight_decay>
   A model file qualified by above hyper parameters will be stored in model/directory with name : fasterrcnn_resnet50_fpn_aug_epoch_<num_epochs>_<learning_rate>_<momentum>_<decay>.pth

* To run validation for augmented train model:
    cd code && python  pytorch-fasterrcnn-validate.py  <train_num_epochs> <train_learning_rate> <train_momentum> <train_weight_decay>
  The training parameters are needed to be provided to this script inorder to identify the corresponding model and run evaluation on them.

