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
    3. cd code && python  pytorch-fasterrcnn-validate.py
#######


* To train:
    cd code &&  python pytorch-fasterrcnn-train.py

* To run validation:
    cd code && python  pytorch-fasterrcnn-validate.py
