#from ultralytics.ultralytics.data.augment import Albumentations
#from ultralytics.ultralytics.utils import LOGGER, colorstr
#from ultralytics.ultralytics.utils.checks import check_version
import sys
from ultralytics import YOLO


import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#YOLOv8s
# Add other HPs here
model_file = "yolov8s.yaml"

train_version = "v1_s"

#Load a Model
model = YOLO(model_file)

#project_path = '/home/paintfrmladmin01/datadrive/LPBlur/ultralytics/my_runs/lpblur/'
project_path = '/home/paintfrmladmin01/datadrive/LPBlur/runs'


config  ={  'data': "LP_yolo_dataset/lp_data.yaml", 
            'epochs': 100,
            'batch': 320,
            'imgsz':320,
            'device':device,
            'patience':10,
            'project':project_path,
            'name':train_version,
            'close_mosaic': 5,
            'mosaic':0.5,
        }

# Train the Model -> yolov8s
results = model.train(**config)



