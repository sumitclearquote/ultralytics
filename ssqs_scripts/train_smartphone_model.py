'''Script to train smartphone model'''
from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_version
import sys
from ultralytics import YOLO

import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def __init__(self, p=1.0):
    """Initialize the transform object for YOLO bbox formatted params."""
    self.p = p
    self.transform = None
    prefix = colorstr("\n\nalbumentations: ")
    try:
        import albumentations as A

        check_version(A.__version__, "1.0.3", hard=True)  # version requirement

        # Transforms
        T = [
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.0),
            A.ImageCompression(quality_lower=75, p=0.02),
        ]
        #'''
        #Add custom augmentation here
        T +=                           [A.RGBShift(r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10), p = 0.25),
                                        #A.Rotate(limit= 45, p =0.5),
                                        A.GaussianBlur(p = 0.25),
                                        #A.Perspective(p=0.3)
                                        ]
        #'''
        
        self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
    except ImportError:  # package not installed, skip
        pass
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")


# Replace the constructor of the repo with the above one. Use this constructor to add whatever custom augmentations is needed. 
Albumentations.__init__ = __init__


#==============================================================================================================

#'''
#YOLOv8s --->============================================================================================================


# A directory inside "yolo_runs" will be created with the below name
project_name = "final_smartphone"

#name of the dataset folder
dataset_name = "final_smartphone_yolo_dataset" #wheelrim and lifting pads were expanded by 5% of bbox area
yolo_cfg = "final_smartphone_data.yaml" #name of the yolo cfg yaml file inside dataset
train_versions = ["v3_n"]
imgsizes = [320]

for train_version in train_versions:
    for imgsize in imgsizes:
        if train_version.endswith("n") and "p" not in train_version:
            # Add other HPs here
            model_file = "yolov8n.yaml"
            lr = 0.001
            if imgsize == 320:
                bsize = 512
            elif imgsize == 640:
                bsize = 128
        if train_version.endswith("n") and 'p' in train_version:
            # Add other HPs here
            model_file = "yolov8n-p2.yaml"
            lr = 0.0001
        elif train_version.endswith("s") and "p" not in train_version:
            model_file = "yolov8s.yaml"
            lr = 0.001
            if imgsize == 320:
                bsize = 256
            elif imgsize == 640:
                bsize = 64

        print(f"Training {model_file.split('.')[0]} ...\n")

        #Load a Model
        model = YOLO(model_file)

        project_path = f'/home/paintfrmladmin01/datadrive/ssqs/yolo_runs/{project_name}'



        config  ={  'data': f"/home/paintfrmladmin01/datadrive/ssqs/datasets/{dataset_name}/{yolo_cfg}", 
                    'epochs': 40,
                    'lr0':lr, #default is 1e-3
                    'batch': bsize,
                    'imgsz':imgsize,
                    'device':device,
                    'patience':10,
                    'project':project_path,
                    'name':train_version,
                    'close_mosaic': 0,
                    'mosaic':0.0,
                }

        # Train the Model -> yolov8s
        results = model.train(**config)
#'''