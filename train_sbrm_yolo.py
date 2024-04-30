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
            A.RandomBrightnessContrast(brightness_limit=(-0.3,-0.1),contrast_limit=(-0.1, 0.1), p = 0.2), #og is 0.05, no other arguments
            A.RandomGamma(p=0.0),
            A.ImageCompression(quality_lower=75, p=0.02),
        ]
        #'''
        #Add custom augmentation here
        T += [  A.HorizontalFlip(p=0.7),
                A.VerticalFlip(p=0.2),
                A.RandomSizedBBoxSafeCrop(height= 800, width=800, erosion_rate=0.3, p = 0.3),
                #A.Affine(scale=(0.9, 1.8), shear=(-20, 20), rotate=(-180,180),  p = 0.2),
                #A.Perspective(p = 0.1),
                #A.ChannelShuffle(p = 0.15),
                #A.ColorJitter(p = 0.05),
                #A.Downscale(scale_min=0.20, scale_max=0.3, p = 0.1),
                #A.MotionBlur(blur_limit = 13, p = 0.2)
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
project_name = "sbrm"

#name of the dataset folder
dataset_name = "sbrm_yolo_dataset"
yolo_cfg = "sbrm_data.yaml" #name of the yolo cfg yaml file inside dataset
train_versions = ["v2_l"] #["v2_s", "v2_m", "v2_l", "v2_x"]


for train_version in train_versions:
    
    if train_version.endswith("n") and "p" not in train_version:
        # Add other HPs here
        model_file = "yolov8n.yaml"
        lr = 0.001
    if train_version.endswith("n") and 'p' in train_version:
        # Add other HPs here
        model_file = "yolov8n-p2.yaml"
        lr = 0.0001
    elif train_version.endswith("s"):
        lr = 0.001
        model_file = "yolov8s.yaml"
        bsize = 32
    elif train_version.endswith("m"):
        model_file = "yolov8m.yaml"
        lr = 0.001
        bsize = 24
    elif train_version.endswith("l"):
        model_file = "yolov8l.yaml"
        lr = 0.001
        bsize = 16
    elif train_version.endswith("x"):
        model_file = "yolov8x.yaml"
        bsize = 8 #batch_size
        lr = 0.001

    print(f"Training {model_file.split('.')[0]} ...\n")

    #Load a Model
    model = YOLO(model_file)

    project_path = f'/home/paintfrmladmin01/datadrive/ssqs/yolo_runs/{project_name}'



    config  ={  'data': f"/home/paintfrmladmin01/datadrive/ssqs/datasets/{dataset_name}/{yolo_cfg}", 
                'epochs': 120,
                'lr0':lr, #default is 1e-3
                'batch': bsize,
                'imgsz':800,
                'device':device,
                'patience':50,
                'project':project_path,
                'name':train_version,
                'close_mosaic': 5,
                'mosaic':0.3,
            }

    # Train the Model -> yolov8s
    results = model.train(**config)
    
    
    
#====================================== THE END ======================================