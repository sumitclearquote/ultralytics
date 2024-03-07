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
    prefix = colorstr("albumentations: ")
    try:
        import albumentations as A

        check_version(A.__version__, "1.0.3", hard=True)  # version requirement

        # Transforms
        T = [
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.0),
            A.RandomGamma(p=0.0),
            A.ImageCompression(quality_lower=75, p=0.0),
        ]
        
        #Add custom augmentation here
        T += [A.HorizontalFlip(p=0.2),
                A.VerticalFlip(p=0.15),
                A.RandomCrop(height=480, width=480, p=0.15),
                A.Rotate(limit= 180, p =0.1)
                ]
        
        self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
    except ImportError:  # package not installed, skip
        pass
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")


# Replace the constructor
Albumentations.__init__ = __init__



#YOLOv8s
# Add other HPs here
model_file = "yolov8s.yaml"

train_version = "v1_s"

#Load a Model
model = YOLO(model_file)

#project_path = '/home/paintfrmladmin01/datadrive/LPBlur/ultralytics/my_runs/lpblur/'
project_path = '/home/paintfrmladmin01/datadrive/LPBlur/runs'


config  ={  'data': "/home/paintfrmladmin01/datadrive/LPBlur/datasets/LP_yolo_dataset/lp_data.yaml", 
            'epochs': 100,
            'batch': 320,
            'imgsz':320,
            'device':device,
            'patience':10,
            'project':project_path,
            'name':train_version,
            'close_mosaic': 5,
            'mosaic':0.5,
            'augment':True
        }

# Train the Model -> yolov8s
results = model.train(**config)



