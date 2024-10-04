'''Script to train fixedcam1 model'''
from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_version
import sys
from ultralytics import YOLO

import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# =============================================================================================================================
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
        #Add custom augmentation here ====
        T +=                           [A.RGBShift(r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10), p = 0.25),
                                        A.Rotate(limit= (-15, 15), p =0.3),
                                        A.GaussianBlur(p = 0.25),
                                        A.Perspective(p=0.3)
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

# ==============================================================================================================================================

def get_config(model_name, imgsize):
    ''' Returns the model cfg file, lr and batch size based on the model name and imgsize used
    '''
    if model_name.endswith("n"):
        model_cfg_file = "yolov8n.yaml"
        lr = 0.001 #0.001
        if imgsize == 224:
            bsize = 512
        elif imgsize == 320:
            bsize = 256
        elif imgsize == 416:
            bsize = 128
        elif imgsize == 640:
            bsize = 96
            
        if "p" in model_name: #v1_n_p model
            model_cfg_file = "yolov8n-p2.yaml"
            lr = 0.0001
            if imgsize == 224:
                bsize = 256
            elif imgsize == 320:
                bsize = 128
            elif imgsize == 416:
                bsize = 64
            elif imgsize == 640:
                bsize = 48
            
            
    elif model_name.endswith("s"): #v1_s model
        model_cfg_file = "yolov8s.yaml"
        
        lr = 0.0001 #0.001
        if imgsize == 224:
            bsize = 128
        elif imgsize == 320:
            bsize = 128
        elif imgsize == 480:
            bsize = 128
        elif imgsize == 640:
            bsize = 64
            
        if "p" in model_name: #v1_s_p model
            model_cfg_file = "yolov8s-p2.yaml"
            lr = 0.00001
            if imgsize == 480:
                bsize = 60
            elif imgsize == 640:
                bsize = 32

    return model_cfg_file, lr, bsize




#==============================================================================================================

#'''
#YOLOv8s --->============================================================================================================


# A directory inside "yolo_runs" will be created with the below name
project_name = "lpscan" # the dir where datasets' folder is present and where results in yolo_runs will be stored.
project_dir = "lpscan" # # Name of project inside 'datadrive'
server_name = "paintfrmladmin01" # username of the remote machine

#name of the dataset folder
dataset_name = "final_ocr_dataset_v1" #wheelrim and lifting pads were expanded by 5% of bbox area
yolo_cfg = "lp_ocr_data.yaml" #name of the yolo cfg yaml file inside dataset

# HYPERPARAMETERS
epochs = 350
patience = 100 # After how many epochs to stop training if results do not improve,.
train_versions = ['v1_s'] #["v1_n", "v1_p_n"]        
imgsizes = [480, 640]    #-original

for train_version in train_versions:
    for imgsize in imgsizes:
        if f"{train_version}_{imgsize}" not in ['v1_n_224', 'v1_n_320']:continue
        #if f"{train_version}_{imgsize}" in ["v1_p_n_224", "v1_p_n_320"]:continue
        
        model_file, lr, bsize = get_config(train_version, imgsize)

        print(f"\nTraining {model_file.split('.')[0]} ({train_version}) with lr {lr} , batch_size {bsize} and imgsize {imgsize} ({epochs} epochs)... ========================================================== >\n")

        #Load a Model
        model = YOLO(model_file)

        project_path = f'/home/{server_name}/datadrive/{project_dir}/yolo_runs/{project_name}'

 

        config  ={  'data': f"/home/{server_name}/datadrive/{project_dir}/datasets/{dataset_name}/{yolo_cfg}", 
                    'epochs': epochs,
                    'lr0':lr, #default is 1e-3
                    'batch': bsize,
                    'imgsz':imgsize,
                    'device':device,
                    'patience':patience,
                    'project':project_path,
                    'name':f"{train_version}_{imgsize}",
                    'close_mosaic': 5,
                    'mosaic':0.3,
                    'workers':8
                }

        # Train the Model -> yolov8s
        results = model.train(**config)