'''
Run instruction: -> python test.py iounmsthresh confthresh
'''
from ultralytics import YOLO
import sys
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


model_type = "v1_s"
data = "spinny" #[val, 'spinny]



model_path = f"my_runs/lpblur/{model_type}/weights/best.pt"
model = YOLO(model_path)

conf = float(sys.argv[2])
iou_nms_thresh = float(sys.argv[1]) #default=0.7
batch_size = 160 if model_type == "v1_s" else 192 if model_type=="v1_n" else 1

if data=='val' #Use validation set
        data_path =  "/home/paintfrmladmin01/datadrive/LPBlur/datasets/LP_yolo_dataset/lp_data.yaml"
elif data == "spinny":
        data_path = "/home/paintfrmladmin01/datadrive/LPBlur/datasets/spinnydata1_yolo_dataset/lp_data.yaml"

config = {'data': data_path,  
          'imgsz' : 480,
          'batch' : batch_size,
         'conf' : conf, 
        'iou' : iou_nms_thresh,  
        'device' : device, 
        'save':True,
        'save_conf' : True, 
        'save_json' : True, 
        'save_txt' : True,
        'project' : f"my_runs/lpblur/{model_type}/val", 
        'name' : f"spinny_val_analysis_{iou_nms_thresh}_{conf}"

        }


print(f"Generating Predictions for NMS:{iou_nms_thresh} and Confidence: {conf}")

res = model.val(**config)
