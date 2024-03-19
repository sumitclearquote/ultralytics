'''
Run instruction: -> python test.py <iounmsthresh> <confthresh>
'''
from ultralytics import YOLO
import sys
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


model_type = "v4_m"
data = "val" #[val, spinny2]



model_path = f"my_runs/lpblur/{model_type}/weights/best.pt"
print("Loading model from: ", model_path)
model = YOLO(model_path)

conf = float(sys.argv[2]) #default = 0.001
iou_nms_thresh = float(sys.argv[1]) #default=0.7
batch_size = 160 if model_type.endswith("s") else 192 if model_type.endswith("n") else 74 if model_type.endswith("m") else 1

if data=='val': #Use validation set
        data_path =  "/home/paintfrmladmin01/datadrive/LPBlur/datasets/LP_yolo_dataset/lp_data.yaml"
elif data == "spinny2":
        data_path = "/home/paintfrmladmin01/datadrive/LPBlur/datasets/spinnydata2_yolo_dataset/lp_data.yaml"

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
        'name' : f"{data}_analysis_{iou_nms_thresh}_{conf}"

        }


print(f"Generating Predictions for NMS:{iou_nms_thresh} and Confidence: {conf}")

res = model.val(**config)
