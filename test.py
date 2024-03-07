from ultralytics import YOLO
import sys
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


model_type = ""

model_path = f"ultralytics/my_runs/lpblur/{model_type}/weights/best.pt"
model = YOLO(model_path)

conf = float(sys.argv[2])
iou_nms_thresh = float(sys.argv[1]) #default=0.7
batch_size = 320


config = {'data': "LP_yolo_dataset/lp_data.yaml",  
          'imgsz' : 480,
          'batch' : batch_size,
         'conf' : conf, 
        'iou' : iou_nms_thresh,  
        'device' : device, 
        'save_conf' : True, 
        'save_json' : True, 
        'save_txt' : True,
        'project' : f"ultralytics/my_runs/lpblur/{model_type}/val", 
        'name' : f"val_analysis_{conf}_{iou_nms_thresh}"

        }


print(f"Generating Predictions for Confidence: {conf}")

res = model.val(**config)
