'''
Run instruction: -> python test.py <iounmsthresh> <confthresh>
'''
from ultralytics import YOLO
import sys
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

dataset_name = "jack_yolo_dataset"
project_name = "jack_detection"
model_version = "v1_n"
yolo_cfg = "jack_data.yaml"
data = "val" #[val, spinny2] # On what data to test
project_dir = "mahindra"



model_path = f"/home/paintfrmladmin01/datadrive/{project_dir}/yolo_runs/{project_name}/{model_version}/weights/best.pt"
print("Loading model from: ", model_path)
model = YOLO(model_path)

iou_nms_thresh = float(sys.argv[1]) #default=0.7
conf = float(sys.argv[2]) #default = 0.001


batch_size = 256

config = {'data': f"/home/paintfrmladmin01/datadrive/{project_dir}/{project_name}/datasets/{dataset_name}/{yolo_cfg}",  
          'imgsz' : 224,
          'batch' : batch_size,
         'conf' : conf, 
        'iou' : iou_nms_thresh,  
        'device' : device, 
        'save':True,
        'save_conf' : True, 
        'save_json' : True, 
        'save_txt' : True,
        'project' : f'/home/paintfrmladmin01/datadrive/{project_dir}/yolo_runs/{project_name}/{model_version}/{data}', 
        'name' : f"{data}_{iou_nms_thresh}_{conf}"

        }


print(f"Generating Predictions for NMS:{iou_nms_thresh} and Confidence: {conf}")

res = model.val(**config)