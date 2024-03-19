import numpy as np
import sys
import os
import torch
sys.path.append("..")
from utils.data_utils import *
from ultralytics import YOLO
from tqdm import tqdm
from collections import Counter, defaultdict

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def generate_predictions(imgdir, model, dest_dir, conf_threshold = None, iou_nms_thresh=None, save_results = False):
    results_dict = defaultdict(lambda: defaultdict(list))
  
    for imgname in tqdm(os.listdir(imgdir)):
        if imgname.endswith("json"):continue
        #if imgname != "":continue
        imgpath = f"{imgdir}/{imgname}"
        
        img = np.array(Image.open(imgpath))
        
        imgw, imgw, _ = img.shape
        
        results = model(imgpath, imgsz=480, iou=iou_nms_thresh, device=device, verbose = False)

        result = results[0]
        
        #If no detections found
        if len(result.boxes.cpu().numpy()) == 0:
            results_dict[imgname]['bbox'] = []
            results_dict[imgname]['scores'] = []
            continue
        
        #Loop through the results
        #for result in results:
        
        boxes = result.boxes.cpu().numpy()
        
        for i, box in enumerate(boxes):
            conf = box.conf[0]
            
            if conf < conf_threshold:# If conf < threshold
                results_dict[imgname]['bbox'] = []
                results_dict[imgname]['scores'] = []
                continue
                
            bbox = box.xyxy[0].astype(int)
            results_dict[imgname]['bbox'].append(bbox.tolist())
            results_dict[imgname]['scores'].append(float(f"{conf:.4f}"))
            

    
    if save_results:
        dump_json(results_dict, f"{dest_dir}/prediction_results.json", indent = 1)
    return results_dict





if __name__ == '__main__':
    iterations = ["v3_m"]
    save_results = True
    dtype = "quotes200" #[val, spinny2, "audit", "quotes200"]
    conf_thresholds = [0.1,0.2,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8]
    iou_nms_thresh = 0.7


    names = {0: 'licenseplate'}
    
    
    for iteration in iterations:
        model = YOLO(f"my_runs/lpblur/{iteration}/weights/best.pt").to(device) #Load Model
        print("Model is on: ", model.device)
        for conf_threshold in conf_thresholds:
            if dtype == "val":
                imgdir = "../datasets/LP_yolo_dataset/val/images"
                dest_dir = f"my_runs/lpblur/{iteration}/val/val_analysis_{str(iou_nms_thresh)}_{str(conf_threshold)}"
            elif dtype == "spinny2":
                imgdir = "../datasets/spinnydata2_yolo_dataset/val/images"
                dest_dir = f"my_runs/lpblur/{iteration}/val/spinny2_analysis_{str(iou_nms_thresh)}_{str(conf_threshold)}"
            elif dtype == "quotes200": #200quotes
                imgdir = f"../datasets/{dtype}"
                dest_dir = f"my_runs/lpblur/{iteration}/val/{dtype}_analysis_{str(iou_nms_thresh)}_{str(conf_threshold)}"
            elif dtype== "audit":
                imgdir = "../datasets/audit_data"
                dest_dir = f"my_runs/lpblur/{iteration}/val/audit_analysis_{str(iou_nms_thresh)}_{str(conf_threshold)}"
            os.makedirs(dest_dir, exist_ok=True)
            
            print(f"Making Predictions using model {iteration} on {dtype} with NMS_THRESH {iou_nms_thresh} and conf_thresh {conf_threshold}")
            results_dict = generate_predictions(imgdir, model,dest_dir, conf_threshold = conf_threshold, iou_nms_thresh=iou_nms_thresh, save_results = save_results)
        
            print(f"Total Images predicted on:  {len(results_dict)} for confidence: {conf_threshold}")