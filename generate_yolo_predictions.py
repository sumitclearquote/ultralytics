import numpy as np
import sys
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
    iteration = "v1_s"
    save_results = True
    conf_threshold = 0.001
    iou_nms_thresh = 0.7


    dest_dir = f"my_runs/lpblur/{iteration}/val/val_analysis_{str(iou_nms_thresh)}_{str(conf_threshold)}"
    os.makedirs(dest_dir, exist_ok=True)

    names = {0: 'licenseplate'}
    #Load Model
    model = YOLO(f"my_runs/lpblur/{iteration}/weights/best.pt")

    imgdir = "../datasets/LP_yolo_dataset/val/images"


    results_dict = generate_predictions(imgdir, model,dest_dir, conf_threshold = conf_threshold, iou_nms_thresh=iou_nms_thresh, save_results = save_results)
    
    print("Total Images predicted on: ", len(results_dict))