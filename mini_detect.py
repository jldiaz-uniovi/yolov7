from pathlib import Path
import random
from collections import Counter
import torch
import cv2
import numpy as np
from dataclasses import dataclass
from models.experimental import attempt_load
from utils.torch_utils import select_device, time_synchronized
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox

# For compatibility with yolov5 mini_detect
@dataclass
class DetectResults:
    time:float
    detect_time:float
    augment:bool
    imgs:list
    files:list
    xyxy:list
    names:list[str]

def load_model(model_name="yolov7", device=""):
    """Load the model
    model_name: one of "yolov7", "yolov7x"
    """
    device = select_device(device)
    model_name += ".pt"
    model = attempt_load(model_name, map_location=device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()  # to FP16
    return model

def detect(image, model, device="", augment=False, agnostic_nms=False,
           conf_thres = 0.25, iou_thres = 0.45, classes = None):
    """Applies the model on the image and returns the object with the results"""
    # Reescalar imagen
    img0 = image
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    stride = int(model.stride.max())
    imgsz = check_img_size(640, s=stride)
    img = letterbox(img0, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)    
    # Detección

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]
    t2 = time_synchronized()
    # Extraer las etiquetas más probables para los objetos detectados
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t3 = time_synchronized()

    names = model.module.names if hasattr(model, 'module') else model.names
    det = pred[0]
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
    t4 = time_synchronized()

    return DetectResults(augment=augment, detect_time=t2-t1, time=t4-t1,
            imgs=[image], files=["noname"], names=names, xyxy=[reversed(det)])

def get_meta(results:DetectResults):
    """Extracts relevant information from the results of the detection
    and generates a Python dictionary with it.

    results:
        The result object returned by detect

    returns:
        Python dictionary
    """

    meta = {
        "meta": {
            "file": results.files[0],
            "time": results.time,
            "augment": results.augment,
            "imagesize": "{}x{}".format(*results.imgs[0].shape[:-1][::-1]),
        },
        "results": [],
    }
    res = []
    for *xyxy, conf, clase in results.xyxy[0]:
        label = results.names[int(clase)]
        dic = {
            "bounding_box": [int(x) for x in xyxy],
            "confidence": conf.item(),
            "label": label
        }
        meta["results"].append(dic)
        res.append(label)
    meta["_summary"] = dict(Counter(res))
    return meta

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """Adds one bounding box with optional label to the image"""

    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def label_image(p, results, save_dir=None):
    """Takes the results of the detection and draws several bounding boxes 
    with labels on the image.

    p: 
        relative path and filename of the source image
    results: 
        the object returned by detect
    save_dir: 
        name of the folder to save the labelled image
    """

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in results.names]

    if save_dir:
        im0 = cv2.imread(str(p))
    else:
        im0 = p
    # im0 = results.imgs[0]
    for *xyxy, conf, clase in reversed(results.xyxy[0]):
        label = f"{results.names[int(clase)]} {conf:.2f}"
        plot_one_box(
            xyxy,
            im0,
            label=label,
            color=colors[int(clase)],
            line_thickness=1,
        )

    # Save results (image with detections)
    if save_dir:
        p = Path(p)
        save_dir = Path(save_dir)
        save_path = str(save_dir / p.name) 
        cv2.imwrite(save_path, im0)
    else:
        ok, buffer = cv2.imencode(".jpg", im0)
        if ok:
            return buffer

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("Usage:")
        print(f"{sys.argv[0]} filename")
        quit()    
    path = sys.argv[1]
    # Load the model
    model = load_model()
    # Load the image
    img = cv2.imread(path)
    # perform detection
    results = detect(img, model)

    # present results, first as json
    meta = get_meta(results)
    print(meta)

    # Then as a labelled image
    img2 = label_image(img, results)
    with open("result.jpg", "wb") as f:
        f.write(img2)