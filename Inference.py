import argparse

import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.general import  *
import torch
import numpy as np


def load(weights='./weights/cedulas_full_v2.pt'):
    device = select_device('cpu')
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    return model,device



def get_parts_cropped(model, device, imgsz=700, conf_th=0.4, iou_th=0.5, imagen=None,th_second=0.7):
    #Se asume que las imagenes ya vienen en 700x700
    half = device.type != 'cpu'
    if half:model.half()
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    img=np.array(imagen[:,:,::-1])
    img=np.moveaxis(img,[0,1,2],[1,2,0])
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_th, iou_th, agnostic=False)
    rects=[]
    classes=[]
    for ttensor_ in pred:
        for data in ttensor_:
            xmin, ymin, xmax, ymax, score, class_id = data
            xmin, ymin, xmax, ymax = np.array([xmin, ymin, xmax, ymax]).astype(np.int)
            if (score >= th_second):
                rects.append([xmin,ymin,xmax,ymax])
                classes.append(int(class_id))
    return rects,classes


