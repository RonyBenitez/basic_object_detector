import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.general import *
import torch
import numpy as np
import cv2
from pathlib import Path

p = Path(__file__).parents[2]


class document_cropper():
    model_p = './weights/cedulas_full_v2.pt'
    min_threshold = 0.7
    model = []
    device = []
    axis0=700

    def __init__(self,min_score=min_threshold,path_m=model_p,ax0=700,device_type='cpu'):
        self.device=select_device(device_type)
        self.model_p=path_m
        self.model=torch.load(self.model_p, map_location=self.device)['model'].float()
        self.axis0=ax0
        self.min_threshold=min_score
        self.model.to(self.device).eval()



    def get_parts_cropped(self,image,ids,not_idx,min_IOU=0.5):
        frame = image.copy()[:,:,::-1]
        faces, rects, classes, ima, rois, maskc = self._get_parts_cropped_(frame, self.min_threshold,ids,not_idx,min_IOU)
        return faces, rects, classes, ima, rois, maskc






    def _get_parts_cropped_(self,image, MIN_CON,idss,not_idss=[1],min_IOU=0.5):
        factor=image.shape[0]/self.axis0
        aspect_r=image.shape[1]/image.shape[0]
        image_r=cv2.resize(image,(int(self.axis0),int(self.axis0)),interpolation=cv2.INTER_CUBIC)
        half = self.device.type != 'cpu'
        if half: self.model.half()
        img_base = torch.zeros((1, 3, self.axis0, self.axis0), device=self.device)
        _ = self.model(img_base.half() if half else img_base) if self.device.type != 'cpu' else None
        img_base = np.moveaxis(image_r, [0, 1, 2], [1, 2, 0])
        img_base = torch.from_numpy(img_base).to(self.device)
        img_base = img_base.half() if half else img_base.float()
        img_base /= 255.0
        if img_base.ndimension() == 3:
            img_base = img_base.unsqueeze(0)
        pred = self.model(img_base, augment=False)[0]
        pred = non_max_suppression(pred, 0.4, min_IOU, agnostic=False)
        rects = []
        classes = []
        faces_en=[]
        imo = image.copy()
        cer = np.zeros_like(imo)
        for ttensor_ in pred:
            for data in ttensor_:
                xmin, ymin, xmax, ymax, score, class_id = data
                xmin, ymin, xmax, ymax = np.array([factor*aspect_r*xmin, factor*ymin, factor*aspect_r*xmax, factor*ymax]).astype(np.int)
                class_id=int(class_id)
                if (score >= MIN_CON):
                    if(class_id in idss):
                        rects.append([xmin, ymin, xmax, ymax])
                        Roi=imo[int(ymin):int(ymax), int(xmin):int(xmax)]
                        faces_en.append(Roi.copy())
                        classes.append(class_id)
                        color=(255,255,255)
                        if (not (class_id in not_idss)):
                            cv2.rectangle(cer, (xmin, ymin), (xmax, ymax), color, -1)
        return imo, rects, classes, imo, faces_en, (cer[:, :, 0] / 255).astype(np.uint8)
