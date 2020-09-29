import cv2
model_path='./weights/cedula_full_v2.pt'
import torch
from demo.document_cropper_v2 import document_cropper
import time
import glob

"""def draw_rects(image=None,ttensor=None,th=0.7):
    #ttensor_=ttensor.type(torch.int)
    for ttensor_ in ttensor:
        for data in ttensor_:
            xmin,ymin,xmax,ymax,score,class_id=data
            xmin,ymin,xmax,ymax=np.array([xmin,ymin,xmax,ymax]).astype(np.int)
            if(score>th):
                cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,0),3)
    return image"""






if __name__ == '__main__':
    dcp=document_cropper(path_m='./best_data.pt')
    with torch.no_grad():
        st = time.time()
        for image_ in glob.glob('/home/rony-warden/Escritorio/projects/warden_rebranding/cedulas_frente/test_images/*.jpg'):
            image=cv2.imread(image_)
            st = time.time()
            faces, rects, classes, ima, rois, maskc=dcp.get_parts_cropped(image,[1,2,3,4,5,10,11,12],[10,12,5])
            print(rects)
            cv2.imshow('f',cv2.resize(255*maskc,(700,700)))
            cv2.waitKey(1)
            #gh=transform_inv(rects,classes,image)
            #cv2.imshow('mask',gh)
            #cv2.imshow('maska', cv2.resize(image,(700,700)))
            #cv2.waitKey(10000)

            print("Total inference time {}".format(time.time()-st))