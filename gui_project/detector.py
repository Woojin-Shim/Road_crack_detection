import sys
from PIL import ImageTk, Image
sys.path.append('yolov5')
import torch
from models.common import DetectMultiBackend
from utils.general import check_img_size, cv2, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import numpy as np

weights='last.pt'  # model.pt path(s)  #########################
data='road.yaml'  # dataset.yaml path  #########################
imgsz=(640, 640)  # inference size (height, width)
conf_thres=0.5  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image
device='' # cuda device, i.e. 0 or 0,1,2,3 or cpu
view_img=False  # show results
save_txt=False  # save results to *.txt
save_conf=False  # save confidences in --save-txt labels
save_crop=False  # save cropped prediction boxes
nosave=False  # do not save images/videos
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS
augment=False  # augmented inference
visualize=False  # visualize features
update=False  # update all models
project='runs/detect'  # save results to project/name
name='exp'  # save results to project/name
exist_ok=False  # existing project/name ok, do not increment
line_thickness=2  # bounding box thickness (pixels)
hide_labels=False  # hide labels
hide_conf=False  # hide confidences
half=False  # use FP16 half-precision inference
dnn=False  # use OpenCV DNN for ONNX inference
device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size
model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

def detect(src):
    im = letterbox(src, imgsz, stride=stride, auto=True)[0]
    im = im[..., [2, 1, 0]].transpose(2, 0, 1)
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]
    pred = model(im, augment=augment, visualize=visualize)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    det = pred[0]
    if len(det):
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], src.shape).round()
        return det

def detect2(src,th):
    im = letterbox(src, imgsz, stride=stride, auto=True)[0]
    im = im[..., [2, 1, 0]].transpose(2, 0, 1)
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]
    pred = model(im, augment=augment, visualize=visualize)
    pred = non_max_suppression(pred, th, iou_thres, classes, agnostic_nms, max_det=max_det)
    det = pred[0]
    if len(det):
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], src.shape).round()
        return det

def draw_boxes(src, det):
    dst = src.copy()
    annotator = Annotator(dst, line_width=line_thickness, example=str(names))
    for *xyxy, conf, cls in reversed(det):
        c = int(cls)  # integer class
        annotator.box_label(xyxy, None, color=colors(c+2, True))
    return dst

def draw_boxes2(src, det):
    count_list = [0,0]
    dst = src.copy()
    color = [(255,0,0),(0,0,255),(255,255,255)]
    annotator = Annotator(dst, line_width=line_thickness, example=str(names))
    for *xyxy, conf, cls in reversed(det):
        c = int(cls)  # integer class
        annotator.box_label(xyxy, None, color=color[c])
        count_list[c] += 1

    return (dst,count_list[0],count_list[1])


def r_img(src,th=conf_thres,str='',set=None):
    p,c = 0,0
    det = detect2(src,th)
    if det is not None:
        dst,p,c= draw_boxes2(src, det)
    else:
        dst = src
    if set is not None:
        size,t1,t2 = set
        cv2.putText(dst,str,(8,20),cv2.FONT_HERSHEY_SIMPLEX,size,(0,0,0),t1,2)
        cv2.putText(dst,str,(8,20),cv2.FONT_HERSHEY_SIMPLEX,size,(18,233,157),t2,2)
    else:
        cv2.putText(dst,str,(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),3,2)
        cv2.putText(dst,str,(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(18,233,157),2,2)
    return (dst,p,c)
            
if __name__ == '__main__':
    import cv2
    src = cv2.imread('test2.png')
    det = detect2(src,0.5)
    r_img(src,th=0.5)