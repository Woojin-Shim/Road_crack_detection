import sys
sys.path.append('yolov5')

import torch
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

weights='last.pt'  # model.pt path(s)  #########################
# source='data/images'  # file/dir/URL/glob, 0 for webcam
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
line_thickness=3  # bounding box thickness (pixels)
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
    # dst = src.copy()

    # t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    pred = model(im, augment=augment, visualize=visualize)

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # Process predictions
    det = pred[0]  # per image

    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], src.shape).round()

        return det

def draw_boxes(src, det):
    dst = src.copy()
    annotator = Annotator(dst, line_width=line_thickness, example=str(names))
    for *xyxy, conf, cls in reversed(det):
        c = int(cls)  # integer class
        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
        annotator.box_label(xyxy, None, color=colors(c, True))

    return dst

if __name__ == '__main__':
     import cv2
     import os

     for root, dirs, filenames in os.walk('C:\\Users\\aischool\\vscode_test\\project_html'):
        for filename in filenames:
            if '.png' in filename:
                src = cv2.imread(os.path.join(root, filename))
                det = detect(src)
                if det is not None:
                    dst = draw_boxes(src, det)
                    # cv2.imshow('dst', dst)
                    # cv2.waitKey(0)
                    cv2.imwrite('C:/Users/aischool/Desktop/새 폴더/'+filename.replace('.png','_detected.png'), dst)
                    cv2.imwrite('C:/Users/aischool/Desktop/새 폴더/'+filename, src)
                else:
                    pass