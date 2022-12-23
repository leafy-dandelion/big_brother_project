import numpy as np

from openvino.runtime import Core
import torch
from torch import nn
import onnxruntime
import cv2 

from yolov5.models.experimental import attempt_load
from yolov_5.utils.utils import (
    letterbox,
    check_img_size,
    non_max_suppression,
    scale_coords,
)


class YOLOv5CV_P(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(
        self,
        weights="./yolov_5/weights/yolov5s.pt",
        imgsz=640,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device="cpu",
    ):
        super().__init__()

        model = attempt_load(weights) #, map_location=device)
        w = str(weights[0] if isinstance(weights, list) else weights)
        names = (
            model.module.names if hasattr(model, "module") else model.names
        )  # get class names
        stride = int(model.stride.max())  # model stride

        imgsz = check_img_size(imgsz, s=stride)  # check image size

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        y = self.model(im, augment=augment)
        return y[0]

    def rescale_coords(self, shape_into_model, shape_of_raw_img,det):
        height_coef = shape_of_raw_img[0]/shape_into_model[0]
        width_coef = shape_of_raw_img[1]/shape_into_model[1]
        rescaled_detect = list()
        for detect in det:
            rescaled_det = []
            rescaled_det.append(detect[0]*width_coef)
            rescaled_det.append(detect[1]*height_coef)
            rescaled_det.append(detect[2]*width_coef)
            rescaled_det.append(detect[3]*height_coef)
            rescaled_det = torch.tensor(rescaled_det).to(self.device)
            rescaled_detect.append(rescaled_det)
        rescaled_detect = tuple(rescaled_detect)
        final_detect = torch.stack(rescaled_detect, dim = 0)
        return final_detect


    @torch.no_grad()
    def detect(self, image):
        start_shape = image.shape[:2]
        img = letterbox(image, self.imgsz, stride=self.stride, auto=True)[0]
        
        final_shape = img.shape[:2]


        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        
        image_tensor = torch.from_numpy(img).to(self.device)
        image_tensor = image_tensor.float()
        image_tensor /= 255  # 0 - 255 to 0.0 - 1.0
        

        
        pred = self.model(image_tensor.unsqueeze(0))[0]
        det = non_max_suppression(pred.unsqueeze(0), self.conf_thres, self.iou_thres, classes=None,
                                  agnostic=False)[0]

        
        if det is not None and len(det):
        # Rescale boxes from img_size to im0s size
            det[:, :4] = self.rescale_coords(final_shape, start_shape,det[:,:4]).round()
        return det

    def detect_with_onnx(self,image,weights = './yolov_5/weights/yolov5s_b.onnx'):
        session =  onnxruntime.InferenceSession(weights, None)
        input_name = session.get_inputs()[0].name
        start_shape = image.shape[:2]
        img = letterbox(image, new_shape = [416,416],  stride=self.stride, auto=False)[0]
        final_shape = img.shape[:2]


        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        
       
        
        img = img/255
        img = np.expand_dims(img, axis = 0)
        
        
        pred = [session.run(None, {input_name: img})][0]
        pred = np.array(pred)
        pred = torch.from_numpy(pred).to(self.device)
        det = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None,
                                  agnostic=False)[0]
        if det is not None and len(det):
        # Rescale boxes from img_size to im0s size
            det[:, :4] = self.rescale_coords(final_shape, start_shape,det[:,:4]).round()
        return det



    
