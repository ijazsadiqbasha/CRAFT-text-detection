from typing import List, Tuple, Optional
import os
import time
import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
import cv2
from CRAFT.craft import CRAFT, init_CRAFT_model
from CRAFT.refinenet import RefineNet, init_refiner_model
from CRAFT.craft_utils import adjustResultCoordinates, getDetBoxes
from CRAFT.imgproc import resize_aspect_ratio, normalizeMeanVariance

    
def preprocess_image(image: np.ndarray, canvas_size: int, mag_ratio: bool):
    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    return x, ratio_w, ratio_h


class CRAFTModel:
    
    def __init__(
        self,
        craft_model_path: str,
        device: torch.device,
        refiner_model_path: Optional[str] = None,
        use_refiner: bool = True,
        fp16: bool = True,
        canvas_size: int = 1280,
        mag_ratio: float = 1.5,
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.4
    ):
        self.craft_model_path = craft_model_path
        self.refiner_model_path = refiner_model_path
        self.use_refiner = use_refiner
        self.device = device
        self.fp16 = fp16

        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text

        # Validate model paths
        if not os.path.exists(craft_model_path):
            raise FileNotFoundError(f"CRAFT model not found at: {craft_model_path}")

        if use_refiner:
            if refiner_model_path is None:
                raise ValueError("refiner_model_path must be provided when use_refiner=True")
            if not os.path.exists(refiner_model_path):
                raise FileNotFoundError(f"Refiner model not found at: {refiner_model_path}")

        # Loading models
        self.net = init_CRAFT_model(craft_model_path, device, fp16=fp16)
        if self.use_refiner:
            self.refiner = init_refiner_model(refiner_model_path, device)
        else:
            self.refiner = None
        
    def get_text_map(self, x: torch.Tensor, ratio_w: int, ratio_h: int) -> Tuple[np.ndarray, np.ndarray]:
        x = x.to(self.device)

        # forward pass
        with torch.no_grad():
            y, feature = self.net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # refine link
        if self.refiner:
            with torch.no_grad():
                y_refiner = self.refiner(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()
            
        return score_text, score_link

    def get_polygons(self, image: Image.Image) -> List[List[List[int]]]:
        x, ratio_w, ratio_h = preprocess_image(np.array(image), self.canvas_size, self.mag_ratio)
        
        score_text, score_link = self.get_text_map(x, ratio_w, ratio_h)
        
        # Post-processing
        boxes, polys = getDetBoxes(
            score_text, score_link, 
            self.text_threshold, self.link_threshold, 
            self.low_text, True
        )
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: 
                polys[k] = boxes[k]
            else:
                polys[k] = adjustResultCoordinates(polys[k], ratio_w, ratio_h)

        res = []
        for poly in polys:
            res.append(poly.astype(np.int32).tolist())
        return res
    
    def _get_boxes_preproc(self, x, ratio_w, ratio_h) -> List[List[List[int]]]:
        score_text, score_link = self.get_text_map(x, ratio_w, ratio_h)
        
        # Post-processing
        boxes, polys = getDetBoxes(
            score_text, score_link, 
            self.text_threshold, self.link_threshold, 
            self.low_text, False
        )
        
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        boxes_final = []
        if len(boxes)>0:
            boxes = boxes.astype(np.int32).tolist()
            for box in boxes:
                boxes_final.append([box[0], box[2]])

        return boxes_final
    
    def get_boxes(self, image: Image.Image) -> List[List[List[int]]]:
        x, ratio_w, ratio_h = preprocess_image(np.array(image), self.canvas_size, self.mag_ratio)
        
        boxes_final = self._get_boxes_preproc(x, ratio_w, ratio_h)
        return boxes_final
