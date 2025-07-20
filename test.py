import torch
from PIL import Image
import numpy as np
from CRAFT import CRAFTModel, draw_polygons


if __name__ == "__main__":
    model = CRAFTModel(
        craft_model_path='weights/craft_mlt_25k.pth',
        device='cuda',
        refiner_model_path='weights/craft_refiner_CTW1500.pth',
        use_refiner=True,
        fp16=True
    )
    
    img = Image.open('images/cafe_sign.jpg')
    polygons = model.get_polygons(img)
    
    result = draw_polygons(img, polygons)
    result.save('images/result.jpg')
    print(f'Result saved to: images/result.jpg')