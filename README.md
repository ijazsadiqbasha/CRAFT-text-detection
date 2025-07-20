# CRAFT-text-detection

An unofficial PyTorch implementation of CRAFT text detector with better interface and fp16 support

> This is not official implementation. I partially use code from the [original repository](https://github.com/clovaai/CRAFT-pytorch)

Main features of this implementation:
- User-friendly interface
- Easier to integrate this model in your project
- fp16 inference support
- Improved compatibility with modern dependencies

## Recent Updates

This version has been updated for better compatibility and dependency management:
- Removed automatic model downloading - users now specify model paths explicitly
- Replaced skimage with PIL for lighter dependencies
- Improved error handling and validation

## Installation

Recommended:
```bash
pip install git+https://github.com/boomb0om/CRAFT-text-detection/
```
or
```bash
git clone https://github.com/boomb0om/CRAFT-text-detection
cd CRAFT-text-detection/
pip install -r requirements.txt
```

To test model you can run `test.py` file.

## Model Weights

You need to download the model weights manually. The pre-trained models are available at:
- CRAFT model: [craft_mlt_25k.pth](https://huggingface.co/boomb0om/CRAFT-text-detector/blob/main/craft_mlt_25k.pth)
- Refiner model: [craft_refiner_CTW1500.pth](https://huggingface.co/boomb0om/CRAFT-text-detector/blob/main/craft_refiner_CTW1500.pth)

## Examples

```python
from PIL import Image
from CRAFT import CRAFTModel, draw_polygons

# Initialize model with explicit paths to downloaded weights
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
```

You can find more usage examples in [example.ipynb](example.ipynb)

![](images/cafe_sign.jpg)

Detected polygons:

![](images/result.jpg)
