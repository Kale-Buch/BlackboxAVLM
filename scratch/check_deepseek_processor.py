from transformers import AutoProcessor
from PIL import Image
import torch

model_name = "deepseek-ai/DeepSeek-OCR-2"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

image = Image.new('RGB', (1024, 1024), color='red')
inputs = processor(text="<|User|>: <image>\nExtract text.\n<|Assistant|>:", images=[image], return_tensors="pt")

print("Keys in inputs:", inputs.keys())
if "images_spatial_crop" in inputs:
    print("images_spatial_crop shape:", inputs["images_spatial_crop"].shape)
    print("images_spatial_crop values:", inputs["images_spatial_crop"])
if "images" in inputs:
    print("images shape:", inputs["images"].shape)
