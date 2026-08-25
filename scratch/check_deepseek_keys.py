from transformers import AutoProcessor
import torch
from PIL import Image
import numpy as np

model_name = "deepseek-ai/DeepSeek-OCR-2"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
image = Image.fromarray(np.uint8(np.random.rand(1024, 1024, 3) * 255))
inputs = processor(text="<|User|>: <image>\nHello\n<|Assistant|>:", images=[image], return_tensors="pt")
print("Keys in inputs:", inputs.keys())
