from transformers import AutoProcessor
from PIL import Image
import torch

model_name = "deepseek-ai/DeepSeek-OCR-2"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

print("Processor type:", type(processor))
if hasattr(processor, "image_processor"):
    print("Image processor type:", type(processor.image_processor))
    
image = Image.new('RGB', (1024, 1024), color='red')
# Try to process just the image
img_inputs = processor.image_processor(image, return_tensors="pt")
print("Keys in img_inputs:", img_inputs.keys())
for k, v in img_inputs.items():
    if isinstance(v, torch.Tensor):
        print(f"{k} shape: {v.shape}")
    else:
        print(f"{k}: {v}")
