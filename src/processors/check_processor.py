from transformers import AutoConfig

config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-VL2", trust_remote_code=True)
print("Module:", config.__class__.__module__, flush=True)
print("Auto map:", config.auto_map, flush=True)
import sys
sys.stdout.flush()