import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
# 静默loguru
from loguru import logger
logger.remove()

from src.data.ecommerce_dataset import EcommerceProductDataset
d = EcommerceProductDataset("data/ecommerce", max_text_length=64, image_size=32, split="train")
s = d[0]
print("keys:", list(s.keys()))
print("has input_ids:", "input_ids" in s)
if "input_ids" in s:
    print("input_ids shape:", s["input_ids"].shape)
    print("labels shape:", s["labels"].shape)
    print("OK - dataset fixed!")
else:
    print("FAIL - no input_ids")
