"""快速测试：验证数据集和模型loss计算是否正常"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import io
from loguru import logger
logger.remove()
_sink = sys.stdout
try:
    if hasattr(sys.stdout, "buffer"):
        _sink = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass
logger.add(_sink, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

print("=" * 50)
print("1. 测试数据集 input_ids / labels 生成")
print("=" * 50)

from src.data.ecommerce_dataset import EcommerceProductDataset, create_ecommerce_dataloader

dataset = EcommerceProductDataset(
    data_path="data/ecommerce",
    max_text_length=128,  # 小一点加速测试
    image_size=224,
    split="train",
    include_copies=True,
)

sample = dataset[0]
print(f"  样本 keys: {list(sample.keys())}")
print(f"  input_ids 存在: {'input_ids' in sample}")
print(f"  labels 存在: {'labels' in sample}")
if "input_ids" in sample:
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  labels shape: {sample['labels'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  pixel_values shape: {sample['pixel_values'].shape}")
else:
    print("  [ERROR] input_ids NOT found! 修复失败!")
    sys.exit(1)

print("\n" + "=" * 50)
print("2. 测试 DataLoader batch")
print("=" * 50)

loader = create_ecommerce_dataloader(dataset, batch_size=4, shuffle=False)
batch = next(iter(loader))
print(f"  batch keys: {list(batch.keys())}")
print(f"  batch input_ids shape: {batch['input_ids'].shape}")
print(f"  batch labels shape: {batch['labels'].shape}")

print("\n" + "=" * 50)
print("3. 测试模型 forward + loss")
print("=" * 50)

from src.utils.config import get_config
config = get_config()

from src.models.multimodal_model import MultimodalBaseModel
model = MultimodalBaseModel(config.config)
model.eval()

import torch
with torch.no_grad():
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        task="matching",
        labels=batch["labels"],
    )

print(f"  outputs keys: {list(outputs.keys())}")
loss = outputs.get("loss")
if loss is not None:
    print(f"  loss = {loss.item():.4f}")
    print("\n  [OK] 模型正确返回 loss!")
else:
    print("\n  [ERROR] 模型仍然未返回 loss!")

print("\n" + "=" * 50)
print("全部测试完成!")
print("=" * 50)
