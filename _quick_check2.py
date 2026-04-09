import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from loguru import logger
logger.remove()

import torch
from src.data.ecommerce_dataset import EcommerceProductDataset, create_ecommerce_dataloader
from src.utils.config import get_config
from src.models.multimodal_model import MultimodalBaseModel

# 数据
d = EcommerceProductDataset("data/ecommerce", max_text_length=64, image_size=224, split="train")
loader = create_ecommerce_dataloader(d, batch_size=4, shuffle=False)
batch = next(iter(loader))
print("batch keys:", list(batch.keys()))

# 模型
config = get_config()
model = MultimodalBaseModel(config.config)
model.eval()

with torch.no_grad():
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        task="matching",
        labels=batch["labels"],
    )

print("output keys:", list(out.keys()))
loss = out.get("loss")
if loss is not None:
    print(f"loss = {loss.item():.4f}")
    print("SUCCESS - model returns loss!")
else:
    print("FAIL - no loss returned")
