"""检查环境和数据目录"""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Check Python
print(f"Python: {sys.version}")

# Check PyTorch
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch: NOT INSTALLED")

# Check key packages
for pkg in ["transformers", "click", "loguru", "fastapi", "numpy", "faiss"]:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "ok")
        print(f"{pkg}: {ver}")
    except ImportError:
        print(f"{pkg}: NOT INSTALLED")

# Check data directory
print("\n--- Data directories ---")
for d in ["data", "data/ecommerce", "data/ecommerce/images", "data/processed", "data/raw", "data/preference"]:
    exists = os.path.exists(d)
    if exists and os.path.isdir(d):
        files = os.listdir(d)
        print(f"{d}: EXISTS ({len(files)} items) -> {files[:10]}")
    else:
        print(f"{d}: MISSING")

# Check outputs directory
print("\n--- Output directories ---")
for d in ["outputs", "outputs/sft", "outputs/dpo", "outputs/ppo"]:
    exists = os.path.exists(d)
    print(f"{d}: {'EXISTS' if exists else 'MISSING'}")
