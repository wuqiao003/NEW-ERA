FROM python:3.10-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 项目代码
COPY . .

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=configs/base_config.yaml

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
