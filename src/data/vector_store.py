"""
向量检索引擎 — 基于 FAISS 的商品语义检索
核心能力:
  1. 商品特征向量的索引构建与持久化
  2. 文本 / 图像 → 商品的跨模态检索
  3. 增量更新索引（新商品上架）
  4. 支持过滤条件（类目、价格区间等）
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("⚠️ faiss 未安装，向量检索将使用 numpy 暴力搜索作为后备")


# ============ FAISS 向量存储 ============


class ProductVectorStore:
    """
    商品向量存储与检索引擎

    支持两种后端：
      - faiss (推荐，高性能)
      - numpy brute-force (后备方案)
    """

    def __init__(
        self,
        dim: int = 512,
        index_type: str = "IVFFlat",
        nlist: int = 100,
        nprobe: int = 10,
        use_gpu: bool = False,
        index_path: Optional[str] = None,
    ):
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu and FAISS_AVAILABLE

        # 元数据存储 (id -> product info)
        self.metadata: Dict[str, Dict[str, Any]] = {}
        # id -> index position 映射
        self.id_to_pos: Dict[str, int] = {}
        self.pos_to_id: Dict[int, str] = {}
        self.current_size: int = 0

        # numpy 后备存储
        self._np_vectors: Optional[np.ndarray] = None

        # 初始化索引
        self.index = None
        self._init_index()

        # 加载已有索引
        if index_path and os.path.exists(index_path):
            self.load(index_path)

        logger.info(
            f"🔍 向量检索引擎初始化: dim={dim}, backend={'faiss' if FAISS_AVAILABLE else 'numpy'}, "
            f"index_type={index_type}"
        )

    def _init_index(self):
        """初始化 FAISS 索引"""
        if not FAISS_AVAILABLE:
            self._np_vectors = np.zeros((0, self.dim), dtype=np.float32)
            return

        if self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.dim)  # 内积 (余弦相似度)
        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist)
            self.index.nprobe = self.nprobe
        elif self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.dim, 32)
            self.index.hnsw.efSearch = 64
        else:
            self.index = faiss.IndexFlatIP(self.dim)

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                logger.warning(f"GPU 索引创建失败，回退 CPU: {e}")

    def add(
        self,
        product_id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        添加单个商品向量

        Args:
            product_id: 商品 ID
            vector: 特征向量 [dim]
            metadata: 商品元数据 (title, category, price, ...)
        """
        vector = self._normalize(vector.reshape(1, -1))

        if FAISS_AVAILABLE and self.index is not None:
            # IVFFlat 需要先训练
            if hasattr(self.index, "is_trained") and not self.index.is_trained:
                # 延迟训练 —— 在 batch_add 中处理
                pass
            self.index.add(vector)
        else:
            if self._np_vectors is not None and self._np_vectors.shape[0] > 0:
                self._np_vectors = np.vstack([self._np_vectors, vector])
            else:
                self._np_vectors = vector.copy()

        pos = self.current_size
        self.id_to_pos[product_id] = pos
        self.pos_to_id[pos] = product_id
        if metadata:
            self.metadata[product_id] = metadata
        self.current_size += 1

    def batch_add(
        self,
        product_ids: List[str],
        vectors: np.ndarray,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        批量添加商品向量

        Args:
            product_ids: 商品 ID 列表
            vectors: 特征向量矩阵 [N, dim]
            metadata_list: 元数据列表
        """
        assert len(product_ids) == vectors.shape[0], "ID数和向量数不匹配"
        vectors = self._normalize(vectors)

        if FAISS_AVAILABLE and self.index is not None:
            # IVFFlat 需要训练
            if hasattr(self.index, "is_trained") and not self.index.is_trained:
                train_size = max(self.nlist * 10, vectors.shape[0])
                if vectors.shape[0] >= self.nlist:
                    self.index.train(vectors)
                else:
                    # 数据不够时补充随机向量
                    pad = np.random.randn(
                        train_size - vectors.shape[0], self.dim
                    ).astype(np.float32)
                    pad = self._normalize(pad)
                    self.index.train(np.vstack([vectors, pad]))

            self.index.add(vectors)
        else:
            if self._np_vectors is not None and self._np_vectors.shape[0] > 0:
                self._np_vectors = np.vstack([self._np_vectors, vectors])
            else:
                self._np_vectors = vectors.copy()

        for i, pid in enumerate(product_ids):
            pos = self.current_size + i
            self.id_to_pos[pid] = pos
            self.pos_to_id[pos] = pid
            if metadata_list and i < len(metadata_list):
                self.metadata[pid] = metadata_list[i]

        self.current_size += len(product_ids)
        logger.info(f"📦 批量添加 {len(product_ids)} 条向量，总计 {self.current_size}")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        category_filter: Optional[str] = None,
        price_range: Optional[Tuple[float, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        向量检索

        Args:
            query_vector: 查询向量 [dim]
            top_k: 返回 top-k 结果
            category_filter: 类目过滤
            price_range: 价格区间 (min, max)

        Returns:
            [{"product_id": str, "score": float, "metadata": dict}, ...]
        """
        if self.current_size == 0:
            return []

        query = self._normalize(query_vector.reshape(1, -1))

        # 有过滤条件时多检索一些
        search_k = top_k * 3 if (category_filter or price_range) else top_k

        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(query, min(search_k, self.current_size))
            scores = scores[0]
            indices = indices[0]
        else:
            # numpy 暴力搜索
            scores = (self._np_vectors @ query.T).flatten()
            k = min(search_k, len(scores))
            if k >= len(scores):
                # k 等于总数时直接全排序
                indices = np.argsort(-scores)
            else:
                indices = np.argpartition(-scores, k)[:k]
                indices = indices[np.argsort(-scores[indices])]
            scores = scores[indices]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            pid = self.pos_to_id.get(int(idx))
            if pid is None:
                continue

            meta = self.metadata.get(pid, {})

            # 过滤条件
            if category_filter and meta.get("category", "") != category_filter:
                continue
            if price_range:
                price = meta.get("price", 0)
                if price < price_range[0] or price > price_range[1]:
                    continue

            results.append({
                "product_id": pid,
                "score": round(float(score), 4),
                "metadata": meta,
            })

            if len(results) >= top_k:
                break

        return results

    @torch.no_grad()
    def search_by_text(
        self,
        text: str,
        encoder,
        tokenizer=None,
        top_k: int = 10,
        **filter_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        文本 → 商品检索

        Args:
            text: 查询文本
            encoder: 文本编码模型 (需返回向量)
            tokenizer: 分词器
            top_k: 返回数
        """
        # 编码查询文本
        if tokenizer is not None:
            enc = tokenizer(
                text, max_length=128, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            text_features = encoder(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
        else:
            text_features = encoder(text)

        if isinstance(text_features, dict):
            text_features = text_features.get(
                "text_features",
                text_features.get("fused_features", next(iter(text_features.values()))),
            )
        if isinstance(text_features, torch.Tensor):
            text_features = text_features.cpu().numpy()

        if text_features.ndim > 1:
            text_features = text_features.mean(axis=0) if text_features.shape[0] > 1 else text_features.squeeze(0)

        return self.search(text_features, top_k=top_k, **filter_kwargs)

    @torch.no_grad()
    def search_by_image(
        self,
        image_tensor: torch.Tensor,
        encoder,
        top_k: int = 10,
        **filter_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        图像 → 商品检索

        Args:
            image_tensor: 图像张量 [3, H, W] 或 [1, 3, H, W]
            encoder: 图像编码模型
            top_k: 返回数
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        img_features = encoder(image_tensor)
        if isinstance(img_features, dict):
            img_features = img_features.get(
                "vision_features",
                img_features.get("fused_features", next(iter(img_features.values()))),
            )
        if isinstance(img_features, torch.Tensor):
            img_features = img_features.cpu().numpy()

        if img_features.ndim > 1:
            img_features = img_features.mean(axis=0) if img_features.shape[0] > 1 else img_features.squeeze(0)

        return self.search(img_features, top_k=top_k, **filter_kwargs)

    def build_index_from_dataset(
        self,
        dataset,
        encoder,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        """
        从电商数据集构建索引

        Args:
            dataset: EcommerceProductDataset 实例
            encoder: 多模态编码模型
            batch_size: 批次大小
            device: 设备
        """
        from torch.utils.data import DataLoader

        start_time = time.time()
        logger.info(f"🔨 开始构建商品索引: {len(dataset)} 条数据")

        encoder.eval()
        encoder.to(device)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_ids = []
        all_vectors = []
        all_meta = []

        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)

            # 获取文本特征
            if "input_ids" in batch:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                )
            else:
                outputs = encoder(pixel_values=pixel_values)

            features = outputs.get(
                "fused_features",
                outputs.get("text_features", outputs.get("vision_features")),
            )
            if features is None:
                continue

            # 池化到 [B, D]
            if features.dim() == 3:
                features = features.mean(dim=1)

            vectors = features.cpu().numpy()
            all_vectors.append(vectors)

            # 收集 ID 和元数据
            ids = batch.get("id", [f"P{i}" for i in range(len(vectors))])
            if isinstance(ids, torch.Tensor):
                ids = [str(i.item()) for i in ids]

            for i, pid in enumerate(ids):
                all_ids.append(pid)
                meta = {
                    "category": batch.get("category", [""])[i] if isinstance(batch.get("category"), list) else "",
                    "price": float(batch["price"][i]) if "price" in batch else 0.0,
                }
                all_meta.append(meta)

        if all_vectors:
            all_vectors = np.concatenate(all_vectors, axis=0).astype(np.float32)
            self.batch_add(all_ids, all_vectors, all_meta)

        elapsed = time.time() - start_time
        logger.info(f"✅ 索引构建完成: {self.current_size} 条, 耗时 {elapsed:.1f}s")

    # ============ 持久化 ============

    def save(self, path: str):
        """保存索引和元数据到磁盘"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 保存元数据
        meta_data = {
            "dim": self.dim,
            "size": self.current_size,
            "index_type": self.index_type,
            "metadata": self.metadata,
            "id_to_pos": self.id_to_pos,
            "pos_to_id": {str(k): v for k, v in self.pos_to_id.items()},
        }
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

        # 保存索引
        if FAISS_AVAILABLE and self.index is not None:
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(path / "index.faiss"))
            else:
                faiss.write_index(self.index, str(path / "index.faiss"))
        elif self._np_vectors is not None:
            np.save(str(path / "vectors.npy"), self._np_vectors)

        logger.info(f"💾 向量索引已保存: {path}")

    def load(self, path: str):
        """从磁盘加载索引"""
        path = Path(path)

        meta_file = path / "meta.json"
        if not meta_file.exists():
            logger.warning(f"⚠️ 未找到索引元数据: {meta_file}")
            return

        with open(meta_file, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        self.dim = meta_data.get("dim", self.dim)
        self.current_size = meta_data.get("size", 0)
        self.metadata = meta_data.get("metadata", {})
        self.id_to_pos = meta_data.get("id_to_pos", {})
        self.pos_to_id = {int(k): v for k, v in meta_data.get("pos_to_id", {}).items()}

        # 加载索引
        index_file = path / "index.faiss"
        npy_file = path / "vectors.npy"

        if FAISS_AVAILABLE and index_file.exists():
            self.index = faiss.read_index(str(index_file))
            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                except Exception:
                    pass
        elif npy_file.exists():
            self._np_vectors = np.load(str(npy_file))

        logger.info(f"📂 向量索引已加载: {self.current_size} 条 from {path}")

    # ============ 工具方法 ============

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2 归一化（用于余弦相似度）"""
        vectors = vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return vectors / norms

    def __len__(self) -> int:
        return self.current_size

    def get_product_info(self, product_id: str) -> Optional[Dict[str, Any]]:
        """获取商品元数据"""
        return self.metadata.get(product_id)

    def clear(self):
        """清空索引"""
        self.metadata.clear()
        self.id_to_pos.clear()
        self.pos_to_id.clear()
        self.current_size = 0
        self._init_index()
        logger.info("🗑️ 向量索引已清空")
