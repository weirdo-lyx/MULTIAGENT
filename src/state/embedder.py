"""M3 句向量生成器：优先使用本地 BGE 模型，离线时提供确定性兜底。"""
from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np

from src.config import load_config


class Embedder:
    """文本嵌入生成器。

    默认从 ``config/config.yaml`` 读取 ``embedder.model_name``，优先加载
    sentence-transformers 的本地模型。若模型未安装、离线不可用或测试显式使用
    ``model_name="mock"``，会退回到确定性哈希向量，保证离线环境也能跑通实验流程。
    """

    def __init__(
        self,
        model_name: str | None = None,
        offline: bool | None = None,
        dimension: int = 512,
        allow_fallback: bool = True,
    ) -> None:
        cfg = load_config().get("embedder", {})
        self.model_name = model_name or cfg.get("model_name", "BAAI/bge-small-zh-v1.5")
        self.offline = (
            bool(offline)
            if offline is not None
            else bool(cfg.get("offline", False)) or os.environ.get("HF_HUB_OFFLINE") == "1"
        )
        self.dimension = dimension
        self.allow_fallback = allow_fallback
        self._model: Any | None = None
        self._load_error: Exception | None = None

    def encode(self, texts: list[str]) -> np.ndarray:
        """将文本列表编码为 ``float32`` 矩阵，形状为 ``(N, 512)``。

        Args:
            texts: 待编码文本列表。

        Returns:
            ``np.ndarray``，真实 BGE 模型或确定性兜底都会返回 float32 向量。
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        model = self._get_model()
        if model is None:
            return self._encode_fallback(texts)

        try:
            vectors = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except TypeError:
            # 兼容较老版本 sentence-transformers，不支持部分关键字参数。
            vectors = model.encode(texts, convert_to_numpy=True)
        except Exception as exc:
            if not self.allow_fallback:
                raise
            self._load_error = exc
            return self._encode_fallback(texts)

        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _get_model(self) -> Any | None:
        """懒加载嵌入模型，避免测试导入阶段触发模型加载。"""
        if self.model_name in {"mock", "deterministic", "hash"}:
            return None
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            return None

        try:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {}
            if self.offline:
                kwargs["local_files_only"] = True
            self._model = SentenceTransformer(self.model_name, **kwargs)
            return self._model
        except TypeError as exc:
            # 某些旧版本不支持 local_files_only；离线环境由 HF_HUB_OFFLINE 兜住。
            try:
                from sentence_transformers import SentenceTransformer

                old_offline = os.environ.get("HF_HUB_OFFLINE")
                if self.offline:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                try:
                    self._model = SentenceTransformer(self.model_name)
                finally:
                    if self.offline:
                        if old_offline is None:
                            os.environ.pop("HF_HUB_OFFLINE", None)
                        else:
                            os.environ["HF_HUB_OFFLINE"] = old_offline
                return self._model
            except Exception as retry_exc:
                self._load_error = retry_exc
                if not self.allow_fallback:
                    raise retry_exc from exc
                return None
        except Exception as exc:
            self._load_error = exc
            if not self.allow_fallback:
                raise
            return None

    def _encode_fallback(self, texts: list[str]) -> np.ndarray:
        """生成确定性单位向量，用于离线测试和模型缺失时的可复现实验。"""
        rows: list[np.ndarray] = []
        for text in texts:
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
            seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dimension).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            rows.append(vec)
        return np.vstack(rows).astype(np.float32)
