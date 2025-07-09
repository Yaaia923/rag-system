import os
import torch
import logging
from typing import List, Optional
from langchain.embeddings.base import Embeddings
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # 尝试从新位置导入
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Using langchain_huggingface.HuggingFaceEmbeddings")
except ImportError:
    # 如果新位置不可用，尝试从旧位置导入
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        logger.info("Using langchain_community.embeddings.HuggingFaceEmbeddings")
    except ImportError:
        # 如果都不可用，尝试从原始位置导入
        from langchain.embeddings import HuggingFaceEmbeddings

        logger.info("Using langchain.embeddings.HuggingFaceEmbeddings")


class LangChainEmbeddingWrapper(Embeddings):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.embedding_model.embed(texts)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.embedding_model.embed([text])
        return vector[0].tolist()


class EmbeddingModel:
    MODEL_CONFIGS = {
        "bge-large-zh": "BAAI/bge-large-zh-v1.5",
        "bge-base-zh": "BAAI/bge-base-zh-v1.5",
        "bge-small-zh": "BAAI/bge-small-zh-v1.5",
    }

    def __init__(self, model_name: Optional[str] = None, cache_dir: Optional[str] = None):
        if model_name is None:
            model_name = os.getenv("EMBEDDING_MODEL", "bge-large-zh")

        if model_name in self.MODEL_CONFIGS:
            self.model_name = self.MODEL_CONFIGS[model_name]
        else:
            self.model_name = model_name

        if cache_dir is None:
            cache_dir = os.getenv("MODEL_CACHE_DIR", "models")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # 创建 LangChain 兼容的嵌入模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {device}")

        self.langchain_embedding = HuggingFaceEmbeddings(
            model_name=self.model_name,
            cache_folder=self.cache_dir,
            model_kwargs={'device': device}  # 自动GPU加速
        )

        # 创建自定义包装器
        self.wrapper = LangChainEmbeddingWrapper(self)

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.langchain_embedding.embed_documents(texts)
        return np.array(embeddings, dtype='float32')

    def get_langchain_embedding(self):
        return self.wrapper