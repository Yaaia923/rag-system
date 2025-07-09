import os
import json
import logging
from typing import List
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    SUPPORTED_EXTS = ('.txt', '.pdf', '.docx', '.md')

    @staticmethod
    def get_loader(file_path: str):
        if file_path.endswith('.txt'):
            return TextLoader(file_path, encoding='utf-8')
        elif file_path.endswith('.pdf'):
            return PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            return Docx2txtLoader(file_path)
        elif file_path.endswith('.md'):
            return UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    @classmethod
    def load_directory(cls, data_dir: str) -> List[Document]:
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        all_docs = []
        for fname in os.listdir(data_dir):
            if not any(fname.endswith(ext) for ext in cls.SUPPORTED_EXTS):
                continue

            fpath = os.path.join(data_dir, fname)
            try:
                loader = cls.get_loader(fpath)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = fname  # 关键溯源标识
                all_docs.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {fname}")
            except Exception as e:
                logger.error(f"Error loading {fname}: {str(e)}")

        return all_docs


def build_vector_store(docs, embedding_model, index_path: str):
    # 分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 适合中文语义的片段长度
        chunk_overlap=50,  # 确保上下文连贯
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]  # 中文敏感分隔符
    )
    split_docs = text_splitter.split_documents(docs)  # 自动继承metadata

    # 创建向量库
    vector_store = FAISS.from_documents(
        split_docs,
        embedding_model.get_langchain_embedding()
    )

    # 确保目录存在 - 修复路径问题
    os.makedirs(index_path, exist_ok=True)

    # 保存向量库
    vector_store.save_local(index_path)  # 本地持久化

    # 提取文档元数据
    doc_metas = []
    for i, doc in enumerate(split_docs):
        doc_metas.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", f"document_{i}"),
            "file_path": doc.metadata.get("source", "")  # 溯源关键字段
        })

    logger.info(f"Vector store built with {len(split_docs)} chunks at {index_path}")
    return vector_store, doc_metas