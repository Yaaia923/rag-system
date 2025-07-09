from dotenv import load_dotenv
import os
import logging
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from app.core.embedding import EmbeddingModel
from app.core.loader import DocumentLoader, build_vector_store
from langchain_community.vectorstores import FAISS
import requests
import json

load_dotenv()
logger = logging.getLogger(__name__)


class DeepSeekLLMWrapper(LLM):
    """简化版的 DeepSeek API LangChain 包装器"""
    model_name: str = "deepseek-chat"
    temperature: float = 0.3  # 控制专业严谨性

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """调用 DeepSeek API"""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return "API 密钥未设置"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }

        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"API 错误: {str(e)}")
            return f"API 错误: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model_name,
            "temperature": self.temperature
        }


def get_qa_chain(vector_store, llm):
    # 自定义提示模板
    prompt_template = """你是一名行业知识专家，请根据以下上下文回答问题：

    {context}

    问题：{question}

    回答要求：
    1. 基于资料内容给出准确答案
    2. 标注答案来源的具体文档
    3. 保持专业性和简洁性
    4. 如果资料中没有相关信息，请如实告知
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 创建QA链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),  # Top5召回
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True  # 关键溯源开关
    )

    return qa_chain


def prepare_vector_store(data_dir: str, index_path: str = "vector_store"):
    embedding_model = EmbeddingModel()

    # 确保索引路径存在
    os.makedirs(index_path, exist_ok=True)

    # 检查是否已有向量库
    faiss_index_path = os.path.join(index_path, "index.faiss")
    if os.path.exists(faiss_index_path):
        try:
            vector_store = FAISS.load_local(
                index_path,
                embedding_model.get_langchain_embedding(),
                allow_dangerous_deserialization=True
            )
            # 从向量库中提取元数据
            doc_metas = [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "file_path": doc.metadata.get("source", "")
                }
                for doc in vector_store.docstore._dict.values()
            ]
            logger.info(f"Loaded existing vector store with {len(doc_metas)} documents")
            return vector_store, doc_metas, embedding_model
        except Exception as e:
            logger.error(f"Error loading existing vector store: {e}")

    # 加载并处理文档
    docs = DocumentLoader.load_directory(data_dir)
    if not docs:
        raise ValueError(f"No documents found in {data_dir}")

    # 构建向量库
    vector_store, doc_metas = build_vector_store(
        docs,
        embedding_model,
        index_path
    )

    return vector_store, doc_metas, embedding_model


def rag_qa(question: str, qa_chain, doc_metas: List[dict]):
    try:
        # 使用LangChain QA链获取答案
        result = qa_chain({"query": question})  # 执行RAG流程

        # 提取来源文档
        source_docs = []
        for source_doc in result.get("source_documents", []):
            source_docs.append({
                "content": source_doc.page_content,
                "source": source_doc.metadata.get("source", "unknown"),
                "file_path": source_doc.metadata.get("source", "")  # 提取溯源信息#（溯源核心）
            })

        return result["result"], source_docs
    except Exception as e:
        logger.error(f"RAG QA error: {str(e)}")
        return f"处理错误: {str(e)}", []