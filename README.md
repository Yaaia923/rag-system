# RAG垂直领域知识问答系统

## 项目简介
本系统基于RAG（检索增强生成）技术，结合BGE中文嵌入模型与大语言模型，实现行业知识的智能问答与答案溯源。

## 目录结构
- app/core/embedding.py  嵌入模型封装
- app/core/loader.py     文档加载、分段、向量化、向量库构建
- app/core/rag_chain.py  RAG主流程
- app/web/gradio_app.py  Web问答界面
- data/                  行业原始文档及说明
- vector_store/          向量知识库
- requirements.txt       依赖清单

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 将行业文档放入`data/`目录
3. 运行：`python app/web/gradio_app.py`
4. 浏览器访问Gradio界面，体验问答

## 主要特性
- 支持txt/pdf/docx多格式文档
- 集成BGE大模型，优化中文检索
- 答案可溯源，标注具体文档来源
- 一键部署，自动构建知识库
- 代码结构清晰，易于扩展

## 创新点
- 行业适配性强，支持多格式与多类型知识融合
- 中文嵌入模型+BGE，提升行业语义检索效果
- 答案溯源机制，增强可信度



