# 🧠 基于 RAG 技术的垂直领域知识问答系统

[![GitHub last commit](https://img.shields.io/github/last-commit/Yaaia923/rag-system)](https://github.com/Yaaia923/rag-system)
[![GitHub repo size](https://img.shields.io/github/repo-size/Yaaia923/rag-system)](https://github.com/Yaaia923/rag-system)

> 一个基于 DeepSeek-R1 和 BGE 模型构建的专业领域知识问答系统，支持中文语义理解和答案溯源

## ✨ 核心功能

- **专业领域问答**：针对工业制造、医疗健康等垂直领域提供精准问答
- **答案溯源**：为每个答案提供可验证的文档来源
- **双模交互**：
  - 命令行界面（CLI）适合批量查询
  - Web 界面（Gradio）提供可视化交互
- **实时知识更新**：支持增量更新知识库
- **中文优化**：专为中文专业术语设计的分割和检索算法

## 🛠️ 技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 文本嵌入 | BGE-large-zh-v1.5 | 中文语义理解冠军模型 |
| 大语言模型 | DeepSeek-R1 | 中文强化，128K上下文 |
| 向量数据库 | FAISS | 毫秒级检索，支持增量更新 |
| 框架 | LangChain | 成熟的 RAG 实现框架 |
| 前端界面 | Gradio | 低代码可视化部署 |

## 🚀 快速开始

### 前提条件

- Python 3.10+
- Git
- DeepSeek API 密钥（[申请地址](https://platform.deepseek.com/)）

### 安装步骤

1. **克隆仓库**：
   ```bash
   git clone https://github.com/Yaaia923/rag-system.git
   cd rag-system
   ```

2. **创建虚拟环境**：
   ```bash
   python -m venv rag-venv
   # Windows (PowerShell):
   .\rag-venv\Scripts\Activate.ps1
   # Windows (CMD):
   rag-venv\Scripts\activate.bat
   # Linux/MacOS:
   source rag-venv/bin/activate
   ```

3. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

4. **配置环境变量**：
   创建 `.env` 文件并添加以下内容：
   ```env
   # DeepSeek API 密钥
   DEEPSEEK_API_KEY=your_api_key_here
   
   # 数据目录
   DATA_DIR=./data
   
   # 向量存储目录
   INDEX_PATH=./vector_store
   ```

5. **准备知识库文档**：
   将您的行业文档（PDF/DOCX/TXT）放入 `data/` 目录

### 运行系统

#### 命令行界面（CLI）
```bash
python app/main_cli.py
```

![CLI 演示](https://via.placeholder.com/600x300?text=CLI+Interface+Screenshot)

#### Web 界面
```bash
python web/gradio_app.py
```
访问 `http://localhost:8899`（默认用户名: admin，密码: password）

![Web 界面演示](https://via.placeholder.com/600x300?text=Web+Interface+Screenshot)

## 📂 项目结构

```
rag-system/
├── app/                # 核心应用代码
│   ├── core/           # 核心模块
│   │   ├── embedding.py # 嵌入模型处理
│   │   ├── loader.py   # 文档加载与处理
│   │   └── rag_chain.py # RAG 流程实现
│   └── main_cli.py     # 命令行入口
├── web/                # Web 界面
│   ├── assets/         # 静态资源
│   └── gradio_app.py   # Gradio 应用
├── data/               # 知识库文档
│   ├── technical_specifications.txt   # 技术规范
│   ├── industry_standards.txt         # 行业标准
│   └── equipment_manual.txt           # 设备手册
├── .gitignore          # Git 忽略规则
├── README.md           # 项目文档
├── requirements.txt    # 依赖列表
└── .env.example        # 环境变量示例
```

## 🧪 测试系统

1. 单元测试：
   ```bash
   pytest tests/
   ```

2. API 测试：
   ```bash
   python tests/test_api.py
   ```

3. 端到端测试：
   ```bash
   python tests/test_system.py
   ```

## 🔧 自定义配置

在 `app/core/rag_chain.py` 中可调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temperature` | 0.3 | 控制回答严谨性（0-1，值越小越严谨） |
| `search_kwargs` | {"k": 5} | 检索文档数量 |
| `chunk_size` | 500 | 文档分块大小 |
| `chunk_overlap` | 50 | 分块重叠大小 |


## 主要特性
- 支持txt/pdf/docx/md多格式文档
- 集成BGE大模型，优化中文检索
- 答案可溯源，标注具体文档来源
- 一键部署，自动构建知识库
- 代码结构清晰，易于扩展

## 创新点
- 行业适配性强，支持多格式与多类型知识融合
- 中文嵌入模型+BGE，提升行业语义检索效果
- 答案溯源机制，增强可信度



