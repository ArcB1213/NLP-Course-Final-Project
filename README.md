# Course RAG Agent

中文信息处理课程期末项目：基于 RAG 与任务型 Agent 的中文课程学习助手。

## 环境

```powershell
conda activate course-rag-agent
cd backend
pip install -r requirements.txt
```

## 后端启动

```powershell
cd backend
copy .env.example .env
python run.py
```

启动后访问：

```text
GET http://localhost:8000/api/health
```

## 当前阶段

已覆盖阶段 0-2：

- FastAPI 后端骨架
- PDF / Markdown / TXT 上传、解析、分块、入库
- sentence-transformers embedding
- FAISS 向量检索
- BM25 + 向量混合检索
- CrossEncoder reranker
- DeepSeek API RAG 问答
- 引用来源返回

## 检索配置

默认开启 reranker：

```env
ENABLE_RERANKER=true
RERANKER_MODEL=BAAI/bge-reranker-base
VECTOR_TOP_K=10
BM25_TOP_K=10
FINAL_TOP_K=5
VECTOR_WEIGHT=0.6
BM25_WEIGHT=0.4
```

第一次检索会下载 embedding / reranker 模型，耗时会明显更长。
