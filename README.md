# Course RAG Agent

中文信息处理课程期末项目：基于 RAG 与任务型 Agent 的中文课程学习助手。

## 环境

先使用独立 conda 环境，不要在 base 环境中运行：

```powershell
conda create -n course-rag-agent python=3.10 -y
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

启动前需要在 `backend/.env` 中配置：

```env
DEEPSEEK_API_KEY=your_api_key_here
```

启动后访问：

```text
GET http://localhost:8000/api/health
```

## 当前阶段

已覆盖阶段 0-3：

- FastAPI 后端骨架
- PDF / Markdown / TXT 上传、解析、分块、入库
- sentence-transformers embedding
- FAISS 向量检索
- BM25 + 向量混合检索
- CrossEncoder reranker
- DeepSeek API RAG 问答
- 引用来源返回
- `/api/search` 检索调试接口
- 低置信度问答拒答

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

## 接口测试

健康检查：

```powershell
Invoke-RestMethod http://localhost:8000/api/health
```

上传文件：

```powershell
curl.exe -F "files=@D:\NLP-CourseWork\test.md" http://localhost:8000/api/documents/upload
```

检索测试：

```powershell
$body = @{ query = "什么是条件随机场？"; top_k = 5 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/search -Method Post -ContentType "application/json" -Body $body
```

RAG 问答：

```powershell
$body = @{
  query = "什么是条件随机场？"
  task_type = "qa"
  use_pro_model = $false
  top_k = 5
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/chat -Method Post -ContentType "application/json" -Body $body
```

也可以打开 Swagger 页面测试：

```text
http://localhost:8000/docs
```

## 下一步

阶段 4 将实现任务型 Agent：

- `RouterAgent`
- `SummaryTool`
- `QuizTool`
- `GradingTool`
- 统一 `AgentService`
