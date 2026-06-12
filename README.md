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

已覆盖阶段 0-4 后端部分：

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
- 规则型 `RouterAgent`
- `QATool`、`SummaryTool`、`QuizTool`、`GradingTool`
- 统一 `AgentService` 调度入口
- `task_type=auto` 时使用 LLM Router Agent 决策任务类型和检索 query

## 检索配置

默认开启 reranker：

```env
ENABLE_RERANKER=true
RERANKER_MODEL=BAAI/bge-reranker-base
VECTOR_TOP_K=10
BM25_TOP_K=10
FINAL_TOP_K=5
SUMMARY_FINAL_TOP_K=10
VECTOR_WEIGHT=0.6
BM25_WEIGHT=0.4
```

第一次检索会下载 embedding / reranker 模型，耗时会明显更长。

## QA 与 Summary 检索策略

系统现在会在检索阶段区分问答和总结：

- `qa`：精确检索，候选较少，适合回答具体问题。
- `summary`：扩展多个主题 query，召回更多片段，并做来源多样性筛选，适合知识点梳理。
- `default`：用于普通检索调试，行为接近基础混合检索。

## Agent 路由

`/api/chat` 支持两种路由方式：

- 显式任务：前端或调用方传 `task_type="qa" / "summary" / "quiz" / "grade"`，系统尊重用户选择。
- 自动任务：传 `task_type="auto"`，系统调用 DeepSeek Flash 作为 Router Agent，输出任务类型、检索策略和改写后的检索 query。

如果 LLM Router 调用失败或输出无法解析，系统会自动回退到规则路由，保证接口可用。

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
$body = @{ query = "什么是条件随机场？"; top_k = 5; profile = "qa" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/search -Method Post -ContentType "application/json" -Body $body
```

总结检索测试：

```powershell
$body = @{ query = "总结一下中文分词的主要方法"; top_k = 10; profile = "summary" } | ConvertTo-Json
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

RAG 总结：

```powershell
$body = @{
  query = "总结一下中文分词的主要方法"
  task_type = "summary"
  use_pro_model = $false
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/chat -Method Post -ContentType "application/json" -Body $body
```

自动任务路由：

```powershell
$body = @{
  query = "围绕 HMM 和 CRF 的区别出三道题"
  task_type = "auto"
  use_pro_model = $false
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/chat -Method Post -ContentType "application/json" -Body $body
```

答案批改：

```powershell
$body = @{
  query = "题目：CRF 和 HMM 有什么区别？`n我的答案：HMM 是生成式模型，CRF 是判别式模型。"
  task_type = "grade"
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/chat -Method Post -ContentType "application/json" -Body $body
```

也可以打开 Swagger 页面测试：

```text
http://localhost:8000/docs
```

## 下一步

后续可继续补充：

- 前端界面
- Quiz / Grade 专用检索 profile
- RAG 评测集和检索指标
- OCR 文档支持
- 智能体多轮记忆
