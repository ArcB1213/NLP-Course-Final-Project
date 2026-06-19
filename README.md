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

前端页面挂在根路径，启动后端后直接打开：

```text
http://localhost:8000/
```

页面包含三栏布局：左侧资料库（上传 / 列表 / 删除），中间对话（流式输出，Enter 发送、Shift+Enter 换行），右侧设置（任务类型 / top_k / Pro 模型）。右上角圆形按钮可在深色 / 浅色主题之间切换。

支持上传 PDF / Markdown (`.md`/`.markdown`) / TXT，单个文件默认上限 100 MB，可在 `.env` 配置 `MAX_UPLOAD_SIZE_MB` 调整。

## 当前阶段

后端已完成可演示的 RAG + 轻量 Agentic RAG 闭环：

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
- `RouterAgent`：显式任务优先，`task_type=auto` 时使用 LLM 判断任务类型、检索策略和改写 query
- `AgentPlanner`：自动模式下可将复合请求拆成多个子任务，例如“总结 + 出题”
- `EvidenceJudge`：检索后判断证据是否充分，不足时触发二次 query 改写与再检索
- `SummaryCoveragePlanner`：总结任务会先规划复习子主题，并最多进行 3 轮覆盖式检索
- `AgentTrace`：可选返回路由、规划、检索、证据判断、工具调用等决策过程
- `QATool`、`SummaryTool`、`QuizTool`、`GradingTool`
- 统一 `AgentService` 调度入口：`route -> plan -> retrieve -> judge -> retry/coverage retrieve -> tool generate`
- `/api/chat/stream` SSE 流式输出，前端逐字渲染
- DeepSeek 输出被截断时自动续写，降低回答停在半句话的问题
- 删除文档后显式清理分块和孤儿 chunk，避免旧资料继续污染检索结果
- 单页前端（HTML + 原生 JS），由 FastAPI 静态托管
- 实现了Agent的多轮记忆功能，分为短期记忆（保存一定对话轮次）和长期记忆（跨对话保存），并在 AgentTrace 中记录记忆内容和使用情况，方便调试和分析。

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
- `summary`：使用复习覆盖规划器生成子主题 query，最多进行 3 轮覆盖式检索，并做来源和内容多样性筛选，适合知识点梳理。
- `default`：用于普通检索调试，行为接近基础混合检索。

## Agentic RAG 流程

`/api/chat` 和 `/api/chat/stream` 共用同一套 Agent 执行逻辑：

```text
用户问题
  ↓
RouterAgent 判断任务类型、检索 profile、改写 query
  ↓
AgentPlanner 判断是否需要拆成多个子任务
  ↓
HybridRetriever 初次检索 / Summary 覆盖检索
  ↓
EvidenceJudge 或 SummaryCoveragePlanner 判断证据是否充分
  ↓
证据不足时进行 query 扩展 / 改写并继续检索
  ↓
调用 QA / Summary / Quiz / Grading 工具生成回答
  ↓
返回答案、引用来源和可选 AgentTrace
```

任务路由支持两种方式：

- 显式任务：前端或调用方传 `task_type="qa" / "summary" / "quiz" / "grade"`，系统尊重用户选择。
- 自动任务：传 `task_type="auto"`，系统调用 DeepSeek Flash 作为 Router Agent，输出任务类型、检索策略和改写后的检索 query。

如果 LLM Router / Planner / EvidenceJudge 调用失败或输出无法解析，系统会自动回退到规则逻辑，保证接口可用。

如需查看 Agent 决策过程，可在请求中加入：

```json
{
  "extra_context": {
    "debug_agent_trace": true
  }
}
```

返回中的 `agent_trace.steps` 会记录 `route`、`plan`、`retrieve`、`judge_evidence`、`retrieve_retry`、`retrieve_round_1/2/3`、`summary_coverage_judge_*`、`summary_coverage_stop`、`tool_generate` 等步骤。

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
  query = "总结自动分词的主要知识"
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

查看 Agent 决策过程：

```powershell
$body = @{
  query = "总结 HMM 和 CRF 的区别，并围绕它们出三道题"
  task_type = "auto"
  top_k = 8
  extra_context = @{ debug_agent_trace = $true }
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://localhost:8000/api/chat -Method Post -ContentType "application/json" -Body $body
```

流式输出：

```powershell
$body = @{
  query = "总结自动分词的主要知识"
  task_type = "summary"
  top_k = 8
} | ConvertTo-Json
curl.exe -N -H "Content-Type: application/json" -d $body http://localhost:8000/api/chat/stream
```

选择题批改会优先判断学生所选选项是否是题干给定选项中的最佳答案；课程资料中的更完整知识列表只作为补充说明，不会直接导致正确选项被判为“部分正确”。

也可以打开 Swagger 页面测试：

```text
http://localhost:8000/docs
```

## 下一步

后续可继续补充：

- Quiz / Grade 专用检索 profile
- RAG 评测集和检索指标
- OCR 文档支持
- 章节级索引、Parent-Child 检索或文档目录结构解析，进一步增强长文档总结质量
