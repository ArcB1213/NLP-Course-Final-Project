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
DEEPSEEK_THINKING=false
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

PDF 上传默认调用独立环境中的 MinerU 4.0.12 `standard` 本地解析，使用 ONNX 小模型和
llama-cpp 视觉模型（有兼容 GPU 时通过 Vulkan 加速），识别图片表格、OCR 文字及 LaTeX 公式。
保留原始页码、块坐标和图片产物；表格／公式仍需核对，示意图的关系尚未转为文字证据。
图片与检索文本分开保存，图片链接及 Base64 载荷不进入分块、出题或检索上下文。
有核对提示的文档显示“部分识别”，悬停状态可查看原因。首次解析可能需要数分钟。
**首次使用先安装独立 MinerU 环境和标准版模型**；未安装时上传 PDF 会明确报错，不会静默退回原生提取。
已有资料可运行 `backend/reparse_documents.py`，保留文档 ID 并替换旧分块、重建索引，自动备份数据库和索引。
`backend/parse_document.py` 仍是原生提取预览工具；安装、配置和重解析命令见
[文档解析与质量检查](docs/document-processing.md)。

当前仍存在复杂公式识别误差，以及公式与说明被拆开后漏召回的问题；本阶段尚未完成语义分块或相邻片段扩展。
MinerU PDF 处理默认增加文本 LLM 标题判定：把每组最多4页的候选文字、位置坐标和短正文发送到已配置的 DeepSeek 接口，
不发送图片、颜色或高亮信息；由代码校验源块 ID 后恢复页面标题，并给分块补充标题／章节上下文。
章节推断会明确标记；这一步尚未建立完整章节树。结果按输入、模型与提示版本缓存，失败时保留正文和候选并提示复核。
标题请求保留思考模式；若输出达到 6000 token 上限，只重试一次并将上限提高到 12000，仍须通过完整响应与结构校验。
诊断保存在 MinerU 缓存的 `titles/attempts/` 下，包括结束原因和截断请求的 token 用量。重解析已有分块的资料时，
标题处理失败会保留旧分块和解析报告并报错；重跑会复用成功的解析和标题缓存，只对缺失的标题批次调用接口。
批量脚本收到 Ctrl+C 时仍会为已经提交的资料重建索引并保存进度；请等待收尾完成，强制结束进程无法保证执行收尾。
如需完全离线处理，可设 `PDF_TITLE_RESOLUTION=false`，只运行本地 MinerU，保留原有标题选取行为。
运行 `cd backend`、`python -m pytest -q` 可检查自动化回归。测试覆盖解析接入、分块、缓存、失败处理及既有功能，
其中两项真实图片渲染测试需要独立 MinerU 环境（未安装则跳过），不运行模型推理。
这些测试不代表公式识别准确率或 RAG 回答准确率评测。

## 会话摘要

一轮指一条用户问题与一条助手回答。默认完成第 12 轮后首次生成摘要，此后每新增
6 轮更新一次（第 18、24 轮等）。`backend/.env` 可配置
`MEMORY_SUMMARY_THRESHOLD=12`、`MEMORY_SUMMARY_INTERVAL=6`、`MEMORY_RECENT_TURNS=8`。
`MEMORY_SUMMARY_THRESHOLD` 现在按完整轮数计数，旧配置若按消息条数设置，需要调整。

摘要只覆盖较早对话，保留当前任务、有效约束、必要结果的消息 ID 和未解决事项；
不会要求模型复述整段课程知识，也不应把检索失败归因为用户薄弱点。最近 8 轮及
两次摘要之间尚未压缩的消息保留原文。上下文超预算时按完整轮次移出，并提供历史消息目录；
解析历史指代后可回读原文，避免依靠摘要重建题目。超长消息进入摘要前分段处理全文。

摘要仍保存在 SQLite 的 `conversation_summaries.content`，格式为带版本号和更新轮次的
任务状态 JSON；旧文本摘要会在下一次满足条件且生成成功时从原始消息重建，不继承旧摘要内容。
摘要失败、格式不合格或输出截断不会推进覆盖位置。

## 历史会话

左侧“历史会话”按最近活动排序，显示首条提问（或已有会话标题）及消息数。
选择会话会加载完整历史，并在该会话中继续对话；刷新页面会恢复上次选择。
“+”开启新会话，每条历史右侧可删除对应会话。删除会清理该会话的消息和摘要，
不会删除已形成的长期记忆。历史仅恢复已保存的消息正文，不包含未持久化的检索来源和决策过程。
接口为 `GET /api/sessions`、`GET /api/sessions/{session_id}/messages` 和 `DELETE /api/sessions/{session_id}`。

## 长期记忆形成

明确的持续偏好、长期目标、记住要求及纠正即时检查，普通对话每新增 6 轮整理一次
（`MEMORY_BATCH_INTERVAL=6`），允许零新增。相同含义合并或保持不变；自动薄弱点需要
两道不同题目的高置信、带课程引文的错误作答证据。提问、资料缺失和助手失败不属于用户能力证据。
用户自述与练习观察分别标记；最终决策 trace 显示新增、合并、删除、忽略数量和原因。

未验证的历史记忆不会被普通新提问自动改写；需要人工审核或用户明确纠正。
详见[长期记忆形成与人工核对](docs/memory-formation-review.md)。

## 自动检索评测与人工审核

首页点击“检索评测与审核”，或访问 `http://localhost:8000/evaluation.html`。
支持从资料自动出题、真实检索、独立模型预标注、引文高亮、少量人工审核、断点续跑、
JSONL 导出及固定问题的对照运行。已有评测集支持追加 1–200 道题目计划，
保留原题与审核结果；出题时校验题干去重，重复自动重试最多 3 次。默认出题使用 `deepseek-v4-flash`，评审使用
`deepseek-v4-pro`，页面可改模型名，`.env` 可分别设置服务地址和密钥。
该模块评测检索证据，不改变对话流程的置信度规则；机器标签与人工标签分开统计。

建议先创建 4 个问题确认配置；详细步骤、指标口径与 CLI 见
[检索评测使用说明](docs/retrieval-evaluation.md)。

## 系统端到端评测

使用 `backend/evaluate_system.py` 选择多个已标注数据集或 JSONL 导出文件，执行真实
Agent 单轮问答流程，独立评审答案的相关性、证据支持与关键点覆盖。
支持按人工审核/测试分组筛选、共享问题去重、断点续跑、分数据集报告及结果导出。
详见[系统端到端评测说明](docs/system-evaluation.md)。

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
