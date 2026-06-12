# RAG + Agent 中文课程学习助手实现计划

## Summary

基于现有设计文档，从空仓库开始实现一个可演示的课程期末项目。整体按“先闭环、再增强、最后展示”的路线推进：先完成后端最小 RAG 问答闭环，再加入混合检索和拒答机制，再实现任务型 Agent，最后补前端和演示文档。

默认技术选择：后端使用 `FastAPI + SQLAlchemy + SQLite + FAISS + sentence-transformers + httpx`；前端使用 `Vue 3 + Vite + Axios + Element Plus`；第一版 `reranker` 做成可配置能力，默认可先关闭，避免影响 MVP 进度。

## Implementation Phases

### 阶段 0：项目骨架与配置

实现内容：

- 创建 `backend/`、`frontend/`、`docs/` 目录结构。
- 后端建立 `app.main`、`config.py`、`dependencies.py`、`run.py`。
- 使用 `.env.example` 定义 DeepSeek、embedding、路径、chunk、top_k、reranker 等配置。
- 建立 SQLite 数据目录、上传目录、索引目录。
- 加入基础 `README.md` 启动说明。

验收标准：

- `cd backend && python run.py` 能启动 FastAPI。
- `GET /api/health` 返回 `{"status":"ok"}`。
- `.env.example` 中包含 DeepSeek API、模型名、索引路径、chunk 参数，代码中不硬编码 DeepSeek 模型名和 API Key。

### 阶段 1：文档上传、解析、分块与入库

实现内容：

- 实现 `Document`、`Chunk` 数据表和 CRUD。
- 实现 PDF / Markdown / TXT loader。
- 实现 `normalize_text` 和 `ChineseTextSplitter`。
- 实现 `POST /api/documents/upload`、`GET /api/documents`、`DELETE /api/documents/{id}`。
- 上传后同步解析、分块、保存 chunk 元数据。

验收标准：

- 能上传 `.pdf`、`.md`、`.txt` 三类文件。
- 数据库中能看到 document 记录和对应 chunks。
- PDF chunk 保留 `page`；Markdown chunk 保留 `heading`；TXT 可正常处理 `utf-8` 和 `gbk`。
- 删除文档后，对应 document 与 chunks 被删除。
- 至少用一个中文 NLP 示例文件验证：上传成功后 `chunk_count > 0`。

### 阶段 2：最小 RAG 问答闭环

实现内容：

- 实现 `Embedder`，使用 `BAAI/bge-base-zh-v1.5`，query 使用 BGE 推荐前缀，embedding 做归一化。
- 实现 `FaissVectorStore`，维护 `id_mapping`，支持保存和加载。
- 文档上传后生成 embedding 并更新 FAISS。
- 实现 `DeepSeekClient`，用 OpenAI-compatible chat completions 风格封装。
- 实现 QA prompt、`QATool` 和 `POST /api/chat` 的问答路径。
- 返回答案时由后端附带检索到的 sources，不让 LLM 编引用。

验收标准：

- 上传资料后，调用 `/api/chat` 提问“什么是条件随机场？”能返回基于资料的回答。
- `ChatResponse` 至少包含 `task_type`、`answer`、`sources`、`confidence`、`message`。
- `sources` 中包含 `chunk_id`、`document_id`、`source_file`、`page` 或 `heading`、`final_score`。
- DeepSeek API Key 缺失时启动或调用阶段有明确错误提示。
- 代码中 DeepSeek base URL 和 model 全部来自配置。

### 阶段 3：增强检索、调试接口与拒答

实现内容：

- 实现 `BM25Store`，使用 `jieba + rank-bm25`。
- 实现 `HybridRetriever`：向量 top_k + BM25 top_k，去重、归一化、按 `0.6 vector + 0.4 bm25` 融合。
- 实现 `POST /api/search`，便于调试检索结果。
- 实现 `estimate_confidence` 和低置信度 QA 拒答。
- 实现可选 `Reranker` 接口；第一版可以保留开关，默认 `ENABLE_RERANKER=true`。

验收标准：

- `/api/search` 输入查询后能返回排序后的 chunk 列表和分数。
- 对资料中存在的问题，top results 能命中相关段落。
- 对明显无关问题，QA 返回“课程资料中未找到充分依据”类提示，`confidence=low`，`message=low_retrieval_confidence`。
- 删除文档后重建 FAISS 和 BM25，搜索结果不再包含已删除文档。
- 关闭 reranker 时系统仍可完整运行。

### 阶段 4：任务型 Agent 与多工具能力

实现内容：

- 实现 `RouterAgent`，支持 `auto / qa / summary / quiz / grade`。
- 实现 `SummaryTool`、`QuizTool`、`GradingTool`。
- 实现 `AgentService`，统一调度 Router 和各工具。
- `/api/chat` 改为统一 Agent 入口。
- Summary / Quiz / Grade 均基于检索片段构造 prompt，并返回 sources。

验收标准：

- `task_type=auto` 时，以下输入能路由到正确任务：
  - “总结一下中文分词的主要方法” → `summary`
  - “围绕 HMM 和 CRF 的区别出三道题” → `quiz`
  - “我的答案……帮我看看对不对” → `grade`
  - 普通知识问题 → `qa`
- 显式传入 `task_type` 时优先使用用户指定任务。
- 总结输出结构化提纲；出题输出题目、答案、考察点；批改输出总体评价、答对点、需补充点、参考答案。
- 所有任务返回引用来源，且来源来自检索结果而非 LLM 自编。

### 阶段 5：Vue 前端展示

实现内容：

- 创建 Vue 3 + Vite 前端。
- 实现 `client.ts` 封装上传、文档列表、删除、chat、search。
- 单页布局：
  - 左侧文档管理：上传、列表、删除。
  - 中间对话区：任务模式、Pro 开关、输入、回答。
  - 右侧引用来源：文件名、页码、heading、原文片段。
- 使用 Element Plus 做基础 UI，不做复杂视觉设计。

验收标准：

- 前端可上传 PDF / Markdown / TXT。
- 上传后文档列表刷新，能删除文档。
- 用户可选择自动、问答、总结、出题、批改任务。
- 提交问题后能展示回答、置信度和引用来源。
- 后端 `8000`、前端 Vite 页面能联调通过。

### 阶段 6：测试、文档与课程演示材料

实现内容：

- 后端补充单元测试：
  - loader 测试；
  - splitter 测试；
  - router 测试；
  - confidence 测试；
  - API smoke test。
- 准备 `docs/api.md`，记录主要接口请求和返回。
- README 增加安装、启动、配置、示例验收用例。
- 准备一组中文 NLP 课程测试资料和四个演示问题。

验收标准：

- 后端测试命令能通过。
- README 中能按步骤复现实验。
- 能完成四个最终验收用例：
  - 资料问答：CRF 定义 + 页码或标题来源。
  - 知识点总结：中文分词结构化提纲。
  - 练习题生成：HMM 和 CRF 三道题、答案、考察点。
  - 答案批改：指出答案基本正确并补充条件概率建模、特征使用等要点。

## Public Interfaces And Types

后端公开接口固定为：

- `GET /api/health`
- `POST /api/documents/upload`
- `GET /api/documents`
- `DELETE /api/documents/{document_id}`
- `POST /api/search`
- `POST /api/chat`

核心响应类型固定为：

- `DocumentRead`
- `RetrievedChunk`
- `ChatRequest`
- `ChatResponse`

`ChatRequest.task_type` 只接受：

- `auto`
- `qa`
- `summary`
- `quiz`
- `grade`

## Test Plan

- 单元测试覆盖文本解析、中文分块、任务路由、置信度判断。
- API 测试覆盖健康检查、上传、文档列表、搜索、聊天。
- 集成测试使用一个小型中文课程资料文件，验证上传到问答的完整链路。
- 前端手工验收覆盖上传、删除、任务选择、回答展示、引用展示。

## Assumptions

- 仓库当前没有现成代码，按新项目从零搭建。
- 第一版不做 OCR、多轮记忆、用户系统、错题本和多课程切换。
- `reranker` 保留接口和配置开关，但不作为 MVP 必须项。
- 文档上传后同步构建索引；大文件后台任务留到后续扩展。
- 前端以可演示、可操作为目标，不追求复杂视觉效果。
