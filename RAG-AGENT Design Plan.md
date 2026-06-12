# 基于 RAG 与任务型 Agent 的中文课程学习助手：代码设计方案

## 1. 项目目标

实现一个面向中文课程资料的学习辅助系统。系统支持上传 PDF、Markdown、TXT 文件，自动构建课程知识库，并通过 RAG 与任务型 Agent 完成以下功能：

1. 基于课程资料问答；
2. 知识点总结；
3. 练习题生成；
4. 学生答案批改与反馈；
5. 返回答案时给出引用来源；
6. 当资料中没有充分依据时拒答或提示依据不足。

系统使用 DeepSeek API 作为 LLM 生成模型。模型名称、API 地址和 API Key 均通过环境变量配置，方便在 `deepseek-v4-flash` 和 `deepseek-v4-pro` 之间切换。

前端可以使用 Vue 实现简单页面，但项目重点在后端 RAG 与 Agent 流程。

---

## 2. 技术栈建议

### 2.1 后端

- Python 3.10+
- FastAPI
- Uvicorn
- Pydantic
- SQLAlchemy 或 SQLModel
- PyMuPDF：解析 PDF
- markdown / mistune：解析 Markdown
- sentence-transformers：中文 embedding
- FAISS：向量检索
- rank-bm25：关键词检索
- FlagEmbedding 或 sentence-transformers CrossEncoder：reranker，可选但推荐
- httpx：调用 DeepSeek API
- python-dotenv：读取环境变量

### 2.2 前端

- Vue 3
- Vite
- Axios
- Element Plus 或 Naive UI，任选一个，够用即可

### 2.3 默认模型配置

Embedding 模型：

```text
BAAI/bge-base-zh-v1.5
```

如果显存或机器性能有限，可以改成：

```text
BAAI/bge-small-zh-v1.5
```

Reranker 模型：

```text
BAAI/bge-reranker-base
```

如果实现时间有限，可以先不接 reranker，只使用 BM25 + 向量检索融合。

LLM：

```text
DeepSeek API
```

模型名不要在代码中写死，统一从 `.env` 中读取。

---

## 3. 总体架构

系统分为五层：

```text
用户交互层
  ↓
FastAPI 接口层
  ↓
任务路由 Agent 层
  ↓
RAG 工具层
  ├── 文档解析工具
  ├── 分块工具
  ├── 向量检索工具
  ├── BM25 检索工具
  ├── Rerank 工具
  ├── 问答工具
  ├── 总结工具
  ├── 出题工具
  └── 批改工具
  ↓
数据存储层
  ├── 原始文件
  ├── 文档元数据
  ├── chunk 元数据
  ├── FAISS 向量索引
  └── BM25 索引
```

核心运行流程：

```text
上传文件
  ↓
解析文本
  ↓
中文分块
  ↓
保存 chunk 与元数据
  ↓
生成 embedding
  ↓
构建 FAISS 索引
  ↓
构建 BM25 索引
  ↓
用户提问
  ↓
Router 判断任务类型
  ↓
检索相关片段
  ↓
组织 prompt
  ↓
调用 DeepSeek API
  ↓
返回答案 + 引用来源
```

---

## 4. 目录结构设计

建议项目目录如下：

```text
course-rag-agent/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── dependencies.py
│   │   │
│   │   ├── api/
│   │   │   ├── documents.py
│   │   │   ├── chat.py
│   │   │   ├── search.py
│   │   │   └── health.py
│   │   │
│   │   ├── core/
│   │   │   ├── llm_client.py
│   │   │   ├── router_agent.py
│   │   │   ├── prompts.py
│   │   │   └── schemas.py
│   │   │
│   │   ├── document/
│   │   │   ├── loaders.py
│   │   │   ├── splitter.py
│   │   │   ├── normalizer.py
│   │   │   └── ingestion.py
│   │   │
│   │   ├── retrieval/
│   │   │   ├── embedder.py
│   │   │   ├── vector_store.py
│   │   │   ├── bm25_store.py
│   │   │   ├── reranker.py
│   │   │   └── retriever.py
│   │   │
│   │   ├── tools/
│   │   │   ├── qa_tool.py
│   │   │   ├── summary_tool.py
│   │   │   ├── quiz_tool.py
│   │   │   └── grading_tool.py
│   │   │
│   │   ├── db/
│   │   │   ├── database.py
│   │   │   ├── models.py
│   │   │   └── crud.py
│   │   │
│   │   └── utils/
│   │       ├── file_utils.py
│   │       └── text_utils.py
│   │
│   ├── data/
│   │   ├── uploads/
│   │   ├── indexes/
│   │   └── sqlite/
│   │
│   ├── requirements.txt
│   ├── .env.example
│   └── run.py
│
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   └── client.ts
│   │   ├── components/
│   │   │   ├── FileUpload.vue
│   │   │   ├── ChatPanel.vue
│   │   │   ├── SourcePanel.vue
│   │   │   └── TaskSelector.vue
│   │   ├── views/
│   │   │   └── Home.vue
│   │   ├── App.vue
│   │   └── main.ts
│   ├── package.json
│   └── vite.config.ts
│
├── README.md
└── docs/
    ├── design.md
    └── api.md
```

---

## 5. 环境变量设计

后端 `.env`：

```env
APP_NAME=CourseRAGAgent
APP_ENV=dev

DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_PRO_MODEL=deepseek-v4-pro

EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base

UPLOAD_DIR=./data/uploads
INDEX_DIR=./data/indexes
SQLITE_PATH=./data/sqlite/app.db

CHUNK_SIZE=500
CHUNK_OVERLAP=80
VECTOR_TOP_K=10
BM25_TOP_K=10
FINAL_TOP_K=5

ENABLE_RERANKER=true
```

注意：

1. 不要把 API Key 写死到代码中；
2. DeepSeek 的 base_url 和 model 名称必须可配置；
3. Flash / Pro 模型通过请求参数切换，默认使用 Flash；
4. 对于批改、复杂总结等任务，可以允许用户选择 Pro。

---

## 6. 数据模型设计

### 6.1 Document

表示用户上传的一个文档。

字段：

```python
class Document:
    id: str
    filename: str
    file_type: str  # pdf / md / txt
    file_path: str
    title: str | None
    created_at: datetime
    status: str  # uploaded / parsed / indexed / failed
    chunk_count: int
```

### 6.2 Chunk

表示一个可检索文本块。

```python
class Chunk:
    id: str
    document_id: str
    chunk_index: int
    text: str
    source_file: str
    page: int | None
    heading: str | None
    start_char: int | None
    end_char: int | None
    created_at: datetime
```

### 6.3 RetrievedChunk

检索时返回的片段。

```python
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    source_file: str
    page: int | None
    heading: str | None
    vector_score: float | None
    bm25_score: float | None
    rerank_score: float | None
    final_score: float
```

### 6.4 ChatRequest

```python
class ChatRequest:
    query: str
    task_type: str | None  # auto / qa / summary / quiz / grade
    use_pro_model: bool = False
    top_k: int = 5
    extra_context: dict | None = None
```

### 6.5 ChatResponse

```python
class ChatResponse:
    task_type: str
    answer: str
    sources: list[RetrievedChunk]
    confidence: str  # high / medium / low
    message: str | None
```

---

## 7. 文件解析模块设计

文件解析模块位于：

```text
app/document/loaders.py
```

提供统一接口：

```python
class BaseLoader:
    def load(self, file_path: str) -> list[RawPage]:
        raise NotImplementedError
```

RawPage 数据结构：

```python
class RawPage:
    text: str
    page: int | None
    metadata: dict
```

### 7.1 PDFLoader

使用 PyMuPDF。

功能：

1. 逐页读取 PDF；
2. 保留页码；
3. 删除多余空行；
4. 如果页面文本为空，暂时不做 OCR，直接跳过或提示该页不可解析。

```python
class PDFLoader(BaseLoader):
    def load(self, file_path: str) -> list[RawPage]:
        ...
```

第一版不做 OCR。原因是 OCR 会显著增加复杂度，而且课程项目不一定需要。

### 7.2 MarkdownLoader

功能：

1. 读取 `.md`；
2. 保留标题结构；
3. 可以根据 `#`、`##`、`###` 识别 heading。

```python
class MarkdownLoader(BaseLoader):
    def load(self, file_path: str) -> list[RawPage]:
        ...
```

### 7.3 TxtLoader

功能：

1. 读取 `.txt`；
2. 自动尝试 `utf-8`；
3. 失败时尝试 `gbk`；
4. 按段落保存。

```python
class TxtLoader(BaseLoader):
    def load(self, file_path: str) -> list[RawPage]:
        ...
```

### 7.4 LoaderFactory

```python
def get_loader(file_type: str) -> BaseLoader:
    if file_type == "pdf":
        return PDFLoader()
    if file_type in ["md", "markdown"]:
        return MarkdownLoader()
    if file_type == "txt":
        return TxtLoader()
    raise ValueError("Unsupported file type")
```

---

## 8. 文本清洗模块设计

文件：

```text
app/document/normalizer.py
```

主要函数：

```python
def normalize_text(text: str) -> str:
    ...
```

处理逻辑：

1. 统一换行符；
2. 删除连续空行；
3. 删除明显页眉页脚，可先做简单规则；
4. 将全角空格转为普通空格；
5. 保留中文标点；
6. 保留公式、英文术语和模型名，例如 HMM、CRF、BERT。

不要过度清洗，尤其不要删除英文、数字和标点，因为中文 NLP 课程资料中常有中英文术语混合。

---

## 9. 中文分块模块设计

文件：

```text
app/document/splitter.py
```

核心类：

```python
class ChineseTextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        ...

    def split_pages(self, pages: list[RawPage], document_id: str) -> list[ChunkCreate]:
        ...
```

### 9.1 分块策略

优先级：

1. Markdown 按标题切；
2. PDF/TXT 按段落切；
3. 段落过长时按中文句号、问号、感叹号、分号切；
4. 最终保证 chunk 大约 300 到 600 中文字；
5. 相邻 chunk 保留 80 字 overlap。

中文句子分隔符：

```python
SENTENCE_SEPARATORS = ["。", "？", "！", "；", "\n"]
```

### 9.2 Chunk 元数据

每个 chunk 保存：

```python
{
    "document_id": "...",
    "chunk_index": 0,
    "text": "...",
    "source_file": "第3章_序列标注.pdf",
    "page": 12,
    "heading": "条件随机场",
}
```

---

## 10. 索引构建模块设计

文件：

```text
app/document/ingestion.py
```

核心流程：

```python
class IngestionService:
    def ingest_file(self, file_path: str, filename: str) -> Document:
        # 1. 保存 Document 记录
        # 2. 根据文件类型选择 loader
        # 3. 解析文本
        # 4. 清洗文本
        # 5. 分块
        # 6. 保存 Chunk 到数据库
        # 7. 更新 FAISS 和 BM25 索引
        # 8. 更新文档状态
        ...
```

建议第一版上传后立即同步构建索引。文件较大时再改为后台任务。

---

## 11. Embedding 模块设计

文件：

```text
app/retrieval/embedder.py
```

接口：

```python
class Embedder:
    def __init__(self, model_name: str):
        ...

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        ...

    def encode_query(self, query: str) -> np.ndarray:
        ...
```

使用 sentence-transformers：

```python
from sentence_transformers import SentenceTransformer
```

注意：

1. bge 模型通常建议 query 加前缀，例如：`为这个句子生成表示以用于检索相关文章：{query}`；
2. corpus 文本可以直接编码；
3. embedding 需要归一化，便于使用内积或 cosine similarity。

---

## 12. 向量索引模块设计

文件：

```text
app/retrieval/vector_store.py
```

使用 FAISS。

核心接口：

```python
class FaissVectorStore:
    def add(self, chunk_ids: list[str], embeddings: np.ndarray) -> None:
        ...

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[VectorSearchResult]:
        ...

    def save(self) -> None:
        ...

    def load(self) -> None:
        ...
```

需要维护：

```python
index: faiss.Index
id_mapping: list[str]  # faiss 内部 index 到 chunk_id 的映射
```

推荐使用：

```python
faiss.IndexFlatIP
```

前提是 embedding 做 L2 normalize。

---

## 13. BM25 检索模块设计

文件：

```text
app/retrieval/bm25_store.py
```

使用 `rank_bm25`。

中文 BM25 需要分词。可以使用：

```text
jieba
```

核心接口：

```python
class BM25Store:
    def build(self, chunks: list[Chunk]) -> None:
        ...

    def search(self, query: str, top_k: int) -> list[BM25SearchResult]:
        ...
```

分词函数：

```python
def tokenize_zh(text: str) -> list[str]:
    return list(jieba.cut(text))
```

注意保留英文术语，可以在分词后过滤空白，但不要删除英文 token。

---

## 14. Reranker 模块设计

文件：

```text
app/retrieval/reranker.py
```

核心接口：

```python
class Reranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        ...
```

实现方案：

1. 输入 query 和候选 chunk；
2. 构造 pair：`[query, chunk.text]`；
3. 使用 reranker 模型打分；
4. 按分数排序；
5. 返回 top_k。

如果环境性能不足，可以在配置中关闭：

```env
ENABLE_RERANKER=false
```

关闭时直接使用融合分数排序。

---

## 15. 混合检索模块设计

文件：

```text
app/retrieval/retriever.py
```

这是 RAG 中最关键的检索服务。

```python
class HybridRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        ...
```

流程：

```text
query
  ↓
向量检索 top 10
  +
BM25 检索 top 10
  ↓
合并去重
  ↓
分数归一化
  ↓
加权融合
  ↓
reranker 重排序
  ↓
返回 top 5
```

融合分数建议：

```python
final_score = 0.6 * vector_score_norm + 0.4 * bm25_score_norm
```

如果有 reranker，则最终排序优先用 rerank_score。

伪代码：

```python
def retrieve(self, query: str, top_k: int = 5):
    vector_results = self.vector_store.search(
        self.embedder.encode_query(query),
        top_k=settings.VECTOR_TOP_K
    )

    bm25_results = self.bm25_store.search(
        query,
        top_k=settings.BM25_TOP_K
    )

    merged = merge_and_normalize(vector_results, bm25_results)

    if settings.ENABLE_RERANKER:
        return self.reranker.rerank(query, merged, top_k)

    return sorted(merged, key=lambda x: x.final_score, reverse=True)[:top_k]
```

---

## 16. DeepSeek LLM Client 设计

文件：

```text
app/core/llm_client.py
```

使用 OpenAI-compatible chat completions 风格封装，避免业务代码直接依赖具体 API 格式。

```python
class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str, default_model: str):
        ...

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        ...
```

调用示例：

```python
messages = [
    {"role": "system", "content": "你是一个中文课程学习助手。"},
    {"role": "user", "content": prompt},
]
answer = await llm.chat(messages, temperature=0.2)
```

错误处理：

1. API Key 缺失，启动时报错；
2. API 超时，返回友好提示；
3. 模型响应为空，返回失败信息；
4. 对 DeepSeek Flash / Pro 的模型名只从配置读取，不写死。

---

## 17. Prompt 设计

文件：

```text
app/core/prompts.py
```

统一管理 prompt，避免散落在代码中。

### 17.1 QA Prompt

```python
QA_SYSTEM_PROMPT = """
你是一个严谨的中文课程学习助手。
你必须优先根据给定的课程资料回答问题。
如果资料中没有充分依据，请明确说明“课程资料中未找到充分依据”，不要编造。
回答要清晰、简洁，适合学生复习使用。
"""
```

用户 prompt：

```python
def build_qa_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = format_chunks(chunks)
    return f"""
【课程资料片段】
{context}

【用户问题】
{query}

【回答要求】
1. 先直接回答问题。
2. 再给出必要解释。
3. 如果资料依据不足，请说明依据不足。
4. 不要编造课程资料中没有的信息。
"""
```

### 17.2 Summary Prompt

```python
def build_summary_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    return f"""
请根据以下课程资料，完成用户要求的知识点总结。

【课程资料】
{format_chunks(chunks)}

【用户要求】
{query}

【输出格式】
一、核心概念
二、主要方法或知识点
三、方法之间的关系
四、易混淆点
五、复习建议
"""
```

### 17.3 Quiz Prompt

```python
def build_quiz_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    return f"""
请根据以下课程资料生成练习题。

【课程资料】
{format_chunks(chunks)}

【用户要求】
{query}

【要求】
1. 题目必须基于资料内容。
2. 每道题给出参考答案。
3. 每道题标注考察知识点。
4. 如果用户没有指定题型，默认生成判断题、选择题、简答题各一道。
"""
```

### 17.4 Grading Prompt

```python
def build_grading_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    return f"""
你是一个中文课程助教。请根据课程资料，对学生答案进行学习反馈。

【课程资料】
{format_chunks(chunks)}

【学生提交内容】
{query}

【输出格式】
总体评价：基本正确 / 部分正确 / 存在明显问题
答对的点：
需要补充或修改的点：
建议修改后的参考答案：
相关资料依据：
"""
```

---

## 18. 任务路由 Agent 设计

文件：

```text
app/core/router_agent.py
```

任务类型：

```python
class TaskType(str, Enum):
    QA = "qa"
    SUMMARY = "summary"
    QUIZ = "quiz"
    GRADE = "grade"
```

第一版使用规则路由：

```python
class RouterAgent:
    def route(self, query: str, user_task_type: str | None = None) -> TaskType:
        if user_task_type and user_task_type != "auto":
            return TaskType(user_task_type)

        q = query.strip()

        if any(word in q for word in ["总结", "梳理", "归纳", "提纲", "复习"]):
            return TaskType.SUMMARY

        if any(word in q for word in ["出题", "练习题", "选择题", "判断题", "简答题", "考考我"]):
            return TaskType.QUIZ

        if any(word in q for word in ["我的答案", "批改", "对不对", "帮我看看", "评分", "评价一下"]):
            return TaskType.GRADE

        return TaskType.QA
```

可以保留 LLM Router 的扩展接口，但第一版不强制实现。

---

## 19. 工具层设计

### 19.1 QATool

文件：

```text
app/tools/qa_tool.py
```

```python
class QATool:
    def __init__(self, retriever: HybridRetriever, llm: DeepSeekClient):
        ...

    async def run(self, query: str, use_pro_model: bool = False) -> ChatResponse:
        chunks = self.retriever.retrieve(query)
        if is_low_confidence(chunks):
            return low_confidence_response(chunks)

        prompt = build_qa_prompt(query, chunks)
        answer = await self.llm.chat(messages=[...], model=select_model(use_pro_model))
        return ChatResponse(...)
```

### 19.2 SummaryTool

```python
class SummaryTool:
    async def run(self, query: str, use_pro_model: bool = False) -> ChatResponse:
        chunks = self.retriever.retrieve(query, top_k=8)
        prompt = build_summary_prompt(query, chunks)
        answer = await self.llm.chat(...)
        return ChatResponse(...)
```

### 19.3 QuizTool

```python
class QuizTool:
    async def run(self, query: str, use_pro_model: bool = False) -> ChatResponse:
        chunks = self.retriever.retrieve(query, top_k=6)
        prompt = build_quiz_prompt(query, chunks)
        answer = await self.llm.chat(...)
        return ChatResponse(...)
```

### 19.4 GradingTool

```python
class GradingTool:
    async def run(self, query: str, use_pro_model: bool = True) -> ChatResponse:
        chunks = self.retriever.retrieve(query, top_k=6)
        prompt = build_grading_prompt(query, chunks)
        answer = await self.llm.chat(...)
        return ChatResponse(...)
```

批改任务建议默认使用 Pro 模型，因为需要更复杂的判断。

---

## 20. Agent Orchestrator 设计

可以在 `chat.py` API 内直接组织，也可以单独写一个服务：

```text
app/core/agent_service.py
```

```python
class AgentService:
    def __init__(
        self,
        router: RouterAgent,
        qa_tool: QATool,
        summary_tool: SummaryTool,
        quiz_tool: QuizTool,
        grading_tool: GradingTool,
    ):
        ...

    async def handle(self, request: ChatRequest) -> ChatResponse:
        task_type = self.router.route(request.query, request.task_type)

        if task_type == TaskType.QA:
            return await self.qa_tool.run(request.query, request.use_pro_model)

        if task_type == TaskType.SUMMARY:
            return await self.summary_tool.run(request.query, request.use_pro_model)

        if task_type == TaskType.QUIZ:
            return await self.quiz_tool.run(request.query, request.use_pro_model)

        if task_type == TaskType.GRADE:
            return await self.grading_tool.run(request.query, request.use_pro_model)

        raise ValueError("Unsupported task type")
```

---

## 21. 拒答与置信度设计

实现一个简单置信度判断函数：

```python
def estimate_confidence(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "low"

    top_score = chunks[0].final_score

    if top_score >= 0.75:
        return "high"
    if top_score >= 0.45:
        return "medium"
    return "low"
```

如果使用 reranker，可用 rerank_score 判断。

对于低置信度 QA：

```python
if confidence == "low":
    return ChatResponse(
        task_type="qa",
        answer="课程资料中未找到与该问题充分相关的内容。建议补充资料，或换一种更具体的问法。",
        sources=chunks,
        confidence="low",
        message="low_retrieval_confidence"
    )
```

对于总结、出题、批改任务，可以不直接拒答，但需要在回答中说明资料依据不足。

---

## 22. FastAPI 接口设计

### 22.1 健康检查

```http
GET /api/health
```

返回：

```json
{
  "status": "ok"
}
```

### 22.2 上传文档

```http
POST /api/documents/upload
```

表单：

```text
files: UploadFile[]
```

返回：

```json
{
  "documents": [
    {
      "id": "doc_xxx",
      "filename": "第3章_序列标注.pdf",
      "status": "indexed",
      "chunk_count": 42
    }
  ]
}
```

### 22.3 获取文档列表

```http
GET /api/documents
```

返回：

```json
[
  {
    "id": "doc_xxx",
    "filename": "第3章_序列标注.pdf",
    "file_type": "pdf",
    "status": "indexed",
    "chunk_count": 42,
    "created_at": "..."
  }
]
```

### 22.4 删除文档

```http
DELETE /api/documents/{document_id}
```

删除后需要重建索引，或者标记删除并重建索引。

第一版可以简单实现为：

1. 删除数据库中的 document 和 chunks；
2. 重新从剩余 chunks 构建 FAISS 和 BM25。

### 22.5 检索测试接口

```http
POST /api/search
```

请求：

```json
{
  "query": "什么是条件随机场？",
  "top_k": 5
}
```

返回：

```json
{
  "results": [
    {
      "chunk_id": "chunk_xxx",
      "text": "...",
      "source_file": "第3章.pdf",
      "page": 12,
      "heading": "条件随机场",
      "final_score": 0.82
    }
  ]
}
```

这个接口对调试 RAG 非常重要，建议保留。

### 22.6 Chat / Agent 主接口

```http
POST /api/chat
```

请求：

```json
{
  "query": "帮我总结一下中文分词的主要方法",
  "task_type": "auto",
  "use_pro_model": false,
  "top_k": 5
}
```

返回：

```json
{
  "task_type": "summary",
  "answer": "一、核心概念……",
  "confidence": "high",
  "sources": [
    {
      "chunk_id": "chunk_xxx",
      "text": "中文分词是……",
      "source_file": "第2章.pdf",
      "page": 5,
      "heading": "中文分词",
      "final_score": 0.89
    }
  ],
  "message": null
}
```

---

## 23. Vue 前端设计

前端够用即可，重点实现以下页面。

### 23.1 页面布局

单页面即可：

```text
左侧：文档管理
  - 上传 PDF / Markdown / TXT
  - 文档列表
  - 删除文档

中间：聊天区域
  - 任务模式选择：自动 / 问答 / 总结 / 出题 / 批改
  - 是否使用 Pro 模型
  - 输入框
  - 输出结果

右侧：引用来源
  - 展示 source chunks
  - 文件名
  - 页码
  - heading
  - 原文片段
```

### 23.2 组件

```text
FileUpload.vue
DocumentList.vue
TaskSelector.vue
ChatPanel.vue
SourcePanel.vue
```

### 23.3 前端 API 封装

文件：

```text
frontend/src/api/client.ts
```

包含：

```typescript
uploadDocuments(files: File[]): Promise<any>
getDocuments(): Promise<any>
deleteDocument(id: string): Promise<any>
chat(payload: ChatRequest): Promise<ChatResponse>
search(query: string): Promise<any>
```

---

## 24. 后端启动方式

`backend/run.py`：

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
```

启动：

```bash
cd backend
python run.py
```

前端启动：

```bash
cd frontend
npm install
npm run dev
```

---

## 25. requirements.txt 建议

```txt
fastapi
uvicorn[standard]
python-multipart
pydantic
pydantic-settings
python-dotenv
sqlalchemy
pymupdf
markdown
jieba
rank-bm25
numpy
faiss-cpu
sentence-transformers
httpx
```

如果实现 reranker：

```txt
FlagEmbedding
```

如果 FlagEmbedding 安装复杂，可以先用 sentence-transformers 的 CrossEncoder，或者暂时关闭 reranker。

---

## 26. 实现优先级

建议按以下顺序实现。

### 阶段一：最小 RAG 问答系统

1. FastAPI 项目搭建；
2. PDF / Markdown / TXT 上传；
3. 文本解析；
4. chunk 切分；
5. embedding；
6. FAISS 检索；
7. DeepSeek API 调用；
8. RAG 问答；
9. 返回引用来源。

这个阶段完成后，系统已经可用。

### 阶段二：增强检索

1. 加入 BM25；
2. 实现 BM25 + 向量混合检索；
3. 加入检索测试接口；
4. 可选加入 reranker；
5. 加入低置信度拒答。

### 阶段三：Agent 与多任务工具

1. 实现 RouterAgent；
2. 实现 SummaryTool；
3. 实现 QuizTool；
4. 实现 GradingTool；
5. 统一到 AgentService。

### 阶段四：前端展示

1. 文档上传；
2. 文档列表；
3. 对话框；
4. 任务模式选择；
5. 引用来源展示。

---

## 27. 关键实现注意事项

### 27.1 不要把 RAG 写成简单拼接

必须保留以下流程：

```text
检索 → 片段筛选 → prompt 构造 → LLM 生成 → 引用返回
```

### 27.2 引用来源必须来自 chunk 元数据

不要让 LLM 自己编引用。引用来源由后端根据检索结果生成。

### 27.3 LLM 不负责检索

DeepSeek 只负责基于给定 context 生成答案，不负责从文件中找内容。

### 27.4 Prompt 中要明确约束

每个任务 prompt 都要强调：

```text
请基于课程资料回答；
资料不足时说明依据不足；
不要编造。
```

### 27.5 DeepSeek 模型名可配置

代码中不要出现硬编码模型名，例如：

```python
model="deepseek-v4-flash"
```

应写成：

```python
model=settings.DEEPSEEK_MODEL
```

### 27.6 文档删除后需要重建索引

FAISS 不适合频繁删除单条向量。第一版可以在删除文档后重新构建全部索引。

### 27.7 Markdown 标题信息要保留

Markdown 文件中 `#`、`##`、`###` 对学习资料很重要，应保存为 heading metadata。

### 27.8 PDF 页码要保留

PDF chunk 要保存 page 字段，方便前端展示引用来源。

---

## 28. MVP 验收标准

项目最小可行版本应满足：

1. 能上传 PDF、Markdown、TXT；
2. 能解析文本并构建知识库；
3. 能针对资料内容回答问题；
4. 回答中能展示引用来源；
5. 能自动识别问答、总结、出题、批改任务；
6. 能调用 DeepSeek API 生成结果；
7. 有一个可操作的简单前端界面。

示例验收用例：

### 用例 1：资料问答

用户上传中文 NLP 课程资料后提问：

```text
什么是条件随机场？
```

系统应回答 CRF 的定义，并返回相关页码或标题。

### 用例 2：知识点总结

```text
总结一下中文分词的主要方法
```

系统应输出结构化提纲。

### 用例 3：生成练习题

```text
围绕 HMM 和 CRF 的区别出三道题
```

系统应生成题目、参考答案和考察点。

### 用例 4：答案批改

```text
题目：CRF 和 HMM 有什么区别？
我的答案：HMM 是生成式模型，CRF 是判别式模型。
```

系统应指出答案基本正确，并提示可以补充条件概率建模、特征使用等内容。

---

## 29. 后续可扩展功能

第一版完成后，可以继续扩展：

1. 多轮对话记忆；
2. 错题本；
3. 知识点标签；
4. 章节级摘要；
5. 文档内定位预览；
6. OCR 支持扫描版 PDF；
7. 用户学习进度管理；
8. 自动生成复习计划；
9. 支持 DOCX / PPTX；
10. 支持多课程知识库切换。

---

## 30. 总结

本项目的核心不是简单调用大模型，而是实现一个完整的中文课程学习应用系统。系统通过文档解析、中文分块、混合检索、RAG 生成和任务型 Agent 路由，将课程资料转化为可问答、可总结、可练习、可反馈的学习知识库。

推荐最先完成最小 RAG 问答闭环，再扩展 BM25、reranker、任务路由和多工具调用。前端只需实现文档上传、聊天交互和引用展示即可。
