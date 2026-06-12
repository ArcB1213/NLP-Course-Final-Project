# API

## Health

```http
GET /api/health
```

## Upload Documents

```http
POST /api/documents/upload
Content-Type: multipart/form-data
```

Field:

```text
files: UploadFile[]
```

## List Documents

```http
GET /api/documents
```

## Delete Document

```http
DELETE /api/documents/{document_id}
```

## Search

```http
POST /api/search
Content-Type: application/json

{
  "query": "什么是条件随机场？",
  "top_k": 5,
  "profile": "qa"
}
```

返回结果包含 `vector_score`、`bm25_score`、`rerank_score` 和 `final_score`。开启 reranker 时，最终顺序优先由 `rerank_score` 决定。

`profile` 可选：

- `default`：基础混合检索。
- `qa`：偏精确问答检索。
- `summary`：扩展 query 后做广覆盖总结检索。

## Chat

```http
POST /api/chat
Content-Type: application/json

{
  "query": "什么是条件随机场？",
  "task_type": "qa",
  "use_pro_model": false,
  "top_k": 5
}
```

总结请求示例：

```http
POST /api/chat
Content-Type: application/json

{
  "query": "总结一下中文分词的主要方法",
  "task_type": "summary",
  "use_pro_model": false
}
```

自动路由出题请求示例：

```http
POST /api/chat
Content-Type: application/json

{
  "query": "围绕 HMM 和 CRF 的区别出三道题",
  "task_type": "auto",
  "use_pro_model": false
}
```

批改请求示例：

```http
POST /api/chat
Content-Type: application/json

{
  "query": "题目：CRF 和 HMM 有什么区别？\n我的答案：HMM 是生成式模型，CRF 是判别式模型。",
  "task_type": "grade"
}
```

`task_type` 支持：

- `auto`
- `qa`
- `summary`
- `quiz`
- `grade`
