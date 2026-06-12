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
  "top_k": 5
}
```

返回结果包含 `vector_score`、`bm25_score`、`rerank_score` 和 `final_score`。开启 reranker 时，最终顺序优先由 `rerank_score` 决定。

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
