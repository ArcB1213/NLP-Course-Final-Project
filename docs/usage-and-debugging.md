# 检索配置与接口调试

以下配置在 `backend/.env` 中设置；完整字段见 [配置示例](../backend/.env.example)。启动步骤见 [项目 README](../README.md)，请求与响应说明见 [API 文档](api.md)。

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

注意：上述向量/BM25 参数用于基础 `default` profile；当前 `qa` 和 `summary` 在 [检索实现](../backend/app/retrieval/retriever.py) 中使用各自配置，不能认为修改基础权重会同步改变所有任务。

## 查看 Agent 决策过程

如需查看 Agent 决策过程，可在请求中加入：

```json
{
  "extra_context": {
    "debug_agent_trace": true
  }
}
```

返回中的 `agent_trace.steps` 会记录 `route`、`plan`、`retrieve`、`judge_evidence`、`retrieve_retry`、`retrieve_round_1/2/3`、`summary_coverage_judge_*`、`summary_coverage_stop`、`tool_generate` 等步骤。

## PowerShell 调用示例

健康检查：

```powershell
Invoke-RestMethod http://localhost:8000/api/health
```

上传文件（将示例路径替换为实际文件路径）：

```powershell
curl.exe -F "files=@C:\path\to\test.md" http://localhost:8000/api/documents/upload
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


