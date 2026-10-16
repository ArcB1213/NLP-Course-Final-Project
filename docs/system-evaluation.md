# 系统端到端评测

入口：`backend/evaluate_system.py`。不需要启动 HTTP 后端；脚本直接调用与聊天 API
相同的 `AgentService.handle`，执行路由、规划、检索、证据判断/补检索和工具答案生成。
默认 `task_type=auto`；可以用 `--task-type qa` 固定问答路径。

本工具评测**独立单轮问题**。关闭会话记忆，不创建会话、不写入长期记忆，
因此不评测多轮记忆能力、HTTP/SSE 传输或浏览器性能。系统耗时不包含外部评审耗时。

## 最短使用步骤

在项目根目录 PowerShell：

```powershell
conda activate course-rag-agent

# 查看已有标注数据集 ID
python backend/evaluate_system.py list

# 选择一个或多个标注数据集（用实际 ID 替换 DATASET_A / DATASET_B）
python backend/evaluate_system.py create --datasets DATASET_A DATASET_B --name baseline

# create 输出新的 RUN_ID，与标注数据集 ID 不同
python backend/evaluate_system.py run RUN_ID

# 查看报告；评审失败的样本可以重试
python backend/evaluate_system.py report RUN_ID
python backend/evaluate_system.py run RUN_ID --retry-errors

# 导出报告和逐题结果
python backend/evaluate_system.py export RUN_ID --output output/system-evaluation
```

开始时建议加 `--limit 3` 做小规模联调。一次系统回答可能调用模型多次，
此外每个有效答案还会调用一次评审模型；运行产生正常模型费用。

也可以选择导出的 JSONL 文件，路径相对于运行命令时所在目录：

```powershell
python backend/evaluate_system.py create --files evaluation-A.jsonl evaluation-B.jsonl --name from-files
```

`--datasets` 与 `--files` 可同时使用，但同一个数据集不要重复选择。
文件不导入或修改原标注数据库，脚本保存本次选中样本的快照。

## 选择与模型选项

创建运行支持：

| 参数 | 含义 |
| --- | --- |
| `--reviewed-only` | 只选人工接受或修正的样本 |
| `--split test` | 只选测试分组；也可选 calibration，默认 all |
| `--limit 3` | 按输入数据集和样本顺序最多选 3 个独立 case，不是随机抽样 |
| `--task-type auto/qa` | 自动路由或固定问答，默认 auto |
| `--top-k 5` | 覆盖系统请求的召回数量；不指定则使用系统默认 |
| `--use-pro-model` | 与聊天中的 Pro 选项相同；路由也可能按任务使用 Pro |
| `--judge-model MODEL` | 答案评审模型，默认 EVAL_JUDGE_MODEL |

默认只选已有至少一次有效机器评审或人工标签的问题，排除人工标为 excluded 的问题。
尚未出题、没有有效标签的问题不会执行，会在 skipped 中记录。
人工标签优先于机器标签，但这些**历史标签描述旧检索结果**，不作为新答案的正确性真值。
生成的参考关键点仍是机器产物，即使旧召回标签经过人工确认，也不意味着参考答案已被人工认证。

同一 case_id（例如原评测集与对照运行）只执行一次，保留全部数据集成员关系。
总体以独立 case 计数；各数据集报告分别包含其成员，所以各分组数量不能直接相加。
共享 case_id 的题目内容不一致会拒绝创建，避免混淆不同版本。
不同 case_id 的相似题目不会做语义去重。

答案生成模型、检索设置使用项目现有 .env；评审地址/密钥使用
EVAL_JUDGE_BASE_URL / EVAL_JUDGE_API_KEY，空值回退到 DEEPSEEK 配置。
DeepSeek 评审沿用 EVAL_JUDGE_THINKING，默认 false。系统本身的聊天模型行为保持当前配置。

## 报告与评分口径

每次运行保存：
- 问题、参考来源及关键点、成员数据集、历史标注来源。
- 实际系统回答、最终 sources、confidence、完整 AgentTrace。
- 回答生成耗时、评审耗时、模型评审理由及问题列表。
- 系统代码/资料指纹、有关设置、评审模型与提示词版本。

评审仅接收问题、答案、当前原文证据及明确标为机器生成的参考材料，
不接收当前 confidence、检索分数或历史标签，减少这些分数对评审的影响。

评分均为 LLM 辅助评价，**不是经过人工校准的正确率**：

| 指标 | 定义 |
| --- | --- |
| relevance | 0=未回应；1=部分回应；2=直接回应或清楚解释无法回答 |
| groundedness | 0=关键断言无依据；1=部分无依据；2=实质断言均有依据；纯拒答等无可核实断言记 null |
| coverage | 0/1/2 表示对已获参考材料支持的核心需求覆盖程度；参考不足记 null |
| verdict | acceptable / unacceptable / needs_review / uncertain |
| acceptable_rate | acceptable 数 / 已完成评审数；待定也在分母内 |
| high_unacceptable_rate | high 且判 unacceptable 的数量 / high 且 verdict 非 uncertain 的数量 |
| response_modes | 回答、部分回答、拒答各自的数量 |
| known_reference_hit_rate | 返回证据是否命中至少一个出题参考 chunk；排除 missing_detail 题 |
| system_latency_seconds | 有效系统响应的平均/P50/P95 时间；P95 使用 nearest-rank，首题可能含模型加载 |

各维度均值附有效样本数，null 不算 0，也不算满分。
已知参考命中率只是辅助诊断，**不是完整召回率**：出题来源不是唯一正确证据。
对缺失细节问题只根据可见证据评估保留/拒答是否合理，不声称知道全库无答案。

系统异常或 message 含 llm_error 会记录为 system 错误，不作为答案送去评分。
评审失败记录为 judge 错误，保留模型原始返回；均不进入已评分分母。
acceptable_rate 等比例是成功评分样本上的结果，需要同时查看错误数及覆盖数量。

报告含：
- overall_unique_cases：总体独立问题。
- by_dataset：每个输入数据集。
- human_reviewed_cases：至少一个数据集成员已人工审核。
- machine_only_cases：仅机器标注成员。
- skipped：各数据集未选入的原因和数量。

## 保存与恢复

数据在 `backend/data/evaluations/system-runs/evaluations.sqlite`，
与标注结果分开，默认已被 gitignore 忽略。
每次 `run` 结束自动写 JSON 报告和 JSONL 逐题结果到其 `exports` 子目录。
`export --output` 可另存到指定目录；自选导出目录中的实际评测数据不要提交到仓库。

每次只处理答案或评审一个阶段，成功后立即保存。中断后再次 run 继续；
失败样本先跳过，其他题继续执行，使用 --retry-errors 重试失败阶段。
已生成答案不会因为评审重试而再次调用 Agent。
如在结果落盘前强制终止进程，该次调用可能需要重做；
强制终止留下的互斥锁最多 30 分钟过期。

恢复前检查系统代码、资料和有关配置；发生变化时需新建运行，避免混用版本。
评分报告和导出不要求旧环境仍一致。测试不同实现时，分别 create/run，
保留各自 RUN_ID 和报告即可。第一版不自动调阈值、不合并生成一个总评分。

## 测试

```powershell
cd backend
python -m pytest tests/test_system_evaluation.py -q
```

自动测试使用模拟模型，不产生模型费用。

### 模型思考与失败诊断

RAG 系统默认 `DEEPSEEK_THINKING=false`，统一用于流式和非流式 DeepSeek 调用（包括路由、规划和回答）。已有 `.env` 未设置该项时也使用此默认值；重启后端生效。启用思考前需要同时评估各阶段的输出预算，当前路由、规划、QA 的预算分别为 400、700、1200 token。标注和评审继续使用各自的 `EVAL_*_THINKING` 配置。

系统评测失败时，`error.message` 保存具体异常；Agent 返回 `llm_error` 或空答案时，`error.response` 保留失败响应和 trace，失败响应不会参与答案评分。旧记录中已丢失的错误详情无法恢复。修改代码或思考配置后，应使用原数据集创建新运行，旧运行保留作对照，不混合不同版本结果。
