# 检索评测与人工审核

启动后端后，从首页“检索评测与审核”进入，或打开：
http://localhost:8000/evaluation.html

本模块独立于对话流程，调用当前 HybridRetriever，**不执行 Agent 的改写、补检索、
summary coverage loop 或答案生成**。summary profile 是检索器的 summary 策略。
它记录当前 estimate_confidence 的结果，不改变线上置信度阈值，也不自动训练校准器。

## 最小操作流程

1. 先上传课程资料。
2. 创建小型计划（建议先 4 个问题，确认模型配置后再扩大）。
3. 点击“运行 / 继续”。出题、检索、每次评审分别保存；页面显示进度与错误。单题失败后记录错误并继续后续题目，不自动无限重试。
4. 可在当前步骤结束后暂停。关闭页面后，已发送的步骤通常仍会完成，但不会自动发起下一步。
   刷新后重新选择评测集并继续。命令行也能接着同一任务处理。
5. 修正配置/连接后点击“重试失败样本”，再点击运行；成功阶段不会重跑。
6. 从优先队列审核：查看原文高亮、机器理由，接受第一次机器标注，或修改集合/各片段标签。
   “跳过”保留在全部列表，可稍后审核；“排除”用于不明确/无效的问题，不进入指标。已出题且完成检索的失败题可直接填写人工标签，无需等待机器评审通过；也可接受已有的第一次有效机器标注。人工确认或排除后，该题不再自动运行。
7. 导出 JSONL，或修改检索实现/设置后，沿用同批问题创建对照运行。

第一版每次串行调用，模型调用量约为 问题数 × (1 + 评审次数)，重试另计。
对照运行复用已生成问题，仅重新检索、评审。配置输入对新建/对照运行生效，
不改变已创建运行。API 密钥不发给浏览器、不写入评测元数据。

## 模型与数据配置

在 backend/.env 中配置；为空的独立地址/密钥回退到 DEEPSEEK_BASE_URL/DEEPSEEK_API_KEY：

```env
EVAL_GENERATION_MODEL=deepseek-v4-flash
EVAL_JUDGE_MODEL=deepseek-v4-pro
EVAL_GENERATION_THINKING=false
EVAL_JUDGE_THINKING=false
EVAL_GENERATION_BASE_URL=
EVAL_JUDGE_BASE_URL=
EVAL_GENERATION_API_KEY=
EVAL_JUDGE_API_KEY=
EVAL_DIR=./data/evaluations
```

DeepSeek 评测调用默认显式关闭 thinking，以免推理耗尽输出预算而没有 JSON 正文。
可通过上述两个 THINKING 设置分别开启；非 deepseek 开头的模型名不发送该扩展字段。
此设置只影响评测请求，原有聊天请求保持不变。
思考模式说明：https://api-docs.deepseek.com/guides/thinking_mode/

接口使用 OpenAI 兼容的 /chat/completions，要求模型能输出 JSON；
更换服务时分别配置地址、密钥和模型名，重启后端。非兼容服务需要适配客户端。
不读取或暴露密钥给评测页面。模型服务可能将别名指向更新版本，
记录的 model 是请求名，不保证是提供商的固定底层模型版本。
DeepSeek 官方说明：旧 deepseek-v4-flash 别名仍可调用，但映射到 V4.1 Flash：
https://api-docs.deepseek.com/zh-cn/

结果保存到 backend/data/evaluations/evaluations.sqlite，已加入 gitignore。
每步是一次持久化检查点；成功返回前崩溃可能导致该步再次调用。
写锁在正常完成/异常时释放；进程被强制终止时，锁最长 30 分钟后自动过期。
同一个评测集不要同时在浏览器和 CLI 执行。接口按本地开发工具设计，沿用项目的访问控制方式。

## 数据与标签

- 按文档轮流采样，随机选择片段及相邻片段；每个计划循环生成事实、口语改写、
  比较/综合、缺失细节探测四类问题。小型或单片段资料无法保证跨片段题质量，应人工检查。
- 引文优先精确匹配；允许仅有空格、换行等空白差异，并映射回原文子串用于高亮。标点、文字改写和跨片段拼接仍拒绝。生成引文真实不等于问题或参考关键点正确；
  出题来源只是已知参考，不是唯一相关片段。
- 缺失细节探测不预设全库无答案。评审只判断实际召回的上下文。
- 评审输入隐藏检索分数、排名、参考关键点及当前 confidence；以打乱顺序的上下文独立判断。
  两次评审都使用评审模型，第二次再次打乱上下文；一致不代表标签一定正确。
- 标签：irrelevant（无关）、partial（部分支持）、sufficient（充分支持）、uncertain（无法判断）。
  对每个片段及整个集合分别判断。partial/sufficient 要提供逐字引文。
- 队列优先展示 high 但证据非充分、两次评审分歧、无法判断以及约 20% 的随机抽检样本。
  随机抽检成员在创建时固定，不因分数/标签变化。
- 人工修改保留机器评审与人工操作历史；接受按钮以第一次机器评审为准。
- 按来源文档将约 20% 文档划入测试集；同源问题不会跨集合。
  仅一个文档时全部归校准集。它不是对问题难度分层的统计抽样，也不能消除跨文档重复内容。
  调阈值仅使用校准集，测试集用于最后检查；小样本比例只用于调试。

## 报告口径

机器统计使用第一次评审，人类统计只使用已接受/修正的审核标签：
- high 中证据不足：已确定标签为 partial/irrelevant 的 high 数 / 已确定标签的 high 数。
- 充分却判 low：标签为 sufficient 且 confidence=low 的数量 / sufficient 数。
- uncertain 不参与上述比例的分母，单独计数；排除样本从统计中移除。
- 人工队列偏向难例，人工审核子集指标不能外推为全系统准确率。
- 报告分别列出全部/校准/测试，给出分母及置信度分布。
- 对照运行保留 case_id、问题、来源和 split，清除召回、评审及人工标签。
  页面列出两次运行各自的报告以及匹配问题的 confidence 变化；报告本身并非仅匹配子集统计。
- 每次运行保存资料内容指纹、检索配置及检索代码指纹、请求模型名、服务地址、
  提示词版本、随机种子和原文快照。资料、检索设置/代码改变后，旧运行不会继续检索，
  需创建对照运行。更改提示词实现时应递增 PROMPT_VERSION。
- 失败历史会保留模型响应、具体失败引文、对应原文和评审轮次，重试或人工处理后仍可查看并导出。旧失败记录无法补回已丢失的模型响应，需重试产生新的诊断记录。
- 已生成的问题和已经完成的模型输出即为缓存；第一版不在不同评测集之间共享 LLM 缓存。
  不会从机器标签自动宣称概率校准成功。

## 命令行

从项目根目录或 backend 运行（脚本自动切换到 backend，统一 .env / 数据路径）：

```powershell
conda activate course-rag-agent
python backend/evaluate.py create --count 4 --name smoke --judge-passes 2
# 复制输出的评测集 ID
python backend/evaluate.py run DATASET_ID
python backend/evaluate.py status DATASET_ID
python backend/evaluate.py retry DATASET_ID
python backend/evaluate.py run DATASET_ID
python backend/evaluate.py create --source DATASET_ID --name comparison --top-k 8
python backend/evaluate.py export DATASET_ID | Out-File -Encoding utf8 evaluation.jsonl
```

导出首行为 record_type=metadata，后续为 sample，含机器/人工标签、原文和各路分数。
实际评测数据不要提交到仓库。

## 验证

```powershell
cd backend
python -m pytest tests/test_evaluation.py -q
```

测试使用模拟模型与检索结果，不调用付费 API。

## 下一步：评测系统答案

完成标注后，可用 [系统端到端评测脚本](system-evaluation.md) 同时选择多个数据集，
调用真实 Agent 流程并评估本次答案；原标注结果保持独立。

## 向已有评测集追加题目

选择当前评测集，在“运行与报告”输入追加题数（每次 1–200），点击“向当前评测集追加题目计划”，再点击“运行 / 继续”。追加计划不调用模型；失败步骤依旧可以重试。可多次追加，总数不受单次 200 题限制。

新增题沿用该评测集的出题模型、评审模型、资料范围、检索配置和文档分组规则。已有题目、case_id、评审、人工审核结果不变；元数据记录追加次数与数量。处理中会拒绝追加；资料库或检索配置发生变化时应创建新评测集或对照运行，避免混用版本。

出题前提供已有题干，保存前在整个当前评测集内强制去重（包括本批此前已生成的题目），使用 Unicode NFKC、忽略大小写、空白和标点。文字重复最多自动重试 3 次，仍重复则标记失败，保留题目计划且不写入重复题。此检查不保证不同措辞的语义去重，也不清理历史已有重复题。

```powershell
python backend/evaluate.py append DATASET_ID --count 10
python backend/evaluate.py run DATASET_ID
```

API：`POST /api/evaluations/datasets/{dataset_id}/append`，请求体 `{"count": 10}`。
已有系统评测运行使用创建时的数据快照，追加完成后需创建新的系统评测运行才能包含新增问题。
