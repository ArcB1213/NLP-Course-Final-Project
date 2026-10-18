# 文档解析与质量检查

## 正式上传：MinerU standard

PDF 默认通过隔离的 Python 进程运行 MinerU 4.0.12 `standard`。ONNX 小模型在 CPU 运行，
llama-cpp 的视觉模型使用可用的 Vulkan GPU。Markdown／TXT 不调用 MinerU。
没有可用 GPU 时，实际设备选择由 llama-cpp 决定；安装或解析失败会返回错误，不会静默降级。

本机安装目录为 `D:\NLP-CourseWork\data\tools\mineru`，已经安装并下载标准模型。
新机器可从项目根目录安装（Windows，已验证 Python 3.13）：

```powershell
python -m venv data/tools/mineru/venv
& data/tools/mineru/venv/Scripts/python.exe -m pip install "mineru==4.0.12" "mineru-llama-cpp==0.1.3"
$env:MINERU_HOME = "$PWD/data/tools/mineru/state"
$env:MINERU_CONFIG = "$PWD/data/tools/mineru/state/config.yaml"
$env:MINERU_MODEL_BASE_DIR = "$PWD/data/tools/mineru/models"
& data/tools/mineru/venv/Scripts/mineru-kit.exe models download --tier standard --small-backend onnx --vlm-engine llama-cpp --source modelscope
& data/tools/mineru/venv/Scripts/mineru-kit.exe models verify --tier standard --small-backend onnx --vlm-engine llama-cpp
```

模型文件约 1.95 GiB；下载后正式解析只读取本地模型。环境、权重和解析结果都在 Git 忽略的 `data/` 内。
后端默认从仓库位置定位独立解释器与模型，不依赖从根目录还是 `backend` 启动；数据库路径仍遵循原有相对路径配置。
可在 `.env` 配置 `PDF_PARSER=mineru`、`MINERU_PYTHON`、`MINERU_HOME`、`MINERU_CACHE_DIR` 和 `MINERU_TIMEOUT=1800`。
显式设置 `PDF_PARSER=native` 才会使用原生提取。不要将 MinerU 安装进后端 conda 环境。

识别结果按 PDF 的 SHA256 缓存在 `data/parsed/mineru-upload/<hash>-standard-v1/`，
相同内容共享结果；不同文件内容自动重新解析。目录包含 `middle_json.json`、`markdown.md`、原图、日志及后端逐页数据 `pages-text-v3.json`。
检索文本不包含图片链接或 Base64 图片载荷；原图仍单独保存。旧版 `pages.json`、`pages-text-v2.json` 不再读取，优先从已保存的结构化结果重新渲染，不重复运行模型。
只有完整成功才写入 `pages-text-v3.json`；失败不会成为可用缓存。需要同一文件重新执行模型推理时，先移走对应缓存目录再重解析。
后端页级报告保留原始页码、归一化块坐标（0–1）及产物目录。原图留在本地，没有额外公开静态资源接口。

公式和表格的识别结果不等于经过核验：统一记录页级核对提示，图片另标记关系尚未文字化。
HTML 表格和独立公式整体保留，避免切坏单元格／LaTeX；较大的单个块可能超过目标长度。
这些提示会传播到检索片段。未增加针对课程内容的纠错规则。

## 文本 LLM 标题判定

默认 `PDF_TITLE_RESOLUTION=true`：使用 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`，
最多4页一组发送候选文字、原始块类型、归一化位置及每页最多700字符的正文。没有图片、字号、颜色或高亮数据。
标题任务单独启用思考模式，输出预算6000 tokens；不改变聊天任务的思考配置。首次上传会增加外部 API 调用及等待时间。
完全离线或不希望发送这些文字时设为 `false`。旧资料不会自动重建索引。

worker 将未过滤的 header、paragraph_title 和页上部短文本保存在 `title_candidates`，
逐页缓存升级为 `pages-text-v3.json`；旧文件仍保留，但不再作为新处理的输入。
若存在 `middle_json.json` 则只重新渲染，无需重复模型识别。

LLM 只返回候选块 ID，区分页面主标题、章节信息、导航与副标题。代码校验页集合、ID 来源、重复和角色冲突；
格式或引用错误最多反馈修正一次。有效结果才写入 `titles/` 缓存，原始模型返回和拒绝原因保存在其 `attempts/` 中。
缓存键包含候选文本、坐标、正文节选、模型、接口地址和提示版本；不包含 API 密钥。

报告中的 `heading` 为页面主标题，`section_heading` 为当前章节候选，`title_resolution` 保存所选 ID、理由及判定状态。
`section_basis=inferred` 表示根据文字推断，不能视为已识别高亮；不确定时章节为空。
标题不确定或请求失败时不强制选择第一段正文，而是保留候选并记录警告。
确认的标题恢复到正文，`retrieval_context` 让每个分块继承页面标题和章节依据，使向量、BM25、重排与问答使用一致上下文。
这不是完整章节树，也尚未实现跨页父子章节关系或相邻片段扩展。

仅预览指定页、不改资料库（从 `backend` 运行，按实际文件名替换）：

```powershell
python preview_pdf_titles.py 'data/uploads/example.pdf' --pages 25-28 --output ../data/parsed/title-preview
```

输出 `llm-input.json`、`result.json` 和模型响应缓存，可与原页人工核对；其中 `result.json` 包含解析页和最终分块。

## 原生提取预览

`backend/parse_document.py` 仍使用本地 PyMuPDF 进行快速对照预览，不调用模型：

- 按带坐标的文本块读取内容，保留原始页码和候选页标题。
- 只清理在多数页面边缘重复出现的文字；通过边缘重复图片识别部分校徽／装饰图。
- 将可检测的原生网格表格转成 Markdown，避免同一单元格重复进入正文。
- 长表格按完整行分块，每块重复表头；独立 `$$` 公式块保留边界。
- 对正文图片、疑似排列损坏的公式及无文字的图形页面记录诊断警告。

版面排序和问题检测是启发式方法，不保证还原多栏文档的阅读顺序，也不保证检测出每个损坏公式。
表格检测支持 PDF 中的原生文本／线条表格；扫描表格、图片内文字、复杂公式和示意图仍需要 OCR 或视觉模型。
超长单行表格或独立公式会保持完整，可能超过目标 `chunk_size`，应在解析预览中核对。

## 先预览，再导入

从项目根目录运行：

```powershell
conda activate course-rag-agent
python backend/parse_document.py "D:\NLP-CourseWork\data\uploads\doc_2f770d23e0584bd1_3.自动标注.pdf" --output data/parsed/automatic-tagging
```

输出目录由 `--output` 指定，相对于当前工作目录：

- `document.md`：按原始 PDF 页码排列的提取文本。
- `report.json`：页码、标题、文本块坐标、原生表格、正文图片位置、警告及分块数量。

预览不修改原 PDF、SQLite 或检索索引。`data/` 为本地数据目录，已被 Git 忽略。

检查报告时请对照原页，尤其检查表格的行列关系、公式上下标与条件、图片内容是否缺失。
例如示例资料的第 56 页费用表、第 57 页路径图是图片；原生文字提取无法恢复其内容。
第 22 页公式的字符能被提取，但排列已损坏。上述内容会标记为需要进一步识别，不能把附近标题当成完整证据。

## 上传后的状态与接口

上传 PDF 时，原文件旁保存 `<document_id>.processing.json` 形式的解析报告；实际文件名沿用上传文件的主文件名。
报告包含提取后的课程内容，应与上传资料一起作为本地数据管理。

- `indexed`：完成索引且本次未检测到上述警告；不表示解析经过人工认证。
- `indexed_partial`：可用文字已建立索引，但有页面需要核对或进一步识别。前端显示“部分识别”。
- `failed`：解析／索引失败；全扫描且未提取到可索引文字时，不再显示零分块的成功状态。

警告也会写入对应页的文本分块，使生成阶段能看到该页的证据限制。

`GET /api/documents` 与上传响应的文档对象新增 `processing_warnings: string[]`。
`GET /api/documents/{document_id}/processing` 返回完整报告；旧文档尚未生成报告时返回 404。
旧文档仍可正常列出，警告列表默认为空，这不代表其旧解析没有问题。

### 重新解析已有资料

从与启动后端相同的工作目录及环境执行，例如标准的 `backend` 启动方式：

```powershell
conda activate course-rag-agent
cd backend
python reparse_documents.py
```

脚本只处理当前数据库中登记的 PDF，保留文档 ID、文件和创建时间；成功后在事务中替换该文档的全部旧分块，
批处理最后重建 FAISS／BM25 索引。解析失败保留原有分块并记录错误，脚本以非零状态退出。
标题接口失败也会阻止覆盖已有分块及解析报告；这不同于模型正常返回“无法确定标题”。
标题输出截断（`finish_reason=length`）时，从原请求重试一次，将输出上限从 6000 提高到 12000 token，
仍要求正常结束和源块 ID 校验。有效标题缓存继续复用，无需删除 MinerU 解析产物或重新运行本地模型。
每次请求的审计文件位于对应 MinerU 缓存的 `titles/attempts/`，截断记录包含结束原因、输出上限和接口用量信息，
不保存模型思考原文。余额不足属于 HTTP 402，与完整性校验失败不同。
按 Ctrl+C 后会为已提交的文档重建索引并保存进度，请等待收尾；强制结束进程可能跳过这一过程。
备份保存在当前 SQLite 目录的 `before-mineru-<时间>/`，含 `app.db`、旧索引及 `results.json`。
建议维护期间暂停检索和上传，避免读到批处理的中间状态；完成后重启后端，清除内存中的 BM25 缓存。
分块内容和数量会改变，已有检索评测结果不自动重算，应在新索引上重新运行评测。
如果曾从仓库根目录运行，根目录的 `data/sqlite/app.db` 是另一套库；在根目录用 `python backend/reparse_documents.py` 单独处理，两个库不会合并。

也可调用 `POST /api/documents/{document_id}/reparse` 重新入库单个文档，并立即重建当前服务的索引。
该接口复用上述内容缓存，不创建重复文档。`GET /api/documents/{document_id}/processing` 可查看解析方法和核对提示。

## 验证

目前已知的限制是：复杂公式仍可能丢失下标层级、产生额外符号；独立公式块与说明文字分离后，
可能只召回说明而漏掉公式。语义分块、相邻片段扩展和相应效果评测留待下一阶段。

```powershell
cd backend
python -m pytest tests/test_document_processing.py -q
python -m pytest tests/test_mineru_ingestion.py -q
```

测试通过本地生成的小型 PDF 验证边缘文字清理、原生表格、正文图片诊断、分块边界、
页码和警告传递、空内容失败、报告接口及旧文档兼容，不调用模型 API。

PyMuPDF 文档：[文本块和坐标](https://pymupdf.readthedocs.io/en/latest/app1.html)、
[表格检测](https://pymupdf.readthedocs.io/en/latest/page.html#Page.find_tables)。
