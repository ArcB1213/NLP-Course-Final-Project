const $ = (id) => document.getElementById(id);
const labels = {irrelevant: "无关", partial: "部分支持", sufficient: "充分支持", uncertain: "无法判断"};
const kinds = {fact: "事实问答", paraphrase: "口语改写", comparison: "比较 / 综合", missing_detail: "缺失细节探测"};
let dataset = null, datasets = [], running = false, position = 0;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
async function api(path, options = {}) {
  const response = await fetch("/api/evaluations" + path, {
    ...options, headers: {"Content-Type": "application/json", ...options.headers},
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(typeof error.detail === "string" ? error.detail : JSON.stringify(error.detail || response.status));
  }
  return response.json();
}
function fail(error) { $("error").textContent = error.message; }
function act(fn) {
  return async () => {
    $("error").textContent = "";
    try { await fn(); } catch (error) { fail(error); }
  };
}
function request() {
  return {
    name: $("name").value, count: Number($("count").value), top_k: Number($("top-k").value),
    profile: $("profile").value, seed: Number($("seed").value), judge_passes: Number($("judge-passes").value),
    generation_model: $("generation-model").value.trim(), judge_model: $("judge-model").value.trim(),
    document_ids: [...$("documents").querySelectorAll("input:checked")].map(n => n.value),
  };
}
function controls() {
  const selected = !!dataset;
  for (const id of ["datasets", "create", "clone", "refresh"]) $(id).disabled = running || (id === "clone" && !selected);
  $("run").disabled = running || !selected || dataset?.busy || dataset?.progress.pending === 0;
  $("append").disabled = running || !selected || dataset?.busy;
  $("append-count").disabled = running || !selected || dataset?.busy;
  $("pause").disabled = !running;
  $("retry").disabled = running || !selected || !dataset?.progress.errors;
  $("compare").disabled = !selected;
  $("export").hidden = !selected;
  if (selected) $("export").href = "/api/evaluations/datasets/" + dataset.id + "/export";
}
async function loadList(selected = $("datasets").value) {
  datasets = await api("/datasets");
  for (const id of ["datasets", "baseline"]) {
    const prior = id === "datasets" ? selected : $(id).value;
    $(id).replaceChildren(new Option("请选择", ""));
    datasets.forEach(d => $(id).add(new Option(d.config.name + " · " + d.id.slice(0, 8), d.id)));
    $(id).value = prior;
  }
}
async function loadDataset() {
  const id = $("datasets").value;
  dataset = id ? await api("/datasets/" + id) : null;
  render();
}
function rate(value) { return value === null ? "—" : (value * 100).toFixed(1) + "%"; }
function reportTable(report) {
  const table = element("table");
  const head = element("tr");
  ["范围 / 标签来源", "已标注 / 待定", "High 中证据不足", "充分却判 Low", "置信度分布"].forEach(x => head.append(element("th", x)));
  table.append(head);
  for (const [split, name] of [["all", "全部"], ["calibration", "校准"], ["test", "测试"]]) {
    for (const [source, caption] of [["machine", "机器预标注"], ["human", "人工确认子集"]]) {
      const m = report[split][source], row = element("tr");
      [name + " / " + caption, m.labeled + " / " + m.uncertain,
        rate(m.high_insufficient_rate) + "（分母 " + m.high_count + "）",
        rate(m.sufficient_low_rate) + "（分母 " + m.sufficient_count + "）",
        JSON.stringify(m.confidence_counts)].forEach(x => row.append(element("td", x)));
      table.append(row);
    }
  }
  return table;
}
function render() {
  controls();
  if (!dataset) {
    $("report").replaceChildren(); $("review-card").textContent = "请选择评测集。"; return;
  }
  const p = dataset.progress;
  $("progress").max = p.total;
  $("progress").value = p.finished;
  $("progress-text").textContent = "计划 " + p.total + " · 已出题 " + p.generated + " · 已检索 " + p.retrieved +
    " · 自动评审完成 " + p.completed + " · 人工确认 " + p.reviewed + " · 已处理 " + p.finished + " · 待运行 " + p.pending + " · 失败待处理 " + p.errors;
  $("report").replaceChildren(reportTable(dataset.report));
  const {samples, report, ...metadata} = dataset;
  $("metadata").textContent = JSON.stringify(metadata, null, 2);
  if (!running) $("status").textContent = dataset.busy ? "服务端正在处理一步，请稍后刷新。" : "结果已保存，可继续运行或审核。";
  renderCard();
}
function filtered() {
  if (!dataset) return [];
  const mode = $("filter").value;
  return dataset.samples.filter(s => {
    if ($("split").value !== "all" && s.split !== $("split").value) return false;
    if (mode === "errors") return !!s.error;
    if (mode === "all") return true;
    if (mode === "random") return s.random_audit;
    const pending = !s.review;
    const ready = !!s.error || s.judgments.length >= dataset.config.judge_passes;
    return pending && ready && (mode === "pending" || s.error || s.reasons.length);
  }).sort((a, b) => (b.error ? 100 : 0) + (b.reasons?.length || 0) - (a.error ? 100 : 0) - (a.reasons?.length || 0));
}
function highlight(text, quotes) {
  const node = element("div", undefined, "source");
  const ranges = [];
  for (const quote of quotes) {
    let start = text.indexOf(quote);
    while (quote && start !== -1) {
      ranges.push([start, start + quote.length]);
      start = text.indexOf(quote, start + quote.length);
    }
  }
  ranges.sort((a, b) => a[0] - b[0]);
  let cursor = 0;
  for (const [start, end] of ranges) {
    if (start < cursor) continue;
    node.append(document.createTextNode(text.slice(cursor, start)), element("mark", text.slice(start, end)));
    cursor = end;
  }
  node.append(document.createTextNode(text.slice(cursor)));
  return node;
}
function labelSelect(value, chunkId) {
  const select = element("select");
  Object.entries(labels).forEach(([key, caption]) => select.add(new Option(caption, key)));
  select.value = value || "uncertain";
  if (chunkId !== undefined) select.dataset.chunkId = chunkId;
  select.disabled = running;
  return select;
}
function renderCard() {
  const cards = filtered();
  position = Math.max(0, Math.min(position, cards.length - 1));
  $("position").textContent = cards.length ? (position + 1) + " / " + cards.length : "0 / 0";
  const host = $("review-card");
  host.replaceChildren();
  const s = cards[position];
  if (!s) { host.textContent = "当前筛选下暂无样本。可切换“全部样本”或继续运行。"; return; }
  host.append(element("h3", s.generated?.question || "待生成问题"));
  host.append(element("p", kinds[s.kind] + " · " + s.split + " · 当前置信度 " + (s.confidence || "未检索")));
  if (s.kind === "missing_detail") host.append(element("p", "此题用于探测缺失细节；不预设整个知识库无答案。"));
  for (const reason of s.reasons) host.append(element("span", reason, "badge"));
  if (s.error) host.append(element("p", "失败阶段：" + s.error.stage + "；" + s.error.message));
  const failures = [...(s.failure_history || [])];
  if (s.error && !failures.some(x => x.at === s.error.at)) failures.push(s.error);
  if (failures.length) {
    const history = element("details");
    history.append(element("summary", "失败详情与历史（" + failures.length + " 次）"));
    failures.forEach(failure => {
      history.append(element("p", failure.at + " · " + failure.stage +
        (failure.judge_pass ? " · 第 " + failure.judge_pass + " 次评审" : "") + " · " + failure.message));
      for (const detail of failure.quote_failures || []) {
        history.append(element("p", detail.scope + " · " + detail.chunk_id + " · " + detail.reason));
        history.append(element("p", "模型引用："), element("pre", detail.quote));
        history.append(element("p", "对应原文："), element("pre", detail.source_text ?? "引用 ID 不属于允许的片段"));
      }
      const raw = element("details");
      raw.append(element("summary", "查看失败模型响应（未通过校验，不作为有效标注）"),
        element("pre", failure.raw_response || "这次失败未保存模型响应；旧记录需要重试后才能获得详情。"));
      history.append(raw);
    });
    host.append(history);
  }
  const reference = element("details");
  reference.append(element("summary", "出题来源与参考关键点（机器生成，需核实）"));
  reference.append(element("p", (s.generated?.key_points || []).join("；")));
  for (const source of s.sources) {
    reference.append(element("p", source.source_file + " · 页 " + (source.page ?? "—") + " · " + (source.heading || "")));
    const quotes = (s.generated?.evidence || []).filter(q => q.chunk_id === source.chunk_id).map(q => q.quote);
    reference.append(highlight(source.text, quotes));
  }
  host.append(reference);
  const canCorrect = !!s.generated && s.retrieved !== null;
  if (!s.judgments.length) host.append(element("p", canCorrect
    ? "尚无有效机器标注。可根据下方召回原文填写人工标签；保存后不再自动评审此题。"
    : "尚未完成出题和检索，可重试失败步骤或排除此题。"));
  s.judgments.forEach((judgment, index) => host.append(element("p",
    "集合评审 " + (index + 1) + "：" + labels[judgment.overall.label] + " — " + judgment.overall.reason)));
  const overallLabel = element("label", "人工集合标签：");
  const overall = labelSelect(s.review?.overall || s.judgments[0]?.overall.label);
  overall.id = "review-overall";
  overallLabel.append(overall); host.append(overallLabel);
  for (const c of s.retrieved || []) {
    const block = element("div", undefined, "chunk");
    block.append(element("strong", c.source_file + " · 页 " + (c.page ?? "—") + " · " + (c.heading || "")));
    const quotes = s.judgments.flatMap(j => [
      ...j.overall.evidence, ...j.chunks.flatMap(x => x.evidence),
    ]).filter(q => q.chunk_id === c.chunk_id).map(q => q.quote);
    block.append(highlight(c.text, quotes));
    s.judgments.forEach((j, index) => {
      const judgment = j.chunks.find(x => x.chunk_id === c.chunk_id);
      block.append(element("p", "评审 " + (index + 1) + "：" + labels[judgment.label] + " — " + judgment.reason));
    });
    const humanLabel = element("label", "人工片段标签：");
    humanLabel.append(labelSelect(s.review?.chunks?.[c.chunk_id] ||
      s.judgments[0]?.chunks.find(x => x.chunk_id === c.chunk_id)?.label, c.chunk_id));
    block.append(humanLabel);
    const scores = element("details");
    scores.append(element("summary", "查看检索分数"), element("pre", JSON.stringify({
      chunk_id: c.chunk_id, vector_score: c.vector_score, bm25_score: c.bm25_score,
      final_score: c.final_score, rerank_score: c.rerank_score,
    }, null, 2)));
    block.append(scores); host.append(block);
  }
  const note = element("textarea");
  note.id = "review-note"; note.placeholder = "审核备注（例如：问题不明确、引文不足、修正原因）";
  note.value = s.review?.note || ""; note.disabled = running;
  host.append(element("p", s.review ? "已有审核：" + s.review.status + " · " + s.review.at : "尚未人工审核"), note);
  const actions = element("div", undefined, "toolbar");
  for (const [status, text] of [["accepted", "接受第一次机器标注"], ["corrected", "保存人工修正"],
    ["skipped", "跳过"], ["excluded", "排除此问题（无效题）"]]) {
    const button = element("button", text);
    button.disabled = running || (status === "accepted" && !s.judgments.length) ||
      (status === "corrected" && !canCorrect);
    button.addEventListener("click", act(async () => {
      const chunks = {};
      host.querySelectorAll("select[data-chunk-id]").forEach(n => { chunks[n.dataset.chunkId] = n.value; });
      await api("/datasets/" + dataset.id + "/samples/" + s.id + "/review", {
        method: "PUT", body: JSON.stringify({status, overall: status === "corrected" ? overall.value : null,
          chunks: status === "corrected" ? chunks : {}, note: note.value}),
      });
      await loadDataset();
    }));
    actions.append(button);
  }
  host.append(actions);
}
$("create-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  await act(async () => {
    $("create").disabled = true;
    try {
      const created = await api("/datasets", {method: "POST", body: JSON.stringify(request())});
      await loadList(created.id); position = 0; await loadDataset();
    } finally { controls(); }
  })();
});
$("append-form").addEventListener("submit", (event) => {
  event.preventDefault();
  act(async () => {
    const id = dataset.id, count = Number($("append-count").value);
    running = true; controls();
    try {
      dataset = await api("/datasets/" + id + "/append", {method: "POST", body: JSON.stringify({count})});
      await loadList(id);
    } finally { running = false; render(); }
  })();
});
$("clone").onclick = act(async () => {
  if (!$("create-form").reportValidity()) return;
  const created = await api("/datasets/" + dataset.id + "/clone", {method: "POST", body: JSON.stringify(request())});
  await loadList(created.id); position = 0; await loadDataset();
});
$("datasets").onchange = act(async () => { position = 0; $("comparison").replaceChildren(); await loadDataset(); });
$("refresh").onclick = act(async () => { await loadList(); await loadDataset(); });
$("run").onclick = act(async () => {
  const id = dataset.id; running = true; controls(); renderCard();
  try {
    while (running) {
      $("status").textContent = "正在处理一个阶段…关闭页面后不会继续发起下一步，已完成结果会保存。";
      const result = await api("/datasets/" + id + "/step", {method: "POST"});
      await loadDataset();
      if (result.status === "error") $("error").textContent = "已记录失败并继续后续题目：" + result.error;
      if (result.status === "idle") break;
    }
  } finally { running = false; render(); }
});
$("pause").onclick = () => { running = false; $("pause").disabled = true; $("status").textContent = "等待当前步骤结束并保存…"; };
$("retry").onclick = act(async () => {
  for (const s of dataset.samples.filter(s => s.error)) {
    await api("/datasets/" + dataset.id + "/samples/" + s.id + "/retry", {method: "POST"});
  }
  await loadDataset();
});
$("filter").onchange = $("split").onchange = () => { position = 0; renderCard(); };
$("previous").onclick = () => { position--; renderCard(); };
$("next").onclick = () => { position++; renderCard(); };
$("compare").onclick = act(async () => {
  const baseline = $("baseline").value;
  if (!baseline) throw new Error("请选择对照基线");
  const result = await api("/datasets/" + dataset.id + "/compare/" + baseline);
  const host = $("comparison");
  host.replaceChildren(element("p", "匹配到已检索的同批问题：" + result.paired + "。仅比较共享 case_id 的问题。"),
    element("h3", "基线报告"), reportTable(result.baseline),
    element("h3", "当前报告"), reportTable(result.current));
  for (const c of result.changes) host.append(element("p", c.before + " → " + c.after + "：" + c.question));
  if (!result.changes.length) host.append(element("p", "已匹配问题的置信度等级没有变化。"));
});
$("create-form").addEventListener("input", () => {
  $("estimate").textContent = "完整运行约 " + Number($("count").value) * (1 + Number($("judge-passes").value)) +
    " 次模型调用，失败重试会额外调用。对照运行复用问题，不再调用出题模型。";
});
act(async () => {
  const defaults = await api("/defaults");
  $("generation-model").value = defaults.generation_model; $("judge-model").value = defaults.judge_model;
  const response = await fetch("/api/documents");
  if (!response.ok) throw new Error("加载资料列表失败");
  const documents = await response.json();
  $("documents").replaceChildren();
  documents.forEach(doc => {
    const label = element("label"), input = element("input");
    input.type = "checkbox"; input.value = doc.id;
    label.append(input, document.createTextNode(doc.filename + "（" + doc.chunk_count + " 个片段）"));
    $("documents").append(label);
  });
  if (!documents.length) $("documents").textContent = "暂无资料，请返回课程助手上传。";
  await loadList();
  $("create-form").dispatchEvent(new Event("input"));
  controls();
})();
