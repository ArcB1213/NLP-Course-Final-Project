const API_BASE = "/api";

const RENDER_DEPENDENCIES = {
  marked: [
    "https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js",
    "https://unpkg.com/marked@12.0.2/marked.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js",
  ],
  DOMPurify: [
    "https://cdn.jsdelivr.net/npm/dompurify@3.1.5/dist/purify.min.js",
    "https://unpkg.com/dompurify@3.1.5/dist/purify.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.1.5/purify.min.js",
  ],
  katex: [
    "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js",
    "https://unpkg.com/katex@0.16.11/dist/katex.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11/katex.min.js",
  ],
};

const KATEX_STYLES = [
  "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css",
  "https://unpkg.com/katex@0.16.11/dist/katex.min.css",
  "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11/katex.min.css",
];

function loadScriptFromSources(globalName, sources) {
  if (window[globalName]) return Promise.resolve();
  return sources.reduce(
    (attempt, source) => attempt.catch(() => new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = source;
      script.onload = () => window[globalName] ? resolve() : reject(new Error(`${globalName} 未导出`));
      script.onerror = () => reject(new Error(`无法加载 ${source}`));
      document.head.appendChild(script);
    })),
    Promise.reject(new Error(`${globalName} 尚未加载`)),
  );
}

function loadStylesheetFromSources(sources) {
  return sources.reduce(
    (attempt, source) => attempt.catch(() => new Promise((resolve, reject) => {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = source;
      link.onload = resolve;
      link.onerror = () => reject(new Error(`无法加载 ${source}`));
      document.head.appendChild(link);
    })),
    Promise.reject(new Error("KaTeX 样式尚未加载")),
  );
}

const rendererReady = Promise.all([
  loadScriptFromSources("marked", RENDER_DEPENDENCIES.marked),
  loadScriptFromSources("DOMPurify", RENDER_DEPENDENCIES.DOMPurify),
  loadScriptFromSources("katex", RENDER_DEPENDENCIES.katex),
  loadStylesheetFromSources(KATEX_STYLES),
]).then(() => {
  window.marked.setOptions({ breaks: true, gfm: true });
}).catch((error) => {
  console.error("Markdown/LaTeX 渲染依赖加载失败", error);
  throw error;
});

// ---- Resizable layout ----
const LAYOUT_KEY = "course-rag-layout";
const layout = document.querySelector(".layout");
const MIN_LEFT_FR = 0.6;
const MIN_CHAT_FR = 1.5;
const MIN_RIGHT_FR = 0.55;

function readFrs() {
  const cs = getComputedStyle(layout);
  const parse = (name, fallback) => {
    const raw = (cs.getPropertyValue(name) || "").trim();
    const n = parseFloat(raw);
    return Number.isFinite(n) && n > 0 ? n : fallback;
  };
  return {
    left: parse("--left-fr", 1.6),
    chat: parse("--chat-fr", 4),
    right: parse("--right-fr", 1.4),
  };
}

function applyFrs(frs) {
  layout.style.setProperty("--left-fr", `${frs.left}fr`);
  layout.style.setProperty("--chat-fr", `${frs.chat}fr`);
  layout.style.setProperty("--right-fr", `${frs.right}fr`);
}

function loadLayoutPrefs() {
  try {
    const saved = JSON.parse(localStorage.getItem(LAYOUT_KEY) || "null");
    if (saved && saved.left && saved.chat && saved.right) applyFrs(saved);
  } catch {}
}
loadLayoutPrefs();

function flexibleCols() {
  const cols = layout.getBoundingClientRect().width;
  const rs = layout.querySelectorAll(".resizer");
  let resizerPx = 0;
  rs.forEach((r) => { resizerPx += r.getBoundingClientRect().width; });
  const padding = parseFloat(getComputedStyle(layout).paddingLeft) +
    parseFloat(getComputedStyle(layout).paddingRight);
  const colMargins = parseFloat(getComputedStyle(document.querySelector(".col-chat")).marginLeft) +
    parseFloat(getComputedStyle(document.querySelector(".col-chat")).marginRight);
  return Math.max(1, cols - resizerPx - padding - colMargins);
}

function attachResizer(resizer, target) {
  resizer.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    resizer.setPointerCapture(e.pointerId);
    resizer.classList.add("dragging");
    document.body.classList.add("resizing");

    const startX = e.clientX;
    const startFrs = readFrs();
    const totalFr = startFrs.left + startFrs.chat + startFrs.right;
    const totalPx = flexibleCols();
    const pxPerFr = totalPx / totalFr;

    function onMove(ev) {
      const dxFr = (ev.clientX - startX) / pxPerFr;
      const next = { ...startFrs };
      if (target === "left") {
        next.left = startFrs.left + dxFr;
        next.chat = startFrs.chat - dxFr;
      } else {
        next.chat = startFrs.chat + dxFr;
        next.right = startFrs.right - dxFr;
      }
      if (next.left < MIN_LEFT_FR || next.chat < MIN_CHAT_FR || next.right < MIN_RIGHT_FR) return;
      applyFrs(next);
    }

    function onUp() {
      resizer.removeEventListener("pointermove", onMove);
      resizer.removeEventListener("pointerup", onUp);
      resizer.removeEventListener("pointercancel", onUp);
      resizer.classList.remove("dragging");
      document.body.classList.remove("resizing");
      try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(readFrs())); } catch {}
    }

    resizer.addEventListener("pointermove", onMove);
    resizer.addEventListener("pointerup", onUp);
    resizer.addEventListener("pointercancel", onUp);
  });

  resizer.addEventListener("dblclick", () => {
    layout.style.removeProperty("--left-fr");
    layout.style.removeProperty("--chat-fr");
    layout.style.removeProperty("--right-fr");
    try { localStorage.removeItem(LAYOUT_KEY); } catch {}
  });
}

document.querySelectorAll(".resizer").forEach((r) => attachResizer(r, r.dataset.target));

// ---- Theme ----
const THEME_KEY = "course-rag-theme";
const themeToggle = document.getElementById("theme-toggle");

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  themeToggle.textContent = theme === "dark" ? "夜" : "日";
}
applyTheme(localStorage.getItem(THEME_KEY) || "dark");
themeToggle.addEventListener("click", () => {
  const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  localStorage.setItem(THEME_KEY, next);
  applyTheme(next);
});

// ---- Health ----
async function checkHealth() {
  const dot = document.getElementById("health-dot");
  const text = document.getElementById("health-text");
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(res.statusText);
    dot.className = "health ok";
    text.textContent = "已连接";
  } catch (err) {
    dot.className = "health bad";
    text.textContent = "未连接";
  }
}
checkHealth();
setInterval(checkHealth, 15000);

// ---- Chat ----
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatLog = document.getElementById("chat-log");
const chatSend = document.getElementById("chat-send");
const traceToggle = document.getElementById("show-agent-trace");
const TRACE_PREF_KEY = "course-rag-show-agent-trace";
const SESSION_KEY = "course-rag-session-id";
const sessionStatus = document.getElementById("session-status");
const memoryStatus = document.getElementById("memory-status");
const newSessionBtn = document.getElementById("new-session");
const clearSessionBtn = document.getElementById("clear-session");
const clearMemoryBtn = document.getElementById("clear-memory");
let currentSessionId = localStorage.getItem(SESSION_KEY) || "";

traceToggle.checked = localStorage.getItem(TRACE_PREF_KEY) !== "false";
traceToggle.addEventListener("change", () => {
  localStorage.setItem(TRACE_PREF_KEY, String(traceToggle.checked));
});

function updateSessionStatus() {
  sessionStatus.textContent = currentSessionId
    ? `当前会话：${currentSessionId.slice(0, 8)}`
    : "当前会话：未创建";
}

function clearChatLog() {
  chatLog.innerHTML = '<div class="empty-tip">新会话已准备好，可以继续提问。</div>';
}

function saveSessionId(sessionId) {
  if (!sessionId) return;
  currentSessionId = sessionId;
  localStorage.setItem(SESSION_KEY, sessionId);
  updateSessionStatus();
}

async function refreshMemoryStatus() {
  if (!memoryStatus) return;
  try {
    const res = await fetch(`${API_BASE}/memory`);
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    const count = Array.isArray(data.memories) ? data.memories.length : 0;
    memoryStatus.textContent = `长期记忆：${count} 条`;
  } catch {
    memoryStatus.textContent = "长期记忆：无法加载";
  }
}

newSessionBtn.addEventListener("click", () => {
  currentSessionId = "";
  localStorage.removeItem(SESSION_KEY);
  updateSessionStatus();
  clearChatLog();
});

clearSessionBtn.addEventListener("click", async () => {
  if (!currentSessionId) {
    clearChatLog();
    return;
  }
  if (!confirm("清空当前会话历史？长期记忆不会被删除。")) return;
  const sessionId = currentSessionId;
  currentSessionId = "";
  localStorage.removeItem(SESSION_KEY);
  updateSessionStatus();
  try {
    await fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
  } finally {
    clearChatLog();
  }
});

clearMemoryBtn.addEventListener("click", async () => {
  if (!confirm("清空全部长期记忆？当前会话历史不会被删除。")) return;
  clearMemoryBtn.disabled = true;
  try {
    const res = await fetch(`${API_BASE}/memory`, { method: "DELETE" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await refreshMemoryStatus();
  } catch (err) {
    memoryStatus.textContent = `长期记忆：清空失败 ${err.message}`;
  } finally {
    clearMemoryBtn.disabled = false;
  }
});

updateSessionStatus();
refreshMemoryStatus();

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = chatInput.value.trim();
  if (!query) return;
  const taskType = document.getElementById("task-type").value;
  const topK = parseInt(document.getElementById("chat-topk").value, 10) || 5;
  const usePro = document.getElementById("use-pro").checked;
  const showAgentTrace = traceToggle.checked;

  const tip = chatLog.querySelector(".empty-tip");
  if (tip) tip.remove();

  appendMessage("user", query);
  chatInput.value = "";
  chatSend.disabled = true;

  const agentMsg = appendMessage("agent", "");
  agentMsg.bubble.innerHTML = '<span class="spinner"></span> 思考中...';

  try {
    await streamChat({
      query,
      task_type: taskType,
      use_pro_model: usePro,
      top_k: topK,
      session_id: currentSessionId || null,
      extra_context: { debug_agent_trace: showAgentTrace },
    }, agentMsg);
  } catch (err) {
    agentMsg.bubble.textContent = `请求失败：${err.message}`;
  } finally {
    chatSend.disabled = false;
  }
});

function appendMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = role === "user" ? "你" : "Agent";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(meta);
  wrap.appendChild(bubble);
  chatLog.appendChild(wrap);
  chatLog.scrollTop = chatLog.scrollHeight;
  return { wrap, meta, bubble };
}

function extractMath(text) {
  const expressions = [];
  const codePattern = /(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)/g;
  const source = String(text || "")
    .split(codePattern)
    .map((part, index) => {
      if (index % 2 === 1) return part;
      return part
        .replace(/\\\[([\s\S]*?)\\\]|\$\$([\s\S]*?)\$\$/g, (_, bracketed, dollars) => {
          const mathIndex = expressions.push({ expression: bracketed ?? dollars, display: true }) - 1;
          return `\n<div class="math-placeholder math-display" data-math-index="${mathIndex}"></div>\n`;
        })
        .replace(/\\\(([^\n]*?)\\\)|\$(?!\$)([^\n$]+?)\$/g, (_, parenthesized, dollars) => {
          const mathIndex = expressions.push({ expression: parenthesized ?? dollars, display: false }) - 1;
          return `<span class="math-placeholder" data-math-index="${mathIndex}"></span>`;
        });
    })
    .join("");
  return { source, expressions };
}

function renderMathPlaceholders(target, expressions) {
  target.querySelectorAll("[data-math-index]").forEach((element) => {
    const item = expressions[Number(element.dataset.mathIndex)];
    if (!item) return;
    window.katex.render(item.expression, element, {
      displayMode: item.display,
      throwOnError: false,
      strict: "ignore",
      trust: false,
    });
  });
}

const pendingMarkdown = new WeakMap();

function renderMarkdown(target, text) {
  const renderVersion = {};
  pendingMarkdown.set(target, renderVersion);

  if (!window.marked || !window.DOMPurify || !window.katex) {
    target.textContent = text || "";
    rendererReady.then(() => {
      if (pendingMarkdown.get(target) === renderVersion) renderMarkdown(target, text);
    }).catch(() => {
      if (pendingMarkdown.get(target) === renderVersion) {
        target.dataset.renderError = "Markdown/LaTeX 渲染组件加载失败";
      }
    });
    return;
  }

  const { source, expressions } = extractMath(text);
  const html = window.marked.parse(source);
  target.innerHTML = window.DOMPurify.sanitize(html, {
    ADD_ATTR: ["data-math-index"],
  });
  renderMathPlaceholders(target, expressions);
}

async function streamChat(payload, agentMsg) {
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let answerText = "";
  let firstDelta = true;
  let metaInfo = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const evt of events) {
      const line = evt.trim();
      if (!line.startsWith("data:")) continue;
      const dataStr = line.slice(5).trim();
      if (!dataStr) continue;
      let event;
      try { event = JSON.parse(dataStr); } catch { continue; }

      if (event.type === "meta") {
        metaInfo = event;
        if (event.session_id) saveSessionId(event.session_id);
        agentMsg.meta.innerHTML = renderMetaLine(event);
      } else if (event.type === "delta") {
        if (firstDelta) { agentMsg.bubble.textContent = ""; firstDelta = false; }
        answerText += event.text || "";
        renderMarkdown(agentMsg.bubble, answerText);
        chatLog.scrollTop = chatLog.scrollHeight;
      } else if (event.type === "error") {
        agentMsg.bubble.textContent = event.message || "发生错误";
        return;
      } else if (event.type === "done") {
        if (firstDelta) {
          renderMarkdown(agentMsg.bubble, answerText || "(无回复)");
        } else {
          renderMarkdown(agentMsg.bubble, answerText);
        }
        if (metaInfo && metaInfo.agent_trace && metaInfo.agent_trace.steps?.length) {
          agentMsg.bubble.appendChild(renderAgentTrace(metaInfo.agent_trace));
        }
        if (metaInfo && metaInfo.sources && metaInfo.sources.length) {
          agentMsg.bubble.appendChild(renderSources(metaInfo.sources));
        }
        refreshMemoryStatus();
      }
    }
  }
}

function renderMetaLine(meta) {
  const conf = meta.confidence || "medium";
  const tag = `<span class="confidence-tag ${conf}">${conf}</span>`;
  const sessionTag = meta.session_id ? `<span>会话 ${String(meta.session_id).slice(0, 8)}</span>` : "";
  const parts = [`Agent · ${meta.task_type || "?"}`, tag];
  if (meta.message) parts.push(`<span style="color:var(--warn)">· ${meta.message}</span>`);
  if (sessionTag) parts.push(sessionTag);
  return parts.join(" ");
}

const TRACE_STEP_LABELS = {
  memory_resolve: "记忆解析",
  memory_write: "写入记忆",
  memory_fallback: "记忆回退",
  route: "任务路由",
  plan: "任务规划",
  retrieve: "首次检索",
  judge_evidence: "证据判断",
  retrieve_retry: "二次检索",
  tool_generate: "工具生成",
  tool_stream: "流式生成",
};

function renderAgentTrace(trace) {
  const steps = Array.isArray(trace?.steps) ? trace.steps : [];
  const panel = document.createElement("details");
  panel.className = "agent-trace";

  const summary = document.createElement("summary");
  summary.innerHTML = `<span>Agent 决策过程</span><span class="trace-count">${steps.length} 步</span>`;
  panel.appendChild(summary);

  const timeline = document.createElement("div");
  timeline.className = "trace-timeline";
  steps.forEach((step, index) => timeline.appendChild(renderAgentStep(step, index)));
  panel.appendChild(timeline);
  return panel;
}

function renderAgentStep(step, index) {
  const item = document.createElement("section");
  const status = step.status === "failed" ? "failed" : "ok";
  item.className = `trace-step trace-${step.step_type || "unknown"} ${status}`;

  const header = document.createElement("div");
  header.className = "trace-step-header";
  const title = document.createElement("strong");
  title.textContent = `${index + 1}. ${TRACE_STEP_LABELS[step.step_type] || step.step_type || "未知步骤"}`;
  const badge = document.createElement("span");
  badge.className = `trace-status ${status}`;
  badge.textContent = status === "failed" ? "失败" : "完成";
  header.append(title, badge);
  item.appendChild(header);

  const fields = traceStepFields(step);
  if (fields.length) {
    const grid = document.createElement("dl");
    grid.className = "trace-fields";
    fields.forEach(([label, value]) => {
      const term = document.createElement("dt");
      term.textContent = label;
      const description = document.createElement("dd");
      description.textContent = formatTraceValue(value);
      grid.append(term, description);
    });
    item.appendChild(grid);
  }

  if (step.message) {
    const message = document.createElement("div");
    message.className = "trace-message";
    message.textContent = step.message;
    item.appendChild(message);
  }
  return item;
}

function traceStepFields(step) {
  const input = step.input || {};
  const output = step.output || {};
  if (step.step_type === "memory_resolve") {
    return [
      ["会话", input.session_id],
      ["已有摘要", output.has_summary],
      ["短期消息", output.recent_message_count],
      ["长期记忆", output.long_term_memory_count],
      ["mode", output.mode],
      ["confidence", output.confidence],
      ["answer_query", output.answer_query],
      ["retrieval_query", output.retrieval_query],
      ["referenced_turns", output.referenced_turns],
      ["reason", output.reason],
    ].filter(([, value]) => value !== undefined && value !== null && value !== "");
  }
  if (step.step_type === "memory_write") {
    return [
      ["会话", input.session_id],
      ["任务", input.task_type],
      ["回答字数", output.answer_chars],
    ].filter(([, value]) => value !== undefined && value !== null && value !== "");
  }
  if (step.step_type === "route") {
    return [
      ["任务类型", output.task_type],
      ["检索策略", output.retrieval_profile],
      ["改写 Query", output.rewritten_query],
      ["路由原因", output.reason],
      ["使用 Pro", output.needs_pro_model],
    ].filter(([, value]) => value !== undefined && value !== null && value !== "");
  }
  if (step.step_type === "plan") {
    const subtasks = Array.isArray(output.subtasks) ? output.subtasks : [];
    return [
      ["子任务数", subtasks.length],
      ["子任务", subtasks.map((task) => `${task.task_type}: ${task.query}`).join("\n")],
    ];
  }
  if (step.step_type === "retrieve" || step.step_type === "retrieve_retry") {
    return [
      ["Query", input.queries],
      ["检索策略", input.profile],
      ["Top K", input.top_k],
      ["片段数量", output.chunk_count],
      ["置信度", output.confidence],
    ].filter(([, value]) => value !== undefined && value !== null);
  }
  if (step.step_type === "judge_evidence") {
    return [
      ["证据充分", output.is_sufficient],
      ["判断原因", output.reason],
      ["建议 Query", output.suggested_queries],
      ["建议策略", output.suggested_profile],
      ["建议拒答", output.should_refuse],
    ].filter(([, value]) => value !== undefined && value !== null && value !== "");
  }
  if (step.step_type === "tool_generate" || step.step_type === "tool_stream") {
    return [
      ["工具", input.tool],
      ["Query", input.query],
      ["置信度", output.confidence],
      ["消息", output.message],
    ].filter(([, value]) => value !== undefined && value !== null && value !== "");
  }
  return [
    ["输入", input],
    ["输出", output],
  ];
}

function formatTraceValue(value) {
  if (typeof value === "boolean") return value ? "是" : "否";
  if (Array.isArray(value)) return value.length ? value.join("\n") : "无";
  if (value && typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value ?? "无");
}

function renderSources(sources) {
  const wrap = document.createElement("div");
  wrap.className = "sources";
  sources.forEach((s, i) => {
    const det = document.createElement("details");
    const sum = document.createElement("summary");
    const label = s.heading ? `${s.source_file} · ${s.heading}` : s.source_file;
    const score = (s.final_score ?? 0).toFixed(3);
    sum.textContent = `[${i + 1}] ${label}  (score ${score})`;
    const body = document.createElement("div");
    body.className = "src-text";
    renderMarkdown(body, s.text);
    det.appendChild(sum);
    det.appendChild(body);
    wrap.appendChild(det);
  });
  return wrap;
}

// ---- Documents ----
const ALLOWED_EXTS = ["pdf", "md", "markdown", "txt"];
const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const docList = document.getElementById("doc-list");
const docCount = document.getElementById("doc-count");
const refreshBtn = document.getElementById("refresh-docs");

fileInput.addEventListener("change", async () => {
  const files = Array.from(fileInput.files || []);
  if (!files.length) return;

  const rejected = [];
  for (const f of files) {
    const ext = (f.name.split(".").pop() || "").toLowerCase();
    if (!ALLOWED_EXTS.includes(ext)) {
      rejected.push(`${f.name}：不支持的格式 .${ext}`);
    } else if (f.size > MAX_UPLOAD_BYTES) {
      const mb = (f.size / 1024 / 1024).toFixed(1);
      rejected.push(`${f.name}：${mb} MB 超过 100 MB 上限`);
    }
  }
  if (rejected.length) {
    uploadStatus.className = "upload-status err";
    uploadStatus.textContent = `已拦截：${rejected.join("；")}`;
    fileInput.value = "";
    return;
  }

  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  uploadStatus.className = "upload-status";
  uploadStatus.innerHTML = `<span class="spinner"></span> 上传 ${files.length} 个文件...`;
  try {
    const res = await fetch(`${API_BASE}/documents/upload`, { method: "POST", body: fd });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try { const errBody = await res.json(); if (errBody && errBody.detail) detail = errBody.detail; } catch {}
      throw new Error(detail);
    }
    const data = await res.json();
    uploadStatus.className = "upload-status ok";
    uploadStatus.textContent = `已上传 ${data.documents.length} 个文件`;
    fileInput.value = "";
    loadDocuments();
  } catch (err) {
    uploadStatus.className = "upload-status err";
    uploadStatus.textContent = `上传失败：${err.message}`;
  }
});

refreshBtn.addEventListener("click", loadDocuments);

async function loadDocuments() {
  docList.innerHTML = '<li class="empty-tip"><span class="spinner"></span> 加载中...</li>';
  try {
    const res = await fetch(`${API_BASE}/documents`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const docs = await res.json();
    renderDocuments(docs);
  } catch (err) {
    docList.innerHTML = `<li class="empty-tip" style="color:var(--danger)">加载失败：${err.message}</li>`;
  }
}

function renderDocuments(docs) {
  docCount.textContent = docs.length;
  if (!docs.length) {
    docList.innerHTML = '<li class="empty-tip">暂无文档</li>';
    return;
  }
  docList.innerHTML = "";
  docs.forEach((d) => {
    const li = document.createElement("li");
    li.className = "doc-item";
    li.innerHTML = `
      <div class="doc-name" title="${escapeHTML(d.filename)}">${escapeHTML(d.filename)}</div>
      <div class="doc-meta">
        <span>${d.file_type} · ${d.chunk_count} 块</span>
        <span class="right">
          <span class="status-pill ${d.status}">${d.status}</span>
          <button class="doc-del" data-id="${d.id}">删除</button>
        </span>
      </div>`;
    docList.appendChild(li);
  });
  docList.querySelectorAll("button.doc-del").forEach((btn) => {
    btn.addEventListener("click", () => deleteDocument(btn.dataset.id));
  });
}

async function deleteDocument(id) {
  if (!confirm("确认删除该文档及其分块？")) return;
  try {
    const res = await fetch(`${API_BASE}/documents/${id}`, { method: "DELETE" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    loadDocuments();
  } catch (err) {
    alert(`删除失败：${err.message}`);
  }
}

function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

loadDocuments();
