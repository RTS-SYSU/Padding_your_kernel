"use strict";
const byId = (id) => document.getElementById(id);
const number = new Intl.NumberFormat("zh-CN");
const typeButtons = [...document.querySelectorAll("[data-kind]")];
const caseNames = {
  original: "56 列 · 24–31",
  holdout: "56 列 · 24–31",
  "family-104-16-80": "104 列 · 16–23",
  "family-136-16-80": "136 列 · 16–23",
  "family-104-8-80": "104 列 · 8–15",
  top_left: "左上 40 × 40 像素",
  center: "中心 40 × 40 像素",
};
let active = "dual";
let database = [];
let configs = {};
let dualRows = {};
let liveKind = active;
const liveChoices = {};
let history = false;
let job = null;
let busy = false;
let pollTimer;
let submitting = false;
let selectedPreview = null;

function element(tag, contents, className) {
  const node = document.createElement(tag);
  if (contents !== undefined) node.textContent = contents;
  if (className) node.className = className;
  return node;
}

function options(select, entries, keep) {
  select.replaceChildren(...entries.map(([key, label]) => new Option(label, key)));
  if (entries.some(([key]) => key === keep)) select.value = keep;
}

function caseLabel(key, rows) {
  const label = caseNames[key] ?? key;
  if (active !== "dual") return label;
  if (Number.isInteger(rows)) return number.format(rows) + " 行 · " + label;
  const fallback = key === "original" ? "较小规模" : key === "holdout" ? "较大规模" : "行数待检测";
  return fallback + " · " + label;
}

function compatibleRuns() { return database.filter((run) => run.kind === active); }
function currentRun() { return compatibleRuns()[Number(byId("run").value)]; }

function showRuns(keepRun = "", keepCase = "") {
  const runs = compatibleRuns();
  const previous = runs.findIndex((run) => run.run === keepRun);
  options(byId("run"), runs.map((run, i) => [String(i), run.run]), String(previous));
  showCases(keepCase);
}

function showCases(keep = "") {
  const run = currentRun();
  options(byId("case"), Object.entries(run?.cases ?? {}).map(([key, item]) => [key, caseLabel(key, item.rows)]), keep);
  selectedPreview = null;
  byId("prediction").textContent = "";
  render();
}

function detail(label, value) {
  const row = element("div", undefined, "detail");
  row.append(element("span", label), element("strong", value));
  return row;
}

function drawPhase(target, title, pitch, offset, pavia) {
  const period = pavia ? 16 : 32;
  const phase = pitch % period;
  const units = pavia ? 48 : 64;
  const lane = element("div", undefined, "lane");
  const head = element("div", undefined, "lane-head");
  head.append(element("strong", title));
  const band = element("div", undefined, "addresses");
  band.style.setProperty("--units", units);
  const end = pavia ? phase + 32 : phase + offset + 8;
  band.setAttribute("role", "img");
  band.setAttribute("aria-label", title + "行号 1，从相位 " + phase + " 读到 " + (end - 1) + "，每 " + period + " 元素一个边界");
  for (let i = 0; i < units; ++i) {
    const cell = element("span", undefined, "cell");
    if (i && i % period === 0) cell.classList.add("edge");
    if (i >= phase && i < phase + (pavia ? 32 : 8)) cell.classList.add("first");
    if (!pavia && i >= phase + offset && i < end) cell.classList.add("second");
    band.append(cell);
  }
  lane.append(head, band);
  target.append(lane);
}

function renderLayout(c, preview) {
  const pavia = active === "pavia";
  byId("calculator").hidden = pavia;
  const { width, pad, offset } = preview ?? c;
  byId("phase-key").textContent = pavia ? "32 B / sector" : "128 B / 地址块";
  const tiles = byId("tiles");
  tiles.replaceChildren();
  [["逻辑行宽", width], ["补后行距", width + pad], ["行尾补量 p", "+" + pad]].forEach(([label, value]) => {
    const node = element("div", undefined, "tile");
    node.append(element("span", label + "（元素）"), element("strong", value));
    tiles.append(node);
  });
  const visual = byId("visual");
  visual.replaceChildren();
  drawPhase(visual, "原布局", width, offset, pavia);
  drawPhase(visual, "补后布局", width + pad, offset, pavia);
}

function comparison(pair) {
  if (!pair) return ["—", "—", "—"];
  const delta = (pair[1] - pair[0]) / pair[0] * 100;
  return [...pair.map((v) => number.format(v)), pair[0] ? (delta > 0 ? "+" : "") + delta.toFixed(2) + "%" : "—"];
}

function counter(profiles, name) {
  const values = [0, 1].map((i) => profiles.map((trial) => trial[i][name]).filter(Number.isFinite));
  if (!values.every((arm) => arm.length === profiles.length && arm.length)) return null;
  return values;
}

function profiled(profiles, name) {
  const values = counter(profiles, name);
  if (!values) return ["—", "—", "—"];
  const spans = values.map((list) => [Math.min(...list), Math.max(...list)]);
  if (spans.every(([min, max]) => min === max)) return comparison(spans.map(([min]) => min));
  const deltas = values[0].flatMap((base, i) => base ? [(values[1][i] / base - 1) * 100] : []);
  const range = (a, b, fmt) => a === b ? fmt(a) : fmt(a) + "–" + fmt(b);
  return [...spans.map(([a, b]) => range(a, b, (v) => number.format(v))),
    deltas.length === profiles.length ? range(Math.min(...deltas), Math.max(...deltas), (v) => (v > 0 ? "+" : "") + v.toFixed(2) + "%") : "—"];
}

function renderMeasured(c) {
  const pavia = active === "pavia";
  const rows = byId("metrics");
  rows.replaceChildren();
  const profiles = c?.profiles ?? [];
  const metrics = [
    [pavia ? "不同输入 sector（事前）" : "覆盖 128 B 地址块（事前）", comparison(c?.coverage)],
    ["发出读 sector（事前）", comparison(c?.expected)],
    ["发出读 sector（实测）", profiled(profiles, "issued")],
    ["L1 读 miss（sector）", counter(profiles, "l1_miss") ? profiled(profiles, "l1_miss") : ["未采集", "未采集", "—"]],
    ["L2 读请求（sector）", profiled(profiles, pavia ? "l2_read" : "l2_reads")],
    ["L2 读 miss（sector）", profiled(profiles, "l2_miss")],
    [pavia ? "输入分配（B）" : "矩阵申请量（B）", comparison(c?.bytes)],
  ];
  for (const [label, values] of metrics) {
    const row = element("tr");
    row.append(element("td", label), ...values.map((value) => element("td", value)));
    rows.append(row);
  }
  byId("measure-sub").textContent = c?.time ? "单次 CUDA kernel 时间的中位数 · " + c.samples + " 对计时" : "等待导入已完成的计时记录";
  const overview = byId("overview");
  overview.replaceChildren();
  overview.hidden = !c?.time;
  if (!c?.time) return;
  const [base, padded] = c.time;
  const ratio = element("div");
  ratio.append(element("strong", (base / padded).toFixed(3) + "×", "speed"), element("small", "实测 kernel 加速比"));
  const bars = element("div");
  for (const [label, time] of [["原布局", base], ["补后", padded]]) {
    const row = element("div", undefined, "barrow");
    const track = element("div", undefined, "bar");
    const bar = element("i");
    bar.style.width = (100 * time / Math.max(base, padded)) + "%";
    track.append(bar);
    row.append(element("span", label), track, element("b", time.toFixed(3) + " ms"));
    bars.append(row);
  }
  overview.append(ratio, bars);
}

function liveResult() {
  if (job?.status === "done" && job.kind === active && job.case === byId("launch-case").value) {
    return job.result?.cases[job.case];
  }
}


function render() {
  const run = currentRun();
  const c = history ? run?.cases[byId("case").value] : liveResult();
  byId("selects").hidden = !history || !run;
  byId("live-controls").hidden = history;
  byId("case-help").hidden = active !== "dual";
  byId("launch").disabled = submitting || busy || job?.status === "running" || !byId("launch-case").value;
  byId("launch-case").disabled = submitting || job?.status === "running" || !configs[active];
  byId("run-status").hidden = !job;
  byId("progress").hidden = job?.status !== "running";
  byId("run-status").classList.toggle("failed", job?.status === "error");
  byId("progress-label").textContent = job ? job.message : "";
  const details = byId("details");
  details.replaceChildren();
  if (c) {
    details.append(detail("输入行数", number.format(c.rows)), detail("GPU", c.device ?? "未登记"));
    if (c.roi) details.append(detail("裁剪起点（行，列）", c.roi.join("，")), detail("像素对", number.format(c.pair_count)));
    else details.append(detail("每次启动", c.rounds + " 遍扫描"), detail("第二窗口", "第 " + c.offset + " 列起"));
  }
  byId("empty-state").hidden = Boolean(c);
  byId("layout-panel").hidden = !c;
  byId("measurements").hidden = !c;
  if (c) {
    renderLayout(c, selectedPreview);
    renderMeasured(c);
    byId("metric-table").hidden = false;
  } else {
    byId("overview").replaceChildren();
    byId("metrics").replaceChildren();
    byId("tiles").replaceChildren();
    byId("visual").replaceChildren();
    byId("empty-message").textContent = history ? "选择历史实验记录查看结果。" :
      job?.status === "running" ? "正在运行，完成后显示本次结果。" : "选择实验和输入配置，运行后查看结果。";
  }
}

function showLaunchCases() {
  options(byId("launch-case"), (configs[active] ?? []).map((key) => [key, caseLabel(key, dualRows[key])]), liveChoices[active]);
}

function selectKind(kind) {
  active = kind;
  for (const button of typeButtons) button.setAttribute("aria-pressed", String(button.dataset.kind === active));
}

async function load() {
  const saved = currentRun()?.run;
  const savedCase = byId("case").value;
  try {
    const reply = await fetch("/api/runs");
    if (!reply.ok) throw new Error("请求失败");
    const data = await reply.json();
    database = data.runs;
    byId("status").textContent = data.skipped.length ? data.skipped.length + " 份记录无法解析" : database.length ? "" : "未找到实测记录。可先使用补量计算器。";
    showRuns(saved, savedCase);
  } catch {
    byId("status").textContent = "实测加载失败，请检查本地服务并刷新。";
  }
}

function adoptJob(next) {
  const isNew = next && next.run !== job?.run;
  job = next;
  if (isNew && !history) {
    liveKind = job.kind;
    liveChoices[liveKind] = job.case;
    selectKind(liveKind);
    showLaunchCases();
  }
  render();
}

async function poll() {
  clearTimeout(pollTimer);
  try {
    const response = await fetch(job ? "/api/job?run=" + encodeURIComponent(job.run) : "/api/job");
    if (!response.ok) throw Error("连接本地执行服务失败。");
    const data = await response.json();
    configs = data.configs;
    dualRows = data.dual_rows;
    busy = data.busy;
    if (!history) showLaunchCases();
    if (busy && !data.job) byId("status").textContent = "另一项实验正在运行，完成后可开始新的实验。";
    else if (!history) byId("status").textContent = "";
    if (data.job) adoptJob(data.job); else render();
  } catch (error) {
    byId("status").textContent = error.message;
  }
  if (busy || job?.status === "running") pollTimer = setTimeout(poll, 650);
}

byId("launch").addEventListener("click", async () => {
  if (submitting || busy || job?.status === "running" || !byId("launch-case").value) return;
  const selection = {kind: active, case: byId("launch-case").value};
  job = null;
  submitting = true;
  selectedPreview = null;
  byId("prediction").textContent = "";
  byId("status").textContent = "";
  render();
  try {
    const response = await fetch("/api/job", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(selection),
    });
    const result = await response.json();
    if (response.status === 409) {
      await poll();
      return;
    }
    if (!response.ok) throw Error(result.error);
    busy = true;
    adoptJob(result);
    pollTimer = setTimeout(poll, 650);
  } catch (error) {
    byId("status").textContent = error.message;
  } finally {
    submitting = false;
    render();
  }
});

byId("launch-case").addEventListener("change", () => {
  liveChoices[active] = byId("launch-case").value;
  selectedPreview = null;
  byId("prediction").textContent = "";
  render();
});

byId("history").addEventListener("click", () => {
  history = !history;
  byId("history").setAttribute("aria-pressed", String(history));
  byId("history").textContent = history ? "返回运行实验" : "历史记录";
  byId("status").textContent = "";
  selectedPreview = null;
  byId("prediction").textContent = "";
  render();
  if (history) load();
  else {
    selectKind(liveKind);
    showLaunchCases();
    render();
  }
});

for (const button of typeButtons) button.addEventListener("click", () => {
  selectKind(button.dataset.kind);
  selectedPreview = null;
  byId("prediction").textContent = "";
  if (history) showRuns();
  else {
    liveKind = active;
    showLaunchCases();
    render();
  }
});
byId("run").addEventListener("change", () => showCases());
byId("case").addEventListener("change", () => { selectedPreview = null; byId("prediction").textContent = ""; render(); });

byId("pad-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = new URLSearchParams(new FormData(event.currentTarget));
  const output = byId("prediction");
  try {
    const response = await fetch("/api/pad?" + query);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    if (active !== "dual") return;
    selectedPreview = {width: Number(query.get("width")), offset: Number(query.get("offset")), ...result};
    output.textContent = "仅更新上方推导，实测仍对应所选配置。";
    render();
  } catch (error) { output.textContent = error.message; }
});
render();
poll();

