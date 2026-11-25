"use strict";

const byId = (id) => document.getElementById(id);
const number = new Intl.NumberFormat("zh-CN");
const timingNumber = new Intl.NumberFormat("zh-CN", { minimumFractionDigits: 3, maximumFractionDigits: 3 });
const ratioNumber = new Intl.NumberFormat("zh-CN", { minimumFractionDigits: 4, maximumFractionDigits: 4 });
const state = {
  mode: "preset", presets: new Map(), pending: false, result: null, resultPreset: null, dirty: false,
  archive: null, archiveError: "", liveAvailable: null, liveReason: "", liveError: "", liveStarting: false, job: null,
};
const liveStorageKey = "padding-your-kernel-live-id";
let liveTimer;
const measurementStatuses = {
  accepted: ["通过重复加速验收", "accepted"],
  not_accepted: ["未复现一致加速", "not-accepted"],
  baseline_only: ["仅基线", ""],
};
const statuses = {
  recommend_padding: "采用行尾补量",
  keep_baseline: "保留原布局",
  unsupported: "超出模型适用范围",
};
const knownRules = {
  "Adjacent-column reuse": "相邻列复用距离",
  "Aligned right-operand stream": "右操作数对齐访问流",
  "Aligned per-stage right operands": "逐阶段右操作数对齐",
  "Signed set distance with one broadcast": "保留一条广播线的同组复用距离",
  "Two-matrix reuse with an odd line pitch": "两个矩阵以奇数缓存线为行距",
  "Stencil window already fits": "模板邻域的复用窗口满足容量约束",
  "Per-stage stencil windows already fit": "各阶段模板窗口满足容量约束",
  "Temporal reuse is in shared memory": "共享内存中的时间复用",
  "Coefficient CTA neighbor reuse": "系数计算块内的邻域复用",
  "Aligned half-line tiles": "对齐的半线分块",
  "Pivot-column group occupancy": "主元列的组占用",
  "Two rows share one tag per warp": "同一 warp 的两行共享缓存线标签",
  "No applicable rectangular pitch": "当前布局无统一矩形行距",
  "Point reuse is in registers": "寄存器中的数据点复用",
  "Global weights stream once per local row": "每个局部步骤依次读取权重行",
};
const knownEvidence = new Map([
  ["At pitch 264, only the first and last lanes can compete for the same matrix set in an adjacent-column reuse interval. With the broadcast reserve, at most two lines intervene.",
    "行距为 264 时，在相邻列复用间隔内，只有首尾 lane 可能争用同一矩阵缓存组。计入广播预留后，最多有两条线介入。"],
  ["PolyBench/GPU fixed study case. A=384 × 256. Full-square rank-k update with one broadcast and one cross-row burst per reduction step.",
    "适用于固定尺寸的 PolyBench/GPU SYRK，A 为 384 × 256。完整方阵的每步归约包含一次广播和一次跨行读取。"],
  ["At the 256-float packed pitch, the L1 reuse interval can contain seven same-set matrix tags.",
    "原行距为 256 时，L1 的有序复用间隔内可出现七条同组矩阵线。"],
  ["At the first positive 32-byte-aligned pitch, d × 264 modulo 32S can create a collision only for |d|=31.",
    "在最小的正 32 B 对齐行距下，仅当 |d|=31 时，d × 264 对 32S 取模才可能产生同组碰撞。"],
  ["The ideal L1 has 32 sets, 4 ways and 128-byte lines with direct modulo indexing and LRU replacement.",
    "理想 L1 有 32 组、4 路和 128 B 缓存线，采用取模索引和 LRU 替换。"],
  ["The ideal L2 uses a single 64-set, 16-way partition with direct modulo indexing. Device slice mapping and the stream filtered by L1 are outside this model.",
    "理想 L2 为单个 64 组、16 路的取模分区。设备分片映射与 L1 筛选后的请求流需另行建模。"],
  ["The recommendation applies to this source implementation, fixed dimensions, and its declared window.",
    "建议适用于给定实现、固定尺寸和声明的复用窗口。"],
  ["Timing, actual cache hash functions, and other interleavings are outside the analytical rule.",
    "真实计时、缓存哈希函数和访问交错须另行验证。"],
]);

function element(tag, value, className) {
  const node = document.createElement(tag);
  if (value != null) node.textContent = String(value);
  if (className) node.className = className;
  return node;
}

function setText(id, value) {
  byId(id).textContent = value ?? "—";
}

function setStatus(message, error = false) {
  setText("input-status", message);
  byId("input-status").classList.toggle("is-error", error);
}

function positiveInteger(value) {
  return Number.isSafeInteger(value) && value > 0;
}

function fmt(value) {
  return Number.isFinite(value) ? number.format(value) : "—";
}

function byteLabel(value) {
  return Number.isFinite(value) ? `${fmt(value)} B` : "—";
}

function suiteLabel(suite) {
  const name = String(suite ?? "").trim();
  if (/^polybench/i.test(name)) return "PolyBench/GPU";
  if (/^rodinia/i.test(name)) return "Rodinia";
  return name;
}

function presetMeta(preset) {
  if (!preset) return "";
  const name = suiteLabel(preset.family ?? preset.suite);
  const size = preset.size ?? preset.shape ?? preset.n;
  const shape = Array.isArray(size) ? size.map(fmt).join(" × ") :
    Number.isFinite(size) ? `N = ${fmt(size)}` : typeof size === "string" ? size : "";
  const caseLabel = preset.case && preset.case !== "default" ? preset.case : "";
  return [name, shape || caseLabel].filter(Boolean).join(" · ");
}

async function jsonRequest(path, options) {
  const response = await fetch(path, options);
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`服务返回了无效的 JSON（HTTP ${response.status}）`);
  }
  if (!response.ok) {
    const message = payload?.error ?? payload?.message ?? payload?.detail;
    const error = new Error(typeof message === "string" ? message : `请求失败（HTTP ${response.status}）`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function renderError(id, message, error = false) {
  setText(id, message);
  byId(id).classList.toggle("is-error", error);
}

function preparationLabel(preparation) {
  if (preparation === "primed") return "预热";
  if (preparation === "eviction_sweep") return "缓存清扫后";
  return preparation || "";
}

function timing(value, unit = false) {
  return Number.isFinite(value) ? `${timingNumber.format(value)}${unit ? " µs" : ""}` : "—";
}

function speedup(value) {
  return Number.isFinite(value) ? `${ratioNumber.format(value)}×` : "—";
}

function measurementContext() {
  return state.mode === "preset" && state.result && !state.dirty && !state.pending ? state.resultPreset : null;
}

function validMeasurement(row) {
  return typeof row?.preset === "string" && typeof row.operator === "string" &&
    Number.isSafeInteger(row.p) && row.p >= 0 && Object.hasOwn(measurementStatuses, row.status) &&
    Number.isFinite(row.baseline_us);
}

function matchMeasurement(row, preset) {
  if (!validMeasurement(row) || row.preset !== preset || !state.result) return false;
  return row.p === state.result.p || (row.p === 0 && row.status === "baseline_only" &&
    state.result.status === "unsupported" && state.result.p == null);
}

function metadataLabel(meta) {
  return [meta.device, preparationLabel(meta.preparation), positiveInteger(meta.processes) ?
    `${fmt(meta.processes)} 个独立进程` : null].filter(Boolean).join(" · ");
}

function trialDescription(meta) {
  return positiveInteger(meta.rounds) && positiveInteger(meta.samples_per_round) ?
    `每个进程 ${fmt(meta.rounds)} 轮，每轮每种布局 ${fmt(meta.samples_per_round)} 个样本。` : "";
}

function showTiming(prefix, row, meta, provenance, message = "", error = false) {
  const hasRow = Boolean(row);
  byId(`${prefix}-data`).hidden = !hasRow;
  byId(`${prefix}-verdict`).hidden = !hasRow;
  setText(`${prefix}-device`, metadataLabel(meta));
  renderError(`${prefix}-message`, message, error);
  if (!hasRow) return;

  const verdict = measurementStatuses[row.status];
  const verdictNode = byId(`${prefix}-verdict`);
  verdictNode.className = `verdict ${verdict?.[1] || ""}`;
  verdictNode.textContent = verdict?.[0] || "已完成";
  const stats = [["基线 p=0", timing(row.baseline_us, true)]];
  const processes = meta.processes || row.process_speedups?.length;
  if (row.p > 0) {
    stats.push([`补后 p=${fmt(row.p)}`, timing(row.padded_us, true)], ["配对加速比", speedup(row.speedup)]);
    if (positiveInteger(processes) && Number.isInteger(row.faster_processes)) {
      verdictNode.textContent += ` · ${row.faster_processes}/${processes}`;
    }
  }
  byId(`${prefix}-stats`).replaceChildren(...stats.map(([label, value]) => {
    const cell = element("div");
    cell.append(element("dt", label), element("dd", value));
    return cell;
  }));
  const ratios = Array.isArray(row.process_speedups) ? row.process_speedups : [];
  byId(`${prefix}-processes`).hidden = !ratios.length;
  byId(`${prefix}-ratios`).replaceChildren(...ratios.map((value, index) =>
    element("li", `进程 ${index + 1} · ${speedup(value)}`)));
  const launchText = positiveInteger(row.launches) ? `计时覆盖 ${fmt(row.launches)} 次内核启动。` : "";
  const checks = row.correctness_checks ?? (prefix === "live-result" ? meta.correctness_checks : null);
  const correctnessText = positiveInteger(checks) ? `${fmt(checks)} 项输出检查全部通过。` : "";
  const comparison = row.p === 0 ? "此配置仅计时基线。" : "配对加速比取各进程成对统计量的中位数。";
  setText(`${prefix}-method`, [provenance, launchText, trialDescription(meta), correctnessText, comparison]
    .filter(Boolean).join(" "));
}

function showArchiveCard() {
  const preset = measurementContext();
  byId("native-measurements").hidden = !preset;
  if (!preset) return;
  if (state.archiveError) {
    showTiming("native", null, {}, "", "无法加载归档计时。可在全部实测区域重试。", true);
    return;
  }
  if (!state.archive) {
    showTiming("native", null, {}, "", "正在加载归档计时…");
    return;
  }
  const row = state.archive.measurements.find((item) => item.preset === preset);
  const match = matchMeasurement(row, preset);
  showTiming("native", match ? row : null, state.archive,
    "归档快照与此预设的源码和输入尺寸对应。",
    match ? "" : row ? "归档布局与当前推导不同，参见全部实测。" : "此预设没有归档计时。");
}

function archiveRow(row, meta) {
  const tr = element("tr");
  const main = element("th");
  main.scope = "row";
  const preset = state.presets.get(row.preset);
  if (preset) {
    const button = element("button", row.operator, "summary-preset");
    button.type = "button";
    button.setAttribute("aria-label", `查看 ${row.operator} ${preset.size || ""} 的分析`);
    button.addEventListener("click", async () => {
      if (state.pending) return;
      state.mode = "preset";
      byId("preset").value = preset.name;
      setText("preset-meta", presetMeta(preset));
      inputChanged();
      await analyze();
      if (measurementContext() === preset.name) byId("native-measurements").scrollIntoView({ block: "start" });
    });
    main.append(button);
  } else {
    main.append(element("span", row.operator));
  }
  main.append(element("small", preset?.size || (row.case !== "default" ? row.case : "")));
  tr.append(main, element("td", fmt(row.p)), element("td", timing(row.baseline_us)),
    element("td", timing(row.padded_us)), element("td", speedup(row.speedup)),
    element("td", Number.isInteger(row.faster_processes) ? `${row.faster_processes}/${meta.processes}` : "—"));
  const verdict = measurementStatuses[row.status];
  const cell = element("td");
  cell.append(element("span", verdict?.[0] || row.status, `verdict ${verdict?.[1] || ""}`));
  tr.append(cell);
  return tr;
}

function showArchiveSummary() {
  const data = state.archive;
  byId("summary-content").hidden = !data;
  byId("retry-measurements").hidden = !state.archiveError;
  renderError("summary-message", state.archiveError || (data ? "" : "正在加载已归档的 GPU 结果…"), Boolean(state.archiveError));
  if (!data) return;
  setText("summary-meta", [metadataLabel(data), data.experiment, data.date].filter(Boolean).join(" · "));
  const counts = [["个配置", data.measurements.length], ["个通过", 0], ["个未复现一致加速", 0], ["个仅基线", 0]];
  for (const row of data.measurements) {
    const index = ["accepted", "not_accepted", "baseline_only"].indexOf(row.status);
    if (index !== -1) counts[index + 1][1] += 1;
  }
  byId("summary-totals").replaceChildren(...counts.map(([label, count]) => {
    const item = element("li");
    item.append(element("strong", fmt(count)), document.createTextNode(` ${label}`));
    return item;
  }));
  byId("summary-rows").replaceChildren(...data.measurements.map((row) => archiveRow(row, data)));
  setText("summary-method", `已验证的归档快照。各行对应既定源码和输入尺寸，时间为 CUDA 事件计时。${trialDescription(data)}全部进程配对中位数均大于 1 才通过。`);
}

async function loadMeasurements() {
  state.archiveError = "";
  showArchiveSummary();
  try {
    const data = await jsonRequest("/api/measurements");
    if (!Array.isArray(data?.measurements) || !data.measurements.length ||
        !data.measurements.every(validMeasurement)) throw new Error("结果列表缺少必要字段。");
    state.archive = data;
  } catch (error) {
    state.archiveError = `无法加载已归档的 GPU 结果。${error.message || ""}`;
  }
  showArchiveSummary();
  showArchiveCard();
}

function showLive() {
  const button = byId("run-live");
  const custom = state.mode !== "preset";
  const job = state.job;
  const running = state.liveStarting || job?.status === "running";
  const selectedPreset = custom ? null : state.presets.get(byId("preset").value);
  byId("live-controls").hidden = custom && !running;
  button.hidden = custom;
  button.disabled = running || state.pending || !state.liveAvailable || !selectedPreset;
  byId("live-progress").hidden = !running;
  const selected = job && state.presets.get(job.preset)?.operator;
  const jobMessage = job?.message || job?.stage || "";
  let status = state.liveReason;
  if (state.liveAvailable == null) status = "正在检查本机运行能力…";
  if (state.liveStarting) status = "正在提交真机实验…";
  else if (job) {
    const stage = { running: "运行中", completed: "已完成", failed: "运行失败" }[job.status];
    status = [selected || job.preset, stage, jobMessage].filter(Boolean).join(" · ");
  }
  renderError("live-status", [status, state.liveError].filter(Boolean).join(" "),
    Boolean(state.liveError) || job?.status === "failed");

  const matchesSelection = Boolean(selectedPreset && job && job.preset === selectedPreset.name && !state.liveStarting);
  byId("live-result").hidden = !matchesSelection;
  if (!matchesSelection) return;
  if (job.status === "completed") {
    const analyzed = measurementContext() === selectedPreset.name;
    const match = analyzed ? matchMeasurement(job.measurement, selectedPreset.name) :
      validMeasurement(job.measurement) && job.measurement.preset === selectedPreset.name;
    showTiming("live-result", match ? job.measurement : null, job,
      `${selectedPreset.operator} · ${presetMeta(selectedPreset)}。由当前真机实验采集。`,
      match ? "" : "本次计时的预设或布局与当前分析不一致。", !match);
  } else {
    showTiming("live-result", null, job, "", jobMessage || (running ? "正在运行真机实验…" : "真机实验未完成。"),
      job.status === "failed");
  }
}

function savedLiveId(value) {
  try {
    if (value === null) window.sessionStorage.removeItem(liveStorageKey);
    if (value) window.sessionStorage.setItem(liveStorageKey, value);
    return window.sessionStorage.getItem(liveStorageKey);
  } catch {
    return null;
  }
}

function followLive(job) {
  if (!job?.id || !job.preset || !["running", "completed", "failed"].includes(job.status)) {
    throw new Error("真机实验返回了无效状态。");
  }
  state.job = job;
  state.liveError = "";
  savedLiveId(job.id);
  window.clearTimeout(liveTimer);
  if (job.status === "running") liveTimer = window.setTimeout(pollLive, 1500);
  showLive();
}

async function pollLive() {
  const id = state.job?.id;
  if (!id || state.job.status !== "running") return;
  try {
    const job = await jsonRequest(`/api/live/${encodeURIComponent(id)}`);
    if (state.job?.id === id) followLive(job);
  } catch (error) {
    if (state.job?.id !== id) return;
    if (error.status === 404 || error.status === 410) {
      savedLiveId(null);
      state.job = null;
      state.liveError = "当前服务中已无该实验记录。";
      await loadLiveCapability();
      return;
    }
    state.liveError = `暂时无法更新状态。${error.message || ""}`;
    showLive();
    liveTimer = window.setTimeout(pollLive, 3500);
  }
}

async function loadLiveCapability() {
  try {
    const data = await jsonRequest("/api/live");
    state.liveAvailable = data.available === true;
    state.liveReason = state.liveAvailable ? "" : (data.reason || "当前环境无法运行真机实验。");
    if (data.active) followLive(data.active);
    else if (!state.job) {
      const id = savedLiveId();
      if (id) {
        try {
          followLive(await jsonRequest(`/api/live/${encodeURIComponent(id)}`));
        } catch (error) {
          if (error.status === 404 || error.status === 410) savedLiveId(null);
        }
      }
    }
  } catch (error) {
    state.liveAvailable = false;
    state.liveReason = `无法检查本机运行能力。${error.message || ""}`;
  }
  showLive();
}

async function startLive() {
  const preset = byId("preset").value;
  if (byId("run-live").disabled || !state.presets.has(preset)) return;
  state.liveStarting = true;
  state.liveError = "";
  showLive();
  if (measurementContext() !== preset) void analyze();
  try {
    const job = await jsonRequest("/api/live", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ preset }),
    });
    followLive(job);
  } catch (error) {
    state.liveError = `无法启动真机实验。${error.message || ""}`;
    await loadLiveCapability();
  } finally {
    state.liveStarting = false;
    showLive();
  }
}

function updateControls() {
  const custom = state.mode === "adjacent-columns";
  byId("preset-fields").hidden = custom;
  byId("custom-fields").hidden = !custom;
  byId("preset-mode").setAttribute("aria-pressed", String(!custom));
  byId("custom-mode").setAttribute("aria-pressed", String(custom));
  byId("preset").disabled = custom || state.pending || !state.presets.size;
  for (const control of byId("custom-fields").querySelectorAll("input, select")) {
    control.disabled = !custom || state.pending;
  }
  byId("analyze").disabled = state.pending || (!custom && !state.presets.size);
  byId("download").disabled = !state.result || state.pending || state.dirty;
  byId("analysis").setAttribute("aria-busy", String(state.pending));
  showArchiveCard();
  showLive();
}

function inputChanged() {
  state.dirty = Boolean(state.result);
  setStatus(state.dirty ? "输入已修改，点击分析布局更新结果。" : "");
  updateControls();
}

function diagram(title, layout, pitch) {
  const card = element("div", null, "diagram");
  const header = element("div", null, "diagram-head");
  header.append(element("strong", title), element("span", `${fmt(pitch)} 元素/行`));
  card.append(header);
  const tail = pitch - layout.logical_width;
  for (let row = 0; row < Math.min(layout.rows, 3); row += 1) {
    const line = element("div", null, "diagram-row");
    const bar = element("div", null, "diagram-bar");
    bar.setAttribute("aria-hidden", "true");
    bar.style.setProperty("--bar-width", `${100 * pitch / layout.physical_pitch}%`);
    bar.append(element("i", null, "bar-data"));
    if (tail > 0) {
      bar.style.setProperty("--pad-width", `${100 * tail / pitch}%`);
      bar.append(element("i", null, "bar-pad"));
    }
    const start = element("span", `+${fmt(row * pitch)}`, "row-start");
    start.setAttribute("aria-label", `行首相对元素偏移 ${fmt(row * pitch)}`);
    line.append(element("span", `行 ${row}`, "quiet"), bar, start);
    card.append(line);
  }
  return card;
}

function layoutLabel(layout) {
  return layout.operand || "目标数组";
}

function showLayout() {
  const layout = state.result?.layouts?.[Number(byId("operand").value || 0)];
  const canDraw = layout && positiveInteger(layout.logical_width) &&
    positiveInteger(layout.rows) && positiveInteger(layout.physical_pitch) &&
    layout.physical_pitch >= layout.logical_width;
  byId("layout-section").hidden = !canDraw;
  setText("pitch-value", canDraw ? `${fmt(layout.logical_width)} → ${fmt(layout.physical_pitch)}` : "—");
  setText("pitch-note", canDraw ? `${layoutLabel(layout)} · ${fmt(layout.rows)} 行` : "");
  if (!canDraw) return;

  const tail = layout.physical_pitch - layout.logical_width;
  byId("diagrams").replaceChildren(
    diagram("原布局", layout, layout.logical_width),
    diagram(tail > 0 ? "补后布局" : "保留布局", layout, layout.physical_pitch),
  );
  setText("operand-name", state.result.layouts.length === 1 ? layoutLabel(layout) : "");
  byId("pad-legend").hidden = tail <= 0;
  setText("diagram-note", layout.rows > 3 ? "显示前 3 行，右侧为行首相对元素偏移" : "右侧为行首相对元素偏移");
}

function showEvidence(result) {
  const localize = (item) => knownEvidence.get(item) ?? item;
  const makeList = (target, value) => {
    const items = Array.isArray(value) ? value : value ? [value] : [];
    byId(target).replaceChildren(...items.map((item) =>
      element("li", typeof item === "string" ? localize(item) : JSON.stringify(item))));
    return items.length;
  };
  setText("rule", knownRules[result.rule] || result.rule);
  setText("reason", localize(result.reason));
  setText("scope", localize(result.scope));
  setText("source", result.source);
  byId("source-row").hidden = !result.source;
  byId("fact-section").hidden = !makeList("facts", result.facts);
  byId("assumption-section").hidden = !makeList("assumptions", result.assumptions);
  byId("operand-rows").replaceChildren(...result.layouts.map((layout) => {
    const row = element("tr");
    row.append(element("th", layoutLabel(layout)), element("td", fmt(layout.rows)),
      element("td", fmt(layout.logical_width)), element("td", fmt(layout.physical_pitch)),
      element("td", byteLabel(layout.extra_bytes)));
    row.firstChild.scope = "row";
    return row;
  }));
  byId("operands-section").hidden = !result.layouts.length;
  byId("evidence").hidden = false;
}

function showResult(result, preset) {
  if (!result || typeof result.operator !== "string" || !Object.hasOwn(statuses, result.status) ||
      (result.p != null && !(Number.isSafeInteger(result.p) && result.p >= 0)) ||
      !Array.isArray(result.layouts)) {
    throw new Error("分析结果缺少必要字段");
  }
  state.result = result;
  state.resultPreset = preset?.name || null;
  state.dirty = false;
  byId("result-empty").hidden = true;
  byId("result-body").hidden = false;
  setText("operator", preset ? result.operator : "自定义相邻列");
  const first = result.layouts[0];
  const shape = first && positiveInteger(first.rows) && positiveInteger(first.logical_width) ?
    `${fmt(first.rows)} × ${fmt(first.logical_width)}` : "";
  setText("result-meta", preset ? presetMeta(preset) || shape : ["相邻列模型", shape].filter(Boolean).join(" · "));
  setText("padding-value", result.p == null ? "—" : `p = ${fmt(result.p)}`);
  setText("result-status", statuses[result.status]);
  setText("bytes-value", byteLabel(result.extra_bytes));
  setText("bytes-note", result.layouts.length ? `${fmt(result.layouts.length)} 个操作数合计` : "");
  byId("operand").replaceChildren(...result.layouts.map((layout, index) => {
    const option = element("option", layoutLabel(layout));
    option.value = String(index);
    return option;
  }));
  byId("operand-label").hidden = result.layouts.length <= 1;
  showLayout();
  showEvidence(result);
}

async function analyze() {
  if (state.pending) return;
  const isPreset = state.mode === "preset";
  const preset = state.presets.get(byId("preset").value);
  if (isPreset && !preset) return;
  if (!isPreset && !byId("analysis-form").reportValidity()) return;
  const input = isPreset ? { mode: "preset", preset: preset.name } : {
    mode: "adjacent-columns",
    width: byId("width").valueAsNumber,
    rows: byId("rows").valueAsNumber,
    reserved_lines: byId("reserved").valueAsNumber,
    base_alignment_bytes: Number(byId("alignment").value),
  };
  if (!isPreset && ![input.width, input.rows, input.reserved_lines].every(Number.isSafeInteger)) {
    setStatus("请输入浏览器可精确表示的整数。", true);
    return;
  }
  state.pending = true;
  setStatus("正在推导布局…");
  updateControls();
  try {
    const result = await jsonRequest("/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(input),
    });
    showResult(result, isPreset ? preset : null);
    setStatus("分析完成。");
  } catch (error) {
    state.dirty = Boolean(state.result);
    setStatus(error.message || "分析失败，请重试。", true);
    if (!state.result) {
      setText("operator", "未生成分析");
      setText("result-empty", "分析失败，请检查输入并重试。");
    }
  } finally {
    state.pending = false;
    updateControls();
  }
}

async function loadPresets() {
  byId("retry-presets").hidden = true;
  try {
    const payload = await jsonRequest("/api/presets");
    if (!Array.isArray(payload?.presets) || !payload.presets.length) throw new Error("没有可用的算子预设");
    const groups = new Map();
    state.presets = new Map(payload.presets.map((preset) => [preset.name, preset]));
    for (const preset of state.presets.values()) {
      const label = suiteLabel(preset.family ?? preset.suite) || "其他";
      if (!groups.has(label)) {
        const group = element("optgroup");
        group.label = label;
        groups.set(label, group);
      }
      const choice = element("option", preset.label || [preset.operator || preset.name, preset.size]
        .filter(Boolean).join(" · "));
      choice.value = preset.name;
      groups.get(label).append(choice);
    }
    byId("preset").replaceChildren(...groups.values());
    const defaultPreset = [...state.presets.values()].find((preset) =>
      /^syrk$/i.test(preset.operator) || /^syrk(?:$|[/:.-])/i.test(preset.name));
    if (defaultPreset) byId("preset").value = defaultPreset.name;
    setText("preset-meta", presetMeta(state.presets.get(byId("preset").value)));
    showArchiveSummary();
    updateControls();
    if (!state.result && state.mode === "preset") await analyze();
  } catch (error) {
    byId("retry-presets").hidden = false;
    setStatus(`无法加载预设。${error.message || ""}`, true);
    if (!state.result) {
      setText("operator", "等待输入");
      setText("result-empty", "重新加载算子预设，或使用相邻列分析。");
    }
    updateControls();
  }
}

function changeMode(mode) {
  if (state.mode === mode || state.pending) return;
  state.mode = mode;
  inputChanged();
}

byId("preset-mode").addEventListener("click", () => changeMode("preset"));
byId("custom-mode").addEventListener("click", () => changeMode("adjacent-columns"));
byId("preset").addEventListener("change", () => {
  setText("preset-meta", presetMeta(state.presets.get(byId("preset").value)));
  inputChanged();
});
byId("custom-fields").addEventListener("input", inputChanged);
byId("analysis-form").addEventListener("submit", (event) => {
  event.preventDefault();
  void analyze();
});
byId("operand").addEventListener("change", showLayout);
byId("retry-presets").addEventListener("click", loadPresets);
byId("retry-measurements").addEventListener("click", loadMeasurements);
byId("run-live").addEventListener("click", startLive);
byId("download").addEventListener("click", () => {
  if (!state.result || state.dirty || state.pending) return;
  const file = new Blob([JSON.stringify(state.result, null, 2) + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(file);
  const anchor = element("a");
  anchor.href = url;
  const name = state.result.operator.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  anchor.download = `padding-${name || "report"}.json`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 500);
});

updateControls();
void loadPresets();
void loadMeasurements();
void loadLiveCapability();
