"use strict";

const byId = (id) => document.getElementById(id);
const number = new Intl.NumberFormat("zh-CN");
const timingNumber = new Intl.NumberFormat("zh-CN", { minimumFractionDigits: 3, maximumFractionDigits: 3 });
const ratioNumber = new Intl.NumberFormat("zh-CN", { minimumFractionDigits: 4, maximumFractionDigits: 4 });
const state = {
  mode: "preset", presets: new Map(), pending: false, result: null, resultPreset: null, dirty: false,
  liveAvailable: null, liveReason: "", liveError: "", liveStarting: false, job: null,
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
const unsupportedReasons = new Map([
  ["Logical width must be a multiple of 2048 FP32 elements for the row-stride term to cancel at both levels.",
    "逻辑行宽须为 2048 个 FP32 元素的倍数。"],
  ["The custom rule requires at least 32 rows so its full-warp certificate applies.",
    "相邻列模型须至少有 32 行。"],
  ["The stated base alignment must guarantee a 32-byte aligned row start.",
    "基址须保证行首按 32 B 对齐。"],
  ["The reserved competing lines occupy every L1 way in the worst case.",
    "保留缓存线已占满理想 L1 的四路。"],
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

function timing(value) {
  return Number.isFinite(value) ? `${timingNumber.format(value)} µs` : "—";
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

function showLiveTiming(row, meta, provenance, message = "", error = false) {
  const hasRow = Boolean(row);
  byId("live-result-data").hidden = !hasRow;
  byId("live-result-verdict").hidden = !hasRow;
  setText("live-result-device", metadataLabel(meta));
  renderError("live-result-message", message, error);
  if (!hasRow) return;

  const verdict = measurementStatuses[row.status];
  const verdictNode = byId("live-result-verdict");
  verdictNode.className = `verdict ${verdict?.[1] || ""}`;
  verdictNode.textContent = verdict?.[0] || "已完成";
  const stats = [["基线 p=0", timing(row.baseline_us)]];
  const processes = meta.processes || row.process_speedups?.length;
  if (row.p > 0) {
    stats.push([`补后 p=${fmt(row.p)}`, timing(row.padded_us)], ["配对加速比", speedup(row.speedup)]);
    if (positiveInteger(processes) && Number.isInteger(row.faster_processes)) {
      verdictNode.textContent += ` · ${row.faster_processes}/${processes}`;
    }
  }
  byId("live-result-stats").replaceChildren(...stats.map(([label, value]) => {
    const cell = element("div");
    cell.append(element("dt", label), element("dd", value));
    return cell;
  }));
  const ratios = Array.isArray(row.process_speedups) ? row.process_speedups : [];
  byId("live-result-processes").hidden = !ratios.length;
  byId("live-result-ratios").replaceChildren(...ratios.map((value, index) =>
    element("li", `进程 ${index + 1} · ${speedup(value)}`)));
  const launchText = positiveInteger(row.launches) ? `计时覆盖 ${fmt(row.launches)} 次内核启动。` : "";
  const checks = row.correctness_checks ?? meta.correctness_checks;
  const correctnessText = positiveInteger(checks) ? `${fmt(checks)} 项输出检查全部通过。` : "";
  const comparison = row.p === 0 ? "此配置仅计时基线。" : "配对加速比取各进程成对统计量的中位数。";
  setText("live-result-method", [provenance, launchText, trialDescription(meta), correctnessText, comparison]
    .filter(Boolean).join(" "));
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
    showLiveTiming(match ? job.measurement : null, job,
      `${selectedPreset.operator} · ${presetMeta(selectedPreset)}。由当前真机实验采集。`,
      match ? "" : "本次计时的预设或布局与当前分析不一致。", !match);
  } else {
    showLiveTiming(null, job, "", jobMessage || (running ? "正在运行真机实验…" : "真机实验未完成。"),
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
  const unsupported = result.status === "unsupported";
  byId("analysis-reason").hidden = !unsupported;
  setText("analysis-reason", unsupported ? unsupportedReasons.get(result.reason) ?? result.reason ?? "" : "");
  setText("bytes-value", byteLabel(result.extra_bytes));
  setText("bytes-note", result.layouts.length ? `${fmt(result.layouts.length)} 个操作数合计` : "");
  byId("operand").replaceChildren(...result.layouts.map((layout, index) => {
    const option = element("option", layoutLabel(layout));
    option.value = String(index);
    return option;
  }));
  byId("operand-label").hidden = result.layouts.length <= 1;
  showLayout();
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
void loadLiveCapability();
