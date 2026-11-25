"use strict";

const byId = (id) => document.getElementById(id);
const number = new Intl.NumberFormat("zh-CN");
const state = { mode: "preset", presets: new Map(), pending: false, result: null, dirty: false };
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
    throw new Error(typeof message === "string" ? message : `请求失败（HTTP ${response.status}）`);
  }
  return payload;
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
