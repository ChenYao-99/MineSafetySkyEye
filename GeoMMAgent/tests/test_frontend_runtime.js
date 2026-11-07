/* test_frontend_runtime.js — 用 node vm 跑前端页面的真实代码
 * =====================================================================
 * 静态契约检查（check_frontend_contract.py）只能证明"引用对得上"，
 * 证明不了"跑起来不炸"。本脚本把页面 <script> 抽出来，在最小 DOM 桩上执行，
 * 覆盖两件事：
 *   A. 解析期代码全部跑通（页面加载时执行的 IIFE / 顶层查询，是最易被重排打破的部分）
 *   B. ROI 画框交互回归（等比缩放 / 正反向拖拽 / 误触清除 / clearRoi）
 *
 * 用法:  node tests/test_frontend_runtime.js [路径.html]
 */
const fs = require("fs");
const vm = require("vm");

const FILE = process.argv[2] || "D:/GeoMMAgent/矿安天眼_前端演示(后端版).html";
const html = fs.readFileSync(FILE, "utf8");
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) { console.error("✗ 找不到 <script> 段"); process.exit(2); }

/* ── 最小 DOM 桩 ────────────────────────────────────────────── */
const handlers = {};           // "elId:type" → [fn]
const ctxCalls = [];           // 记录 2D 上下文调用，用于核对描边坐标
const els = {};                // id → element

function makeCtx2d() {
  const rec = (name) => (...a) => ctxCalls.push([name, ...a]);
  return {
    clearRect: rec("clearRect"), drawImage: rec("drawImage"),
    fillRect: rec("fillRect"), strokeRect: rec("strokeRect"),
    fillStyle: "", strokeStyle: "", lineWidth: 1,
  };
}

// 浏览器里 escapeHtml 是「div.textContent = s; return div.innerHTML」——
// 桩必须复现这个转义，否则断言拿到的永远是空串，"渲染出来了没"就验不了。
const esc = s => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

class El {
  constructor(id) {
    this.id = id; this.style = {}; this.dataset = {}; this.value = "";
    this.src = ""; this.files = []; this.width = 0; this.height = 0; this.disabled = false;
    this._text = ""; this._html = undefined;
    this.classList = {
      _s: new Set(),
      add(...c) { c.forEach(x => this._s.add(x)); },
      remove(...c) { c.forEach(x => this._s.delete(x)); },
      toggle(c, f) { f === undefined ? (this._s.has(c) ? this._s.delete(c) : this._s.add(c)) : (f ? this._s.add(c) : this._s.delete(c)); },
      contains(c) { return this._s.has(c); },
    };
  }
  get textContent() { return this._text; }
  set textContent(v) { this._text = String(v ?? ""); this._html = undefined; }
  get innerHTML() { return this._html !== undefined ? this._html : esc(this._text); }
  set innerHTML(v) { this._html = String(v ?? ""); }
  addEventListener(t, fn) { (handlers[`${this.id}:${t}`] ||= []).push(fn); }
  removeEventListener() {} appendChild() {} insertAdjacentHTML() {}
  click() {} scrollIntoView() {} querySelectorAll() { return []; }
  getContext() { return makeCtx2d(); }
  // 拖拽坐标用 r.width 算比例 —— 让 rect 跟着画布尺寸走
  getBoundingClientRect() { return { left: 0, top: 0, width: this.width || 320, height: this.height || 240 }; }
}

function makeEl(id) { return new El(id); }

const document = {
  getElementById(id) { return (els[id] ||= makeEl(id)); },
  querySelectorAll() { return []; },
  createElement() { return makeEl("_new"); },
};

const windowStub = { addEventListener(t, fn) { (handlers[`window:${t}`] ||= []).push(fn); } };
windowStub.window = windowStub;

class FakeImage {
  // 画布尺寸是按 img.width 算的，桩必须给出真实宽高，否则 cv.width 变 NaN
  constructor(w = 640, h = 480) { this._w = w; this._h = h; }
  get width() { return this._w; }
  get height() { return this._h; }
  set src(v) { this._src = v; if (this.onload) this.onload(); }
  get src() { return this._src; }
}

const sandbox = {
  document, window: windowStub, console,
  Image: FakeImage,
  URL: { createObjectURL: () => "blob:fake", revokeObjectURL() {} },
  FileReader: class { readAsDataURL() {} readAsText() {} },
  fetch: () => Promise.reject(new Error("stub: 无后端")),
  setInterval: () => 0, clearInterval: () => {}, setTimeout: (f) => 0,
  Blob: class {}, alert: () => {}, atob: (s) => s, btoa: (s) => s,
  requestAnimationFrame: (f) => 0,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

/* 顶层 let 不会挂到 vm 上下文上，追加一行探针把要断言的状态暴露出来 */
const probe = `
;globalThis.__probe = {
  get roiFrac() { return roiFrac; },
  get roiImg() { return roiImg; },
  showRoiEditor, drawRoi, updateRoiInfo, clearRoi,
};`;

/* ── 跑 ─────────────────────────────────────────────────────── */
let pass = 0, fail = 0;
const ok = (name, cond, extra) => {
  if (cond) { pass++; console.log(`  OK   ${name}`); }
  else { fail++; console.log(`  FAIL ${name}${extra ? "  → " + extra : ""}`); }
};

console.log(`运行时测试: ${FILE.split(/[\\/]/).pop()}\n`);

console.log("A 解析期（页面加载时执行的全部代码）");
try {
  vm.runInContext(m[1] + probe, sandbox, { filename: "page.js" });
  ok("整个 <script> 无异常执行完毕", true);
} catch (e) {
  ok("整个 <script> 无异常执行完毕", false, e.message);
  console.log("\n解析期就炸了，后续用例无法继续。");
  process.exit(1);
}
ok("探针可用（顶层 let 已暴露）", !!sandbox.__probe);

const P = sandbox.__probe;
const cv = document.getElementById("roiCanvas");
const wrap = document.getElementById("roiWrap");

console.log("\nB ROI 画框交互");

// B1 选文件 → 编辑器出现，画布等比缩放到限宽 320
P.showRoiEditor({ name: "x.png" });
ok("B1 编辑器显示", wrap.style.display === "block");
ok("B1 画布等比缩放 640×480 → 320×240", cv.width === 320 && cv.height === 240,
   `实际 ${cv.width}×${cv.height}`);

// 拖拽处理器是 IIFE 里注册的，从桩里取回来
const down = (handlers["roiCanvas:mousedown"] || [])[0];
const move = (handlers["roiCanvas:mousemove"] || [])[0];
const up = (handlers["window:mouseup"] || [])[0];
ok("B2 拖拽处理器已注册", !!(down && move && up));

const ev = (x, y) => ({ clientX: x, clientY: y, preventDefault() {} });

// B3 正向拖拽 (0,0) → (224,168)：224/320 × 168/240 = 0.7 × 0.7 = 49%
ctxCalls.length = 0;
down(ev(0, 0)); move(ev(224, 168)); up();
const r = P.roiFrac;
ok("B3 正向拖拽面积 49%", r && Math.abs(r.x0) < 1e-9 && Math.abs(r.y0) < 1e-9 &&
   Math.abs(r.x1 - 0.7) < 1e-9 && Math.abs(r.y1 - 0.7) < 1e-9, JSON.stringify(r));
let stroke = ctxCalls.find(c => c[0] === "strokeRect");
ok("B3 描边坐标 [0,0,224,168]",
   stroke && Math.abs(stroke[1]) < 1e-6 && Math.abs(stroke[2]) < 1e-6 &&
   Math.abs(stroke[3] - 224) < 1e-6 && Math.abs(stroke[4] - 168) < 1e-6, JSON.stringify(stroke));

// B3b 非原点起拖 (32,24) → (224,168)：x0/y0 应映射到 0.1，宽高 192/144
P.clearRoi();
ctxCalls.length = 0;
down(ev(32, 24)); move(ev(224, 168)); up();
const rb = P.roiFrac;
ok("B3b 非原点起拖记录正确", rb && Math.abs(rb.x0 - 0.1) < 1e-9 && Math.abs(rb.y0 - 0.1) < 1e-9 &&
   Math.abs(rb.x1 - 0.7) < 1e-9 && Math.abs(rb.y1 - 0.7) < 1e-9, JSON.stringify(rb));
stroke = ctxCalls.find(c => c[0] === "strokeRect");
ok("B3b 描边坐标 [32,24,192,144]",
   stroke && Math.abs(stroke[1] - 32) < 1e-6 && Math.abs(stroke[2] - 24) < 1e-6 &&
   Math.abs(stroke[3] - 192) < 1e-6 && Math.abs(stroke[4] - 144) < 1e-6, JSON.stringify(stroke));

// B4 反向拖拽（右下 → 左上）应与 B3b 同坐标的正向拖拽完全等价
P.clearRoi();
down(ev(224, 168)); move(ev(32, 24)); up();
const r2 = P.roiFrac;
ok("B4 反向拖拽与正向等价", r2 && Math.abs(r2.x0 - rb.x0) < 1e-9 && Math.abs(r2.y0 - rb.y0) < 1e-9 &&
   Math.abs(r2.x1 - rb.x1) < 1e-9 && Math.abs(r2.y1 - rb.y1) < 1e-9,
   JSON.stringify(r2));

// B5 面积 <1% 视为误触 → 清除
P.clearRoi();
down(ev(10, 10)); move(ev(12, 12)); up();
ok("B5 误触（面积<1%）自动清除", P.roiFrac === null, JSON.stringify(P.roiFrac));

// B6 clearRoi
down(ev(0, 0)); move(ev(160, 120)); up();
const had = !!P.roiFrac;
P.clearRoi();
ok("B6 clearRoi 生效", had && P.roiFrac === null);

/* ── C 报告槽隔离（本轮重构的核心目的）─────────────────────────
   改之前 #pipelineSteps / #pipelineReport 是单例 DOM，三条链路共用 ——
   上传检测生成的报告会渲染到「测试集」那一侧。这里正面验证它们已互相独立。 */
console.log("\nC 报告槽隔离（三条链路各写各的）");

const render = sandbox.renderPipelineResult;
ok("C0 renderPipelineResult 可从外部调用", typeof render === "function");

const fakeData = {
  is_mock: false,
  agents: [
    { name: "DetAgent", subtask: "检测子任务", output: "检测输出" },
    { name: "ReasoningAgent", subtask: "推理子任务", output: "推理输出" },
  ],
  knowledge: [{ category: "规程", title: "边坡监测规范" }],
  quality: { guard_pass: true, missing: [], repaired: false, degraded: false,
             self_eval: { overall_pass: true }, warnings: [] },
  final_answer: "**预警结论**：边坡稳定。",
};

const E = id => document.getElementById(id);
const shown = id => E(id).classList.contains("show");

render(fakeData, "upload");
ok("C1 upload 槽显示", shown("rp-upload"));
ok("C1 dataset 槽未被牵连", !shown("rp-dataset"));
ok("C1 compare 槽未被牵连", !shown("rp-compare"));
ok("C1 智能体步骤渲染进 upload 槽",
   E("rp-upload-steps").innerHTML.includes("DetAgent"),
   E("rp-upload-steps").innerHTML.slice(0, 60));
ok("C1 报告正文清理了 Markdown 符号",
   !E("rp-upload-text").textContent.includes("**") &&
   E("rp-upload-text").textContent.includes("预警结论"),
   JSON.stringify(E("rp-upload-text").textContent));
ok("C1 upload 槽有 3 个导出按钮",
   (E("rp-upload-actions").innerHTML.match(/exportReport\('upload'\)/g) || []).length === 1 &&
   E("rp-upload-actions").innerHTML.includes("exportVector('geojson','upload')"));
ok("C1 知识依据绑定到 upload 槽", shown("rp-upload-kb") &&
   E("rp-upload-kblist").innerHTML.includes("边坡监测规范"));

render(fakeData, "dataset");
// 三个场景各自保留自己的最近结果 —— 渲染 B 槽**不应**动 A 槽。
// （旧实现是单例 DOM, 后写的必然覆盖先写的, 这正是"串台"的根源。）
ok("C2 dataset 槽显示", shown("rp-dataset"));
ok("C2 upload 槽保留自己的结果（相互独立，不被覆盖）", shown("rp-upload"));
ok("C2 dataset 导出指向 dataset",
   E("rp-dataset-actions").innerHTML.includes("exportReport('dataset')") &&
   !E("rp-dataset-actions").innerHTML.includes("'upload'"));

render(fakeData, "compare");
ok("C3 compare 槽显示", shown("rp-compare"));
ok("C3 upload / dataset 槽均保留", shown("rp-upload") && shown("rp-dataset"));
// 双期导出必须使用专用接口；禁止走数据集/上传的 PNG 导出链路。
ok("C3 compare 槽提供专用 HTML/JSON 导出，不串用单期导出",
   E("rp-compare-actions").innerHTML.includes("exportCompareReport('html')") &&
   E("rp-compare-actions").innerHTML.includes("exportCompareReport('json')") &&
   !E("rp-compare-actions").innerHTML.includes("exportReport("),
   JSON.stringify(E("rp-compare-actions").innerHTML));

console.log('\nD 真实工作流状态');
render({...fakeData,agents:[{name:'ReasoningAgent',output:'单角色结果',status:'complete'}],
  quality:{workflow:{planned_steps:1,interrupted:false}}},'upload');
ok('D1 单角色计划完成不显示为断链',E('rp-upload-steps').innerHTML.includes('按需单角色分析') && !E('rp-upload-steps').innerHTML.includes('协作未完整执行'));
render({...fakeData,agents:[{name:'EvidenceAuditAgent',output:'错误核查记录',status:'invalid',validation_errors:['audit_numeric_claim_inconsistent']}],
  quality:{workflow:{planned_steps:3,interrupted:true,roles:[{name:'EvidenceAuditAgent',status:'incomplete'},
    {name:'ReasoningAgent',status:'not_executed'},{name:'ReportReviewAgent',status:'not_executed'}]}}},'upload');
ok('D2 前序失败显示中断与尚未执行角色',E('rp-upload-steps').innerHTML.includes('协作未完整执行') && E('rp-upload-steps').innerHTML.includes('尚未执行：ReportReviewAgent'));
ok('D2b 未通过的原始记录明确标记为未采用并说明原因',E('rp-upload-steps').innerHTML.includes('后续角色未采用') && E('rp-upload-steps').innerHTML.includes('量测数值或单位与工具结果不一致'));
render({...fakeData,agents:[{name:'EvidenceAuditAgent',output:'核对完成',status:'complete'},{name:'ReasoningAgent',output:'报告',status:'complete'}],
  quality:{workflow:{planned_steps:2,interrupted:false}}},'upload');
ok('D3 两个实际角色均展示',E('rp-upload-steps').innerHTML.includes('EvidenceAuditAgent') && E('rp-upload-steps').innerHTML.includes('ReasoningAgent') && E('rp-upload-steps').innerHTML.includes('多智能体协作过程'));
ok('D4 中文角色名称便于展示',E('rp-upload-steps').innerHTML.includes('证据核查') && E('rp-upload-steps').innerHTML.includes('报告推理'));
const compactAudit=vm.runInContext('agentOutputText({name:"EvidenceAuditAgent",output:JSON.stringify({decision:"needs_review",checks:["工具统计"],unknowns:[],issues:[],source_ids:[]})})',sandbox);
ok('D5 不渲染空核查段落',!compactAudit.includes('待确认条件：') && !compactAudit.includes('核查意见：') && compactAudit.includes('本次工具字段'));
const noOutside=vm.runInContext('roiDetails({area_ratio:1,risk_inside:.1,mean_prob_inside:.5,level_inside_name:"待复核",risk_outside:null,level_outside:null,outside_available:false})',sandbox);
ok('E1 范围外无观测不渲染为零风险',noOutside.includes('范围外没有可用观测') && !noOutside.includes('范围外 0.00%') && !noOutside.includes('null'));
const observedOutside=vm.runInContext('roiDetails({area_ratio:.5,risk_inside:.1,mean_prob_inside:.5,level_inside_name:"待复核",risk_outside:.02,level_outside:"yellow",outside_available:true})',sandbox);
ok('E2 有观测范围外正确显示占比',observedOutside.includes('范围外 2.00%') && !observedOutside.includes('不计算'));
console.log(`\n通过 ${pass} / 失败 ${fail}`);
process.exit(fail ? 1 : 0);
