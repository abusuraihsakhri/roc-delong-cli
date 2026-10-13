"use strict";

/* All calculations and CSV parsing run locally in the browser. */
const ROC = (() => {
  function parseCSV(text) {
    const rows = [];
    let row = [], cell = "", quoted = false, closed = false;
    text = text.replace(/^\uFEFF/, "");
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (quoted) {
        if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
        else if (ch === '"') { quoted = false; closed = true; }
        else { cell += ch; }
      } else if (ch === '"') {
        if (cell.length || closed) throw new Error("Malformed CSV quoting.");
        quoted = true;
      } else if (ch === ",") {
        row.push(cell); cell = ""; closed = false;
      } else if (ch === "\n" || ch === "\r") {
        if (ch === "\r" && text[i + 1] === "\n") i++;
        row.push(cell); cell = ""; closed = false;
        if (row.some(v => v.trim() !== "")) rows.push(row);
        row = [];
      } else {
        if (closed && !/\s/.test(ch)) throw new Error("Unexpected characters after a quoted CSV field.");
        cell += ch;
      }
    }
    if (quoted) throw new Error("Unclosed quoted CSV field.");
    row.push(cell);
    if (row.some(v => v.trim() !== "")) rows.push(row);
    if (rows.length < 2) throw new Error("CSV must include headers and at least one data row.");
    if (rows.length > 20001) throw new Error("Limit is 20,000 data rows per analysis.");
    const headers = rows.shift().map(v => v.trim());
    if (headers.some(h => !h) || new Set(headers).size !== headers.length)
      throw new Error("CSV header names must be nonempty and unique.");
    if (rows.some(r => r.length !== headers.length))
      throw new Error("CSV rows must contain exactly the same number of columns as the header.");
    return { headers, rows };
  }

  function numericalData(dataset, labelColumn, scoreA, scoreB, positiveLabel) {
    const li = dataset.headers.indexOf(labelColumn);
    const ai = dataset.headers.indexOf(scoreA);
    const bi = scoreB ? dataset.headers.indexOf(scoreB) : -1;
    if (li < 0 || ai < 0 || (scoreB && bi < 0)) throw new Error("Selected column not found.");
    if (li === ai || li === bi) throw new Error("Labels and scores must be in different columns.");
    const labels = [], a = [], b = [];
    for (const [i, row] of dataset.rows.entries()) {
      const raw = [row[ai], ...(bi >= 0 ? [row[bi]] : [])];
      if (raw.some(v => v.trim() === "")) throw new Error("Missing score at CSV row " + (i + 2) + ".");
      const nums = raw.map(Number);
      if (nums.some(v => !Number.isFinite(v))) throw new Error("Invalid/nonfinite score at CSV row " + (i + 2) + ".");
      labels.push(row[li].trim() === positiveLabel);
      a.push(nums[0]);
      if (bi >= 0) b.push(nums[1]);
    }
    const m = labels.filter(Boolean).length, n = labels.length - m;
    if (m < 2 || n < 2) throw new Error("DeLong inference requires at least two subjects per class.");
    return { labels, a, b: bi >= 0 ? b : null, m, n };
  }

  function bound(values, target, upper) {
    let lo = 0, hi = values.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (values[mid] < target || (upper && values[mid] === target)) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  function components(scores, labels) {
    const x = [], y = [];
    scores.forEach((s, i) => (labels[i] ? x : y).push(s));
    const sx = x.slice().sort((a, b) => a - b), sy = y.slice().sort((a, b) => a - b);
    const v10 = x.map(v => (bound(sy, v, false) + bound(sy, v, true)) / (2 * y.length));
    const v01 = y.map(v => 1 - (bound(sx, v, false) + bound(sx, v, true)) / (2 * x.length));
    return { v10, v01, auc: mean(v10) };
  }

  function mean(values) { return values.reduce((s, v) => s + v, 0) / values.length; }
  function sampleVariance(values) {
    if (values.length < 2) throw new Error("At least two observations per class are required.");
    const avg = mean(values);
    return values.reduce((s, v) => s + (v - avg) ** 2, 0) / (values.length - 1);
  }
  function variance(c) { return sampleVariance(c.v10) / c.v10.length + sampleVariance(c.v01) / c.v01.length; }
  function curve(scores, labels) {
    const pairs = scores.map((score, i) => ({score, positive: labels[i]}));
    pairs.sort((a, b) => b.score - a.score);
    const m = labels.filter(Boolean).length, n = labels.length - m;
    const points = [{threshold: Infinity, fpr: 0, tpr: 0, j: 0}];
    let tp = 0, fp = 0;
    for (let i = 0; i < pairs.length;) {
      const threshold = pairs[i].score;
      while (i < pairs.length && pairs[i].score === threshold) {
        pairs[i].positive ? tp++ : fp++; i++;
      }
      const tpr = tp / m, fpr = fp / n;
      points.push({threshold, fpr, tpr, j: tpr - fpr});
    }
    const best = points.slice(1).reduce((b, p) => p.j > b.j ? p : b);
    return {points, best};
  }
  // Complementary error function: stable two-sided normal-tail approximation.
  function erfc(x) {
    const t = 1 / (1 + 0.5 * Math.abs(x));
    const p = t * Math.exp(-x*x - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
      t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 +
      t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? p : 2 - p;
  }
  function analyze(data, paired = false) {
    const {labels, a, b, m, n} = data;
    if (!Array.isArray(labels) || !Array.isArray(a) || labels.length !== a.length ||
        (paired && (!Array.isArray(b) || b.length !== a.length)))
      throw new Error("Scores and labels must have matching lengths.");
    if (m < 2 || n < 2 || m + n !== a.length) throw new Error("At least two subjects per class are required.");
    if (![...a, ...(paired ? b : [])].every(Number.isFinite))
      throw new Error("Scores must be finite.");
    const cA = components(a, labels), rocA = curve(a, labels);
    const ci = c => {
      const se = Math.sqrt(variance(c));
      return {lo: Math.max(0, c.auc - 1.95996398454 * se),
        hi: Math.min(1, c.auc + 1.95996398454 * se), se};
    };
    const result = {m, n, a: {...cA, ...rocA, ci: ci(cA)}};
    if (paired) {
      const cB = components(b, labels), rocB = curve(b, labels);
      result.b = {...cB, ...rocB, ci: ci(cB)};
      const d10 = cA.v10.map((x, i) => x - cB.v10[i]);
      const d01 = cA.v01.map((x, i) => x - cB.v01[i]);
      const v = sampleVariance(d10) / m + sampleVariance(d01) / n;
      const diff = cA.auc - cB.auc;
      if (v === 0 && Math.abs(diff) > 1e-12)
        throw new Error("AUC difference has zero estimated variance; the DeLong z-test is undefined.");
      result.comparison = {
        diff, varDiff: v, z: v ? diff / Math.sqrt(v) : 0,
        p: v ? Math.min(1, Math.max(0, erfc(Math.abs(diff / Math.sqrt(v)) / Math.SQRT2))) : 1
      };
    }
    return result;
  }
  return {parseCSV, numericalData, analyze};
})();

if (typeof module !== "undefined" && module.exports) module.exports = ROC;

if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", () => {
  "use strict";
  const $ = id => document.getElementById(id);
  let dataset = null, outcome = null, exported = [], chart = null;
  const setStatus = (message, error = false) => {
    $("status").textContent = message;
    $("status").classList.toggle("error", error);
  };
  function setColumns() {
    const names = dataset.headers;
    const select = (id, selected) => {
      const el = $(id);
      el.replaceChildren();
      names.forEach(name => {
        const option = document.createElement("option");
        option.value = name; option.textContent = name;
        el.append(option);
      });
      el.value = selected;
    };
    const label = names.find(n => /label|status|outcome|disease/i.test(n)) || names[0];
    const scoreCols = names.filter(n => n !== label);
    if (!scoreCols.length) throw new Error("CSV requires a label and a score column.");
    select("label", label);
    select("scoreA", scoreCols[0]);
    select("scoreB", scoreCols[1] || scoreCols[0]);
    function updateLabels() {
      const idx = names.indexOf($("label").value);
      const values = [...new Set(dataset.rows.map(r => r[idx].trim()))];
      $("positive").replaceChildren();
      values.forEach(v => {
        const option = document.createElement("option");
        option.value = v; option.textContent = v || "(empty)";
        $("positive").append(option);
      });
      $("positive").value = values.includes("1") ? "1" : values[0];
    }
    $("label").onchange = updateLabels;
    updateLabels();
    $("fileDetails").textContent = dataset.rows.length + " rows · " + names.length + " columns";
  }
  function load(text, source) {
    dataset = ROC.parseCSV(text);
    setColumns();
    outcome = null; exported = [];
    $("summary").replaceChildren();
    $("plot").replaceChildren();
    $("preview").replaceChildren();
    $("downloadCsv").disabled = true;
    $("downloadSvg").disabled = true;
    setStatus("Loaded " + source + ". Select columns and click Analyze.");
  }
  $("file").addEventListener("change", async e => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      if (file.size > 5 * 1024 * 1024) throw new Error("Maximum CSV size is 5 MB.");
      load(await file.text(), file.name);
    } catch (err) { dataset = null; setStatus(err.message, true); }
  });
  $("sample").addEventListener("click", () => {
    const rows = ["status,biomarker_a,biomarker_b",
      "1,0.90,0.76","1,0.83,0.72","1,0.78,0.53","1,0.67,0.61",
      "1,0.64,0.48","1,0.60,0.69","1,0.52,0.43","1,0.45,0.58",
      "0,0.63,0.62","0,0.57,0.51","0,0.43,0.45","0,0.40,0.57",
      "0,0.34,0.36","0,0.28,0.42","0,0.21,0.32","0,0.12,0.29"];
    try { load(rows.join("\n"), "built-in demonstration data"); }
    catch (err) { setStatus(err.message, true); }
  });
  $("mode").addEventListener("change", () => {
    $("scoreBField").hidden = $("mode").value !== "compare";
  });
  $("analyze").addEventListener("click", () => {
    try {
      if (!dataset) throw new Error("Upload a CSV or load the sample first.");
      const paired = $("mode").value === "compare";
      const data = ROC.numericalData(dataset, $("label").value,
        $("scoreA").value, paired ? $("scoreB").value : null, $("positive").value);
      outcome = ROC.analyze(data, paired);
      showResults(outcome, paired);
      setStatus("Analysis completed locally; no data uploaded.");
    } catch (err) {
      outcome = null;
      $("downloadCsv").disabled = true;
      $("downloadSvg").disabled = true;
      $("summary").replaceChildren();
      $("plot").replaceChildren();
      $("preview").replaceChildren();
      setStatus(err.message, true);
    }
  });
  function fmt(n, digits = 4) { return Number.isFinite(n) ? n.toFixed(digits) : "undefined"; }
  function resultBox(name, result) {
    const section = document.createElement("section");
    section.className = "result";
    const title = document.createElement("h3");
    title.textContent = name;
    section.append(title);
    const dl = document.createElement("dl");
    const fields = [
      ["AUC", fmt(result.auc)],
      ["95% DeLong CI", fmt(result.ci.lo) + "–" + fmt(result.ci.hi)],
      ["SE", fmt(result.ci.se)],
      ["Youden J", fmt(result.best.j)],
      ["Optimal threshold", fmt(result.best.threshold)],
      ["Sensitivity", fmt(result.best.tpr)],
      ["Specificity", fmt(1 - result.best.fpr)]
    ];
    fields.forEach(([k, v]) => {
      const dt = document.createElement("dt"), dd = document.createElement("dd");
      dt.textContent = k; dd.textContent = v;
      dl.append(dt, dd);
    });
    section.append(dl);
    return section;
  }
  function chartElement(outcome, paired) {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("xmlns", ns); svg.setAttribute("viewBox", "0 0 720 440");
    svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "ROC sensitivity versus false positive rate");
    svg.style.width = "100%";
    const append = (tag, attrs, text) => {
      const el = document.createElementNS(ns, tag);
      Object.entries(attrs).forEach(([k,v]) => el.setAttribute(k, String(v)));
      if (text !== undefined) el.textContent = text;
      svg.append(el);
      return el;
    };
    const left = 68, top = 28, size = 350;
    const x = v => left + v * size, y = v => top + (1 - v) * size;
    append("rect", {x: 0, y: 0, width: 720, height: 440, fill: "#101a2c"});
    for (let t = 0; t <= 1.00001; t += 0.25) {
      append("line", {x1: x(t), y1: y(0), x2: x(t), y2: y(1), stroke: "#34415c"});
      append("line", {x1: x(0), y1: y(t), x2: x(1), y2: y(t), stroke: "#34415c"});
      append("text", {x: x(t), y: 400, fill: "#d3ddea", "text-anchor": "middle", "font-size": 13}, t.toFixed(2));
      append("text", {x: 56, y: y(t) + 4, fill: "#d3ddea", "text-anchor": "end", "font-size": 13}, t.toFixed(2));
    }
    append("line", {x1: x(0), y1: y(0), x2: x(1), y2: y(1), stroke: "#a6b2c9", "stroke-dasharray": "6,6"});
    append("text", {x: 242, y: 429, fill: "#f1f5ff", "text-anchor": "middle", "font-size": 15}, "1 - Specificity (FPR)");
    append("text", {x: 18, y: 208, fill: "#f1f5ff", "text-anchor": "middle", "font-size": 15,
      transform: "rotate(-90 18 208)"}, "Sensitivity (TPR)");
    const draw = (result, color, name, offset) => {
      append("polyline", {points: result.points.map(p => x(p.fpr) + "," + y(p.tpr)).join(" "),
        fill: "none", stroke: color, "stroke-width": 3});
      append("circle", {cx: x(result.best.fpr), cy: y(result.best.tpr), r: 6, fill: color, stroke: "white"});
      append("text", {x: 455, y: 88 + offset, fill: color, "font-size": 16}, name + ": AUC " + fmt(result.auc, 3));
    };
    draw(outcome.a, "#68b8ff", $("scoreA").value, 0);
    if (paired) draw(outcome.b, "#ffbd6e", $("scoreB").value, 30);
    append("text", {x: 455, y: 42, fill: "#f1f5ff", "font-size": 17, "font-weight": 700}, "ROC curves");
    return svg;
  }
  function csvCell(value, untrusted = false) {
    let v = String(value);
    if (untrusted && /^[\s\u0000-\u001f]*[=+\-@]/.test(v)) v = "'" + v;
    return '"' + v.replace(/"/g, '""') + '"';
  }
  function showResults(res, paired) {
    $("summary").replaceChildren();
    const counts = document.createElement("p");
    counts.textContent = res.m + " positive / " + res.n + " negative observations (paired by CSV row).";
    $("summary").append(counts, resultBox($("scoreA").value, res.a));
    if (paired) {
      $("summary").append(resultBox($("scoreB").value, res.b));
      const test = document.createElement("p");
      test.className = "comparison";
      test.textContent = "Paired DeLong: ΔAUC = " + fmt(res.comparison.diff) +
        ", z = " + fmt(res.comparison.z) + ", two-sided p = " + fmt(res.comparison.p, 6);
      $("summary").append(test);
    }
    chart = chartElement(res, paired);
    $("plot").replaceChildren(chart);
    exported = [["test", "threshold", "sensitivity", "specificity", "fpr", "youden_j", "is_optimal_cutpoint"]];
    [[res.a, $("scoreA").value], ...(paired ? [[res.b, $("scoreB").value]] : [])]
      .forEach(([r, name]) => r.points.forEach(p => exported.push([
        name, p.threshold === Infinity ? "Infinity" : p.threshold, p.tpr, 1 - p.fpr,
        p.fpr, p.j, p === r.best
      ])));
    const table = document.createElement("table"), head = document.createElement("thead"), hrow = document.createElement("tr");
    exported[0].forEach(h => { const th = document.createElement("th"); th.textContent = h; hrow.append(th); });
    head.append(hrow); table.append(head);
    const body = document.createElement("tbody");
    exported.slice(1, 11).forEach(row => {
      const tr = document.createElement("tr");
      row.forEach(val => { const td = document.createElement("td"); td.textContent = String(val); tr.append(td); });
      body.append(tr);
    });
    table.append(body); $("preview").replaceChildren(table);
    $("previewNote").textContent = "First " + Math.min(10, exported.length - 1) +
      " of " + (exported.length - 1) + " threshold rows; download for all rows.";
    $("downloadCsv").disabled = false; $("downloadSvg").disabled = false;
  }
  function download(content, type, filename) {
    const url = URL.createObjectURL(new Blob([content], {type}));
    const a = document.createElement("a");
    a.href = url; a.download = filename; document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  $("downloadCsv").addEventListener("click", () => {
    const text = exported.map((r, i) => r.map((v, j) => csvCell(v, i > 0 && j === 0)).join(",")).join("\r\n") + "\r\n";
    download(text, "text/csv;charset=utf-8", "roc-thresholds.csv");
  });
  $("downloadSvg").addEventListener("click", () => {
    if (chart) download(new XMLSerializer().serializeToString(chart), "image/svg+xml", "roc-curves.svg");
  });
});
