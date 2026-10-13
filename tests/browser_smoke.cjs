"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const {parseCSV, numericalData, analyze} = require("../web/roc.js");

const csv = parseCSV('label,a,b\r\n1,0.9,0.9\r\n1,0.8,0.8\r\n0,0.2,0.2\r\n0,0.1,0.1\r\n');
const single = analyze(numericalData(csv, "label", "a", null, "1"));
assert.equal(single.a.auc, 1);
assert.equal(single.a.best.j, 1);
const same = analyze(numericalData(csv, "label", "a", "b", "1"), true);
assert.equal(same.comparison.diff, 0);
assert.equal(same.comparison.p, 1);

const tied = parseCSV('y,score\n1,0.5\n1,0.5\n0,0.5\n0,0.5\n');
assert.equal(analyze(numericalData(tied, "y", "score", null, "1")).a.auc, 0.5);
const quoted = parseCSV('status,"score,A",b\r\n"1",0.2,0.3\r\n"0",0.1,0.2\r\n');
assert.deepEqual(quoted.headers, ["status", "score,A", "b"]);

assert.throws(() => parseCSV("a,b\n1,2,3\n"), /same number/);
assert.throws(() => parseCSV('a,b\n"bad,1\n'), /Unclosed/);
assert.throws(() => numericalData(parseCSV("y,x\n1,0.2\n0,NaN\n"), "y", "x", null, "1"), /nonfinite/);
assert.throws(() => numericalData(parseCSV("y,x\n1,0.2\n1,0.4\n0,0.1\n"), "y", "x", null, "1"), /at least two/);

const perfectDifferent = parseCSV("y,a,b\n1,1,0\n1,1,0\n0,0,1\n0,0,1\n");
assert.throws(() => analyze(numericalData(perfectDifferent, "y", "a", "b", "1"), true), /undefined/);

const html = fs.readFileSync(require("node:path").join(__dirname, "../web/index.html"), "utf8");
assert(html.includes('id="file"') && html.includes('id="plot"'));
assert(!html.includes("DOSSIER-SIM") && !html.includes("fake audit"));
console.log("Browser-side numerical, CSV, and integration smoke checks passed.");
