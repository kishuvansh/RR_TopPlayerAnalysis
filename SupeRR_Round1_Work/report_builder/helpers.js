const fs = require("fs");
const path = require("path");
const {
  Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType,
  BorderStyle, ImageRun, HeadingLevel, TableLayoutType,
} = require("docx");

const WORK = "/Users/vansh/Downloads/SupeRR Selector Season 2 | Problem Statement/SupeRR_Round1_Work";
const FONT = "Calibri";
const INK = "1F2933";
const MUTED = "616E7C";
const ACCENT = "B45309";
const HEAD_FILL = "1F2933";
const ZEBRA = "F3F4F6";
const CONTENT_W = 9638; // A4 with 2cm margins (11906 - 2*1134)

function readCSV(rel) {
  const txt = fs.readFileSync(path.join(WORK, rel), "utf8").trim();
  const rows = [];
  for (const line of txt.split(/\r?\n/)) {
    const out = []; let cur = ""; let q = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (c === '"') { if (q && line[i + 1] === '"') { cur += '"'; i++; } else q = !q; }
      else if (c === "," && !q) { out.push(cur); cur = ""; }
      else cur += c;
    }
    out.push(cur); rows.push(out);
  }
  const head = rows.shift();
  return rows.map(r => Object.fromEntries(head.map((h, i) => [h, r[i]])));
}

// Inline markup: **bold**, *italic*
function runs(text, base = {}) {
  const parts = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) parts.push(new TextRun({ ...base, text: t.slice(2, -2), bold: true }));
    else if (t.startsWith("`")) parts.push(new TextRun({ ...base, text: t.slice(1, -1), font: "Consolas", size: Math.max(16, (base.size || 21) - 2) }));
    else parts.push(new TextRun({ ...base, text: t.slice(1, -1), italics: true }));
    last = m.index + t.length;
  }
  if (last < text.length) parts.push(new TextRun({ text: text.slice(last), ...base }));
  return parts;
}

const P = (text, opts = {}) => new Paragraph({ children: runs(text, opts.run || {}), spacing: { after: 120, line: 276 }, ...opts.para });
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = t => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const Bullet = (text, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(text), spacing: { after: 60, line: 264 } });
const Num = (text) => new Paragraph({ numbering: { reference: "numbers", level: 0 }, children: runs(text), spacing: { after: 60, line: 264 } });
const Caption = t => new Paragraph({ children: runs(t, { size: 17, color: MUTED, italics: true }), spacing: { before: 60, after: 200 } });
const Formula = t => new Paragraph({ children: [new TextRun({ text: t, font: "Cambria Math", size: 21 })], alignment: AlignmentType.CENTER, spacing: { before: 80, after: 140 } });

function Callout(text, fill = "FFF7ED", bar = ACCENT) {
  return new Paragraph({
    children: runs(text),
    shading: { type: ShadingType.CLEAR, color: "auto", fill },
    border: { left: { style: BorderStyle.SINGLE, size: 24, color: bar, space: 8 } },
    spacing: { before: 120, after: 180, line: 276 },
    indent: { left: 160, right: 160 },
  });
}

function Img(file, widthPx, heightPx) {
  const data = fs.readFileSync(path.join(WORK, file));
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
    children: [new ImageRun({ type: "png", data, transformation: { width: widthPx, height: heightPx } })],
  });
}

const border = { style: BorderStyle.SINGLE, size: 2, color: "D1D5DB" };
const borders = { top: border, bottom: border, left: border, right: border };

// cols: [{h:"Header", w: weight, align:"left|right|center"}], rows: array of arrays of strings
function Tbl(cols, rows, opts = {}) {
  const total = cols.reduce((a, c) => a + c.w, 0);
  let widths = cols.map(c => Math.floor((c.w / total) * CONTENT_W));
  widths[widths.length - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const fs_ = opts.size || 17;
  const cell = (text, i, isHead, fill, bold) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders,
    shading: fill ? { type: ShadingType.CLEAR, color: "auto", fill } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: [new Paragraph({
      alignment: isHead ? AlignmentType.LEFT : (cols[i].align === "right" ? AlignmentType.RIGHT : cols[i].align === "center" ? AlignmentType.CENTER : AlignmentType.LEFT),
      children: runs(String(text ?? ""), { size: fs_, color: isHead ? "FFFFFF" : INK, bold: isHead || bold }),
    })],
  });
  const head = new TableRow({ tableHeader: true, children: cols.map((c, i) => cell(c.h, i, true, HEAD_FILL)) });
  const body = rows.map((r, ri) => new TableRow({
    cantSplit: true,
    children: r.map((v, i) => cell(v, i, false, (opts.highlight && opts.highlight(ri)) ? "FEF3C7" : (ri % 2 ? ZEBRA : undefined), opts.boldFirst && i === 0)),
  }));
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED, rows: [head, ...body] });
}

const Spacer = (after = 120) => new Paragraph({ children: [], spacing: { after } });

module.exports = { readCSV, runs, P, H1, H2, H3, Bullet, Num, Caption, Formula, Callout, Img, Tbl, Spacer, FONT, INK, MUTED, ACCENT, CONTENT_W, WORK };
