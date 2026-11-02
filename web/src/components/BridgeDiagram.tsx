// Stage 4 B2 fixes: real topic labels (not "Side A/B"), box area proportional
// to paper count, connector thickness proportional to cross-citations (no
// line at all when there are none), a genuinely different stacked layout
// below 600px (not the same horizontal layout just shrunk), and no text set
// in --mark (DESIGN_SYSTEM.md: graphical-only -- lines/dots/focus rings,
// never text).
const MIN_BOX_W = 130;
const MAX_BOX_W = 260;
const BOX_H = 110;
const MIN_STROKE = 1.5;
const MAX_STROKE = 9;

// Area = width * BOX_H (constant height), so area is linear in `boxWidth`'s
// output -- the simplest honest way to make "box area proportional to paper
// count" literally true, not just "looks bigger".
function boxWidth(size: number, maxSize: number): number {
  if (maxSize <= 0) return MIN_BOX_W;
  return MIN_BOX_W + (MAX_BOX_W - MIN_BOX_W) * Math.min(size / maxSize, 1);
}

// No cross-gap reference is available here (this component only ever
// renders one gap), so the scale is self-contained: sqrt dampens a long
// citation-count tail (a 40-citation bridge isn't 40x as thick as a
// 1-citation one, just clearly thicker), clamped to a readable range.
function connectorWidth(interCitations: number): number {
  if (interCitations <= 0) return 0;
  return Math.min(MIN_STROKE + Math.sqrt(interCitations) * 1.8, MAX_STROKE);
}

// Review fix: long real topic labels ("dense passage retrieval for
// open-domain question answering") overflowed the box at a fixed font size.
// SVG has no native word-wrap, so this greedily wraps to at most 2 lines
// sized to the box's actual width, truncating line 2 with an ellipsis if it's
// still too long -- an approximation (average glyph width, not measured text)
// but one that degrades to "slightly tight" rather than "overflows the box".
const AVG_CHAR_WIDTH_RATIO = 0.56;

function wrapLabel(label: string, boxW: number, fontSize: number): string[] {
  const maxChars = Math.max(6, Math.floor((boxW - 16) / (fontSize * AVG_CHAR_WIDTH_RATIO)));
  if (label.length <= maxChars) return [label];
  const words = label.split(" ");
  let line1 = "";
  let splitAt = 0;
  for (; splitAt < words.length; splitAt++) {
    const next = line1 ? `${line1} ${words[splitAt]}` : words[splitAt];
    if (next.length > maxChars && line1) break;
    line1 = next;
  }
  let line2 = words.slice(splitAt).join(" ");
  if (!line2) return [line1];
  if (line2.length > maxChars) line2 = `${line2.slice(0, Math.max(1, maxChars - 1))}…`;
  return [line1, line2];
}

function BoxLabel({ lines, cx, cy, fontSize }: { lines: string[]; cx: number; cy: number; fontSize: number }) {
  const lineHeight = fontSize + 3;
  const startY = cy - ((lines.length - 1) * lineHeight) / 2;
  return (
    <>
      {lines.map((line, i) => (
        <text key={i} x={cx} y={startY + i * lineHeight} textAnchor="middle" style={{ fill: "var(--text)", fontSize, fontWeight: 700 }}>
          {line}
        </text>
      ))}
    </>
  );
}

export function BridgeDiagram({
  aLabel,
  aSize,
  bLabel,
  bSize,
  similarityPercentile,
  interCitations,
  totalCitations,
}: {
  aLabel: string;
  aSize: number;
  bLabel: string;
  bSize: number;
  similarityPercentile: number;
  interCitations: number;
  totalCitations: number;
}) {
  const maxSize = Math.max(aSize, bSize);
  const wA = boxWidth(aSize, maxSize);
  const wB = boxWidth(bSize, maxSize);
  const stroke = connectorWidth(interCitations);
  const hasBridge = interCitations > 0;
  const ariaLabel =
    `Bridge diagram: ${aLabel}, ${aSize} papers, and ${bLabel}, ${bSize} papers, enclosed in a dashed ` +
    `boundary marking a gap closer than ${similarityPercentile.toFixed(0)}% of this field's topic pairs` +
    (hasBridge ? `, connected by a line marking the gap, labeled ${interCitations} citations.` : ", with no citations crossing between them.");

  return (
    <div className="bridge-diagram">
      {/* Horizontal layout: boxes side by side, hidden below 600px. */}
      <svg className="bridge-horizontal" viewBox="0 0 640 260" width="100%" role="img" aria-label={ariaLabel}>
        <defs>
          <pattern id="gridpat-h" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" style={{ stroke: "var(--border)", strokeWidth: 1 }} />
          </pattern>
        </defs>
        <rect x="6" y="6" width="628" height="248" fill="url(#gridpat-h)" />
        <rect x="6" y="6" width="628" height="248" fill="none" style={{ stroke: "var(--text-muted)", strokeWidth: 1.25 }} strokeDasharray="5 6" />
        <text x="320" y="30" textAnchor="middle" style={{ fill: "var(--text-muted)", fontSize: 14 }}>
          closer than {similarityPercentile.toFixed(0)}% of this field’s topic pairs (embedding centroids)
        </text>

        <g transform={`translate(${160 - wA / 2}, 85)`}>
          <rect width={wA} height={BOX_H} style={{ fill: "var(--region-wash)", stroke: "var(--region)", strokeWidth: 2 }} />
          <BoxLabel lines={wrapLabel(aLabel, wA, 15)} cx={wA / 2} cy={BOX_H / 2 - 10} fontSize={15} />
          <text x={wA / 2} y={BOX_H / 2 + 26} textAnchor="middle" style={{ fill: "var(--text)", fontSize: 13 }}>
            {aSize} papers
          </text>
        </g>
        <g transform={`translate(${480 - wB / 2}, 85)`}>
          <rect width={wB} height={BOX_H} style={{ fill: "var(--region-wash)", stroke: "var(--region)", strokeWidth: 2 }} />
          <BoxLabel lines={wrapLabel(bLabel, wB, 15)} cx={wB / 2} cy={BOX_H / 2 - 10} fontSize={15} />
          <text x={wB / 2} y={BOX_H / 2 + 26} textAnchor="middle" style={{ fill: "var(--text)", fontSize: 13 }}>
            {bSize} papers
          </text>
        </g>

        {hasBridge && (
          <>
            <line x1={160 + wA / 2} y1="140" x2={480 - wB / 2} y2="140" style={{ stroke: "var(--mark)", strokeWidth: stroke }} />
            <circle cx="320" cy="140" r={4} style={{ fill: "var(--mark)" }} />
          </>
        )}
        {/* Review fix: this label used to sit at y=118/140, between the two
            boxes -- at max box width the gap between boxes is only ~60px,
            far narrower than the label text, so it overlapped box B. Moved
            below the box row (y=195) where the full 640px width is open
            regardless of box size; the connector line/circle above still
            marks the link itself. */}
        <text x="320" y="216" textAnchor="middle" style={{ fill: "var(--text)", fontSize: 13, fontWeight: 700 }}>
          {hasBridge ? `${interCitations} citations — the gap` : "0 citations cross — the gap"}
        </text>
        <text x="320" y="238" textAnchor="middle" style={{ fill: "var(--text-muted)", fontSize: 13 }}>
          out of {totalCitations.toLocaleString()} in-corpus citations total
        </text>
      </svg>

      {/* Vertical layout: boxes stacked, shown only below 600px. */}
      <svg className="bridge-vertical" viewBox="0 0 320 420" width="100%" role="img" aria-label={ariaLabel}>
        <defs>
          <pattern id="gridpat-v" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" style={{ stroke: "var(--border)", strokeWidth: 1 }} />
          </pattern>
        </defs>
        <rect x="6" y="6" width="308" height="408" fill="url(#gridpat-v)" />
        <rect x="6" y="6" width="308" height="408" fill="none" style={{ stroke: "var(--text-muted)", strokeWidth: 1.25 }} strokeDasharray="5 6" />
        <text x="160" y="28" textAnchor="middle" style={{ fill: "var(--text-muted)", fontSize: 13 }}>
          closer than {similarityPercentile.toFixed(0)}% of pairs
        </text>

        <g transform={`translate(${160 - wA / 2}, 42)`}>
          <rect width={wA} height={BOX_H} style={{ fill: "var(--region-wash)", stroke: "var(--region)", strokeWidth: 2 }} />
          <BoxLabel lines={wrapLabel(aLabel, wA, 14)} cx={wA / 2} cy={BOX_H / 2 - 10} fontSize={14} />
          <text x={wA / 2} y={BOX_H / 2 + 26} textAnchor="middle" style={{ fill: "var(--text)", fontSize: 13 }}>
            {aSize} papers
          </text>
        </g>

        {hasBridge && <line x1="160" y1="152" x2="160" y2="196" style={{ stroke: "var(--mark)", strokeWidth: stroke }} />}
        <circle cx="160" cy="174" r={4} style={{ fill: hasBridge ? "var(--mark)" : "var(--border)" }} />
        <text x="160" y="178" textAnchor="middle" style={{ fill: "var(--text)", fontSize: 12, fontWeight: 700 }} dx="40">
          {hasBridge ? `${interCitations} citations` : "0 citations"}
        </text>

        <g transform={`translate(${160 - wB / 2}, 198)`}>
          <rect width={wB} height={BOX_H} style={{ fill: "var(--region-wash)", stroke: "var(--region)", strokeWidth: 2 }} />
          <BoxLabel lines={wrapLabel(bLabel, wB, 14)} cx={wB / 2} cy={BOX_H / 2 - 10} fontSize={14} />
          <text x={wB / 2} y={BOX_H / 2 + 26} textAnchor="middle" style={{ fill: "var(--text)", fontSize: 13 }}>
            {bSize} papers
          </text>
        </g>

        <text x="160" y="340" textAnchor="middle" style={{ fill: "var(--text)", fontSize: 13, fontWeight: 700 }}>
          {hasBridge ? "the gap" : "no citations cross — the gap"}
        </text>
        <text x="160" y="364" textAnchor="middle" style={{ fill: "var(--text-muted)", fontSize: 12 }}>
          out of {totalCitations.toLocaleString()} in-corpus
        </text>
        <text x="160" y="382" textAnchor="middle" style={{ fill: "var(--text-muted)", fontSize: 12 }}>
          citations total
        </text>
      </svg>
    </div>
  );
}
