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

export function BridgeDiagram({
  aLabel,
  aSize,
  bLabel,
  bSize,
  similarityPct,
  interCitations,
  totalCitations,
}: {
  aLabel: string;
  aSize: number;
  bLabel: string;
  bSize: number;
  similarityPct: number;
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
    `boundary marking ${similarityPct.toFixed(1)}% semantic similarity` +
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
          {similarityPct.toFixed(1)}% semantic similarity (embedding centroids)
        </text>

        <g transform={`translate(${160 - wA / 2}, 85)`}>
          <rect width={wA} height={BOX_H} style={{ fill: "var(--region-wash)", stroke: "var(--region)", strokeWidth: 2 }} />
          <text x={wA / 2} y={BOX_H / 2 - 10} textAnchor="middle" style={{ fill: "var(--text)", fontSize: 15, fontWeight: 700 }}>
            {aLabel}
          </text>
          <text x={wA / 2} y={BOX_H / 2 + 14} textAnchor="middle" style={{ fill: "var(--text)", fontSize: 13 }}>
            {aSize} papers
          </text>
        </g>
        <g transform={`translate(${480 - wB / 2}, 85)`}>
          <rect width={wB} height={BOX_H} style={{ fill: "var(--region-wash)", stroke: "var(--region)", strokeWidth: 2 }} />
          <text x={wB / 2} y={BOX_H / 2 - 10} textAnchor="middle" style={{ fill: "var(--text)", fontSize: 15, fontWeight: 700 }}>
            {bLabel}
          </text>
          <text x={wB / 2} y={BOX_H / 2 + 14} textAnchor="middle" style={{ fill: "var(--text)", fontSize: 13 }}>
            {bSize} papers
          </text>
        </g>

        {hasBridge && (
          <>
            <line x1={160 + wA / 2} y1="140" x2={480 - wB / 2} y2="140" style={{ stroke: "var(--mark)", strokeWidth: stroke }} />
            <circle cx="320" cy="140" r={4} style={{ fill: "var(--mark)" }} />
          </>
        )}
        <text x="320" y={hasBridge ? 118 : 140} textAnchor="middle" style={{ fill: "var(--text)", fontSize: 13, fontWeight: 700 }}>
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
          {similarityPct.toFixed(1)}% semantic similarity
        </text>

        <g transform={`translate(${160 - wA / 2}, 42)`}>
          <rect width={wA} height={BOX_H} style={{ fill: "var(--region-wash)", stroke: "var(--region)", strokeWidth: 2 }} />
          <text x={wA / 2} y={BOX_H / 2 - 10} textAnchor="middle" style={{ fill: "var(--text)", fontSize: 14, fontWeight: 700 }}>
            {aLabel}
          </text>
          <text x={wA / 2} y={BOX_H / 2 + 14} textAnchor="middle" style={{ fill: "var(--text)", fontSize: 13 }}>
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
          <text x={wB / 2} y={BOX_H / 2 - 10} textAnchor="middle" style={{ fill: "var(--text)", fontSize: 14, fontWeight: 700 }}>
            {bLabel}
          </text>
          <text x={wB / 2} y={BOX_H / 2 + 14} textAnchor="middle" style={{ fill: "var(--text)", fontSize: 13 }}>
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
