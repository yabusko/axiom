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
  return (
    <svg
      viewBox="0 0 640 260"
      width="100%"
      role="img"
      aria-label={`Bridge diagram: Side A, ${aSize} papers, and Side B, ${bSize} papers, enclosed in a dashed boundary marking ${similarityPct.toFixed(1)}% semantic similarity, connected by a thin line marked as the gap, labeled ${interCitations} citations.`}
    >
      <defs>
        <pattern id="gridpat" width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M 20 0 L 0 0 0 20" fill="none" style={{ stroke: "var(--border)", strokeWidth: 1 }} />
        </pattern>
      </defs>
      <rect x="6" y="6" width="628" height="248" fill="url(#gridpat)" />
      <rect
        x="6"
        y="6"
        width="628"
        height="248"
        fill="none"
        style={{ stroke: "var(--text-muted)", strokeWidth: 1.25 }}
        strokeDasharray="5 6"
      />
      <text x="320" y="34" className="bridge-label" style={{ fill: "var(--text-muted)", fontSize: 14 }}>
        {similarityPct.toFixed(1)}% semantic similarity (embedding centroids)
      </text>
      <rect x="50" y="85" width="210" height="110" style={{ fill: "var(--region-wash)", stroke: "var(--region)", strokeWidth: 2 }} />
      <text x="155" y="130" className="bridge-label" style={{ fill: "var(--text)", fontSize: 16, fontWeight: 700 }}>
        Side A
      </text>
      <text x="155" y="152" className="bridge-label" style={{ fill: "var(--text)", fontSize: 14 }}>
        {aSize} papers
      </text>
      <rect x="380" y="85" width="210" height="110" style={{ fill: "var(--region-wash)", stroke: "var(--region)", strokeWidth: 2 }} />
      <text x="485" y="130" className="bridge-label" style={{ fill: "var(--text)", fontSize: 16, fontWeight: 700 }}>
        Side B
      </text>
      <text x="485" y="152" className="bridge-label" style={{ fill: "var(--text)", fontSize: 14 }}>
        {bSize} papers
      </text>
      <line x1="260" y1="140" x2="380" y2="140" style={{ stroke: "var(--mark)", strokeWidth: 2.5 }} />
      <circle cx="320" cy="140" r={4} style={{ fill: "var(--mark)" }} />
      <text x="320" y="118" className="bridge-label" style={{ fill: "var(--mark)", fontSize: 13, fontWeight: 700 }}>
        {interCitations} citations — the gap
      </text>
      <text x="320" y="238" className="bridge-label" style={{ fill: "var(--text-muted)", fontSize: 13 }}>
        out of {totalCitations.toLocaleString()} in-corpus citations total
      </text>
      <title>{aLabel} vs. {bLabel}</title>
    </svg>
  );
}
