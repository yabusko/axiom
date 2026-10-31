import type { KeywordVelocityOut } from "../api";

/** Real per-year counts (keyword_velocity.year_counts) plotted as-is -- no
 * fabricated intermediate points, per DESIGN_SYSTEM.md's sparkline spec. */
function Sparkline({ yearCounts }: { yearCounts: Record<string, number> }) {
  const entries = Object.entries(yearCounts)
    .map(([y, n]) => [Number(y), n] as const)
    .sort((a, b) => a[0] - b[0]);
  if (entries.length < 2) return null;
  const w = 110;
  const h = 30;
  const pad = 4;
  const years = entries.map((e) => e[0]);
  const counts = entries.map((e) => e[1]);
  const minY = Math.min(...years);
  const maxY = Math.max(...years);
  const maxC = Math.max(...counts, 1);
  const x = (year: number) => (maxY === minY ? pad : pad + ((year - minY) / (maxY - minY)) * (w - 2 * pad));
  const y = (count: number) => h - pad - (count / maxC) * (h - 2 * pad);
  const points = entries.map(([yr, c]) => `${x(yr)},${y(c)}`).join(" ");
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
      <polyline points={points} fill="none" style={{ stroke: "var(--text-muted)", strokeWidth: 1.5 }} />
      <circle cx={x(entries[0][0])} cy={y(entries[0][1])} r={2.5} style={{ fill: "var(--text-muted)" }} />
      <circle
        cx={x(entries[entries.length - 1][0])}
        cy={y(entries[entries.length - 1][1])}
        r={2.5}
        style={{ fill: "var(--text-muted)" }}
      />
    </svg>
  );
}

export function TrendRow({ k }: { k: KeywordVelocityOut }) {
  return (
    <div className="trend-row">
      <div>
        <div className="trend-name">{k.concept}</div>
        <Sparkline yearCounts={k.year_counts} />
      </div>
      <div className="trend-ratio">
        {k.recent_count} recent vs. {k.prior_count} prior mentions
        {k.low_confidence ? " · low volume" : ""}
      </div>
    </div>
  );
}
