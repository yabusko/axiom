import { Link } from "react-router-dom";
import type { GapCandidateOut } from "../api";

// Review fix: "barely cites" is only accurate when the cross-citation SHARE
// is actually low -- real field data spans ~1% to ~16%+ share across a
// single field's own top gaps, and 16% sharing a direct citation link isn't
// "barely" anything. OD41 (docs/DECISIONS.md): share = inter_citations /
// total_in_corpus_citations, < 5% reads as "barely cites" (the cited pairs
// in a real field this stage measured: 1/86=1.2%, 2/86=2.3%, 3/86=3.5% all
// read as genuinely sparse; 7/86=8.1%, 8/86=9.3%, 14/86=16.3% do not) --
// everything at or above 5% gets neutral, still-honest phrasing instead.
const LOW_SHARE_THRESHOLD = 0.05;

function gapHeadline(gap: GapCandidateOut, totalInCorpusCitations: number): string {
  const a = gap.a.labels[0] ?? "a cluster";
  const b = gap.b.labels[0] ?? "a cluster";
  const share = gap.inter_citations / Math.max(totalInCorpusCitations, 1);
  const verb = share < LOW_SHARE_THRESHOLD ? "barely cites" : `shares ${gap.inter_citations} citations with`;
  return `${gap.a.size}-paper ${a} cluster ${verb} the ${gap.b.size}-paper ${b} cluster`;
}

export function GapCard({
  gap, fieldId, totalInCorpusCitations,
}: {
  gap: GapCandidateOut; fieldId: string; totalInCorpusCitations: number;
}) {
  return (
    <Link className="gap-card" to={`/fields/${fieldId}/gaps/${encodeURIComponent(gap.gap_id)}`}>
      <h3>{gapHeadline(gap, totalInCorpusCitations)}</h3>
      <div className="gap-stats">
        <span>
          closer than <b>{gap.similarity_percentile.toFixed(0)}%</b> of this field's topic pairs
        </span>
        <span>
          <b>{gap.inter_citations}</b> citations cross
        </span>
        <span>
          <b>{gap.a.size}</b> vs <b>{gap.b.size}</b> papers
        </span>
        <span>
          G-score <b>{gap.g_score.toFixed(3)}</b> (uncalibrated)
        </span>
      </div>
    </Link>
  );
}
