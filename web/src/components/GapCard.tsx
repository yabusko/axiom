import { Link } from "react-router-dom";
import type { GapCandidateOut } from "../api";

function gapHeadline(gap: GapCandidateOut): string {
  const a = gap.a.labels[0] ?? "a cluster";
  const b = gap.b.labels[0] ?? "a cluster";
  return `${gap.a.size}-paper ${a} cluster barely cites the ${gap.b.size}-paper ${b} cluster`;
}

export function GapCard({ gap, index, fieldId }: { gap: GapCandidateOut; index: number; fieldId: string }) {
  return (
    <Link className="gap-card" to={`/fields/${fieldId}/gaps/${index}`}>
      <h3>{gapHeadline(gap)}</h3>
      <div className="gap-stats">
        <span>
          <b>{(gap.semantic_similarity * 100).toFixed(1)}%</b> semantic similarity
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
