import type { PaperRankOut } from "../api";
import { usePaperPanel } from "../hooks/usePaperPanel";

export function PaperRow({ paper, showPagerank = false }: { paper: PaperRankOut; showPagerank?: boolean }) {
  const { openPaper } = usePaperPanel();
  return (
    <div className="paper-row">
      <div className="paper-row-main">
        <button type="button" onClick={() => openPaper(paper.paper_id)}>
          {paper.title ?? paper.paper_id}
        </button>
        <div className="paper-meta">
          {[paper.venue, paper.year, paper.cited_by_count != null ? `${paper.cited_by_count.toLocaleString()} citations` : null]
            .filter(Boolean)
            .join(" · ")}
          {showPagerank && paper.pagerank ? ` · PageRank ${paper.pagerank.toFixed(4)}` : null}
        </div>
      </div>
    </div>
  );
}

export function PaperList({ papers, showPagerank = false }: { papers: PaperRankOut[]; showPagerank?: boolean }) {
  if (papers.length === 0) return null;
  return (
    <div className="paper-list">
      {papers.map((p) => (
        <PaperRow key={p.paper_id} paper={p} showPagerank={showPagerank} />
      ))}
    </div>
  );
}
