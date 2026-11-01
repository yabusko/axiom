import type { FieldStatusOut } from "../api";

const STEP_ORDER = ["fetching", "embedding", "extracting"] as const;
const STEP_LABEL: Record<(typeof STEP_ORDER)[number], string> = {
  fetching: "Finding papers",
  embedding: "Embedding papers",
  extracting: "Extracting methods & datasets",
};

export function BuildProgress({
  status,
  estCostUsd,
  estSeconds,
}: {
  status: FieldStatusOut;
  estCostUsd?: number;
  estSeconds?: number;
}) {
  const currentIndex = status.step ? STEP_ORDER.indexOf(status.step) : STEP_ORDER.length;
  return (
    <ol className="steps">
      {STEP_ORDER.map((step, i) => {
        const state = i < currentIndex ? "done" : i === currentIndex ? "active" : "pending";
        let detail: string;
        if (step === "fetching") {
          // No incremental signal during the fetch itself (the backend only
          // knows the real count once ingest finishes) -- showing a static
          // "0 of N" the whole time it's active looks stalled, not honest
          // about being indeterminate. Only show the real count once done.
          if (state === "pending") detail = "not started";
          else if (state === "active") detail = "fetching from OpenAlex…";
          else detail = `${status.papers_done} of ${status.papers_total ?? "?"} fetched`;
        } else if (step === "embedding") {
          detail = state === "pending" ? "not started" : state === "active" ? "in progress" : "done";
        } else {
          if (state === "pending" && estCostUsd != null && estSeconds != null) {
            detail = `est. ~$${estCostUsd.toFixed(2)} · ~${Math.round(estSeconds / 60) || 1} min, starts after step 2`;
          } else if (state === "active") {
            detail = `${status.papers_done} of ${status.papers_total ?? "?"} papers · $${status.cost_so_far.toFixed(2)} so far`;
          } else {
            detail = "done";
          }
        }
        return (
          <li className={`step ${state}`} key={step}>
            <span className="dot" />
            <div>
              <div className="label">
                {i + 1}. {STEP_LABEL[step]}
              </div>
              <div className="detail">{detail}</div>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
