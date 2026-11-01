import { useEffect, useRef, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api, type MapEntityKind, type MapNodeOut, type MapOut, type MapSimilarityOut } from "../api";
import { usePaperPanel } from "../hooks/usePaperPanel";
import { EmptyState, ErrorState, LoadingState } from "../components/States";

// Categorical palette (dataviz skill reference instance, validated against
// CVD/normal-vision floors in both themes) -- fixed hue order, never cycled.
// A field's 9th+ topic folds into OTHER_COLOR rather than generating a hue.
const PALETTE_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"];
const PALETTE_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
const OTHER_COLOR = "#8A98A0";
const HIGHLIGHT_DIM = "#C7D0D3";

function useIsDarkTheme(): boolean {
  const [isDark, setIsDark] = useState(() => window.matchMedia("(prefers-color-scheme: dark)").matches);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setIsDark(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return isDark;
}

function buildColorScale(topics: MapOut["topics"], dark: boolean): (cid: number) => string {
  const palette = dark ? PALETTE_DARK : PALETTE_LIGHT;
  const order = [...topics].sort((a, b) => b.size - a.size).slice(0, palette.length);
  const byCid = new Map(order.map((t, i) => [t.cid, palette[i]]));
  return (cid: number) => byCid.get(cid) ?? OTHER_COLOR;
}

function GraphView({
  nodes,
  edges,
  mode,
  colorForCid,
  highlightCids,
  onNodeClick,
}: {
  nodes: MapNodeOut[];
  edges: [string, string][];
  mode: "citation" | "embedding";
  colorForCid: (cid: number) => string;
  highlightCids: number[] | null;
  onNodeClick: (paperId: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let destroyed = false;
    // 3d-force-graph's Kapsule-generated instance type doesn't play well with
    // TS's ReturnType over its overloaded factory signature -- `any` here, not
    // on anything public.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let graph: any = null;

    import("3d-force-graph").then((mod) => {
      if (destroyed || !container) return;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const ForceGraph3D = mod.default as any;
      const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const graphData = {
        nodes: nodes.map((n) => ({
          id: n.paper_id,
          title: n.title ?? n.paper_id,
          cid: n.cid,
          ...(mode === "embedding" ? { fx: n.x, fy: n.y, fz: n.z } : {}),
        })),
        links: edges.map(([source, target]) => ({ source, target })),
      };
      graph = ForceGraph3D()(container)
        .graphData(graphData)
        .width(container.clientWidth)
        .height(520)
        .backgroundColor("rgba(0,0,0,0)")
        .nodeRelSize(4)
        .nodeLabel((n: { title: string }) => n.title)
        .nodeColor((n: { cid: number }) =>
          highlightCids && highlightCids.length > 0
            ? highlightCids.includes(n.cid)
              ? colorForCid(n.cid)
              : HIGHLIGHT_DIM
            : colorForCid(n.cid),
        )
        .linkOpacity(0.25)
        .linkWidth(0.4)
        .onNodeClick((n: { id: string }) => onNodeClick(n.id))
        .cooldownTicks(mode === "embedding" || reducedMotion ? 0 : 150)
        .warmupTicks(mode === "embedding" ? 1 : 0);
      if (reducedMotion) {
        const controls = graph.controls() as { autoRotate?: boolean };
        controls.autoRotate = false;
      }
    });

    const onResize = () => {
      if (graph && container) graph.width(container.clientWidth);
    };
    window.addEventListener("resize", onResize);

    return () => {
      destroyed = true;
      window.removeEventListener("resize", onResize);
      if (container) container.innerHTML = "";
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, edges, mode, highlightCids]);

  return (
    <div
      ref={containerRef}
      className="map-canvas"
      role="img"
      aria-label={
        mode === "citation"
          ? "3D citation graph: papers as nodes colored by topic, lines are in-corpus citations"
          : "3D embedding map: papers placed by similarity, closer points are more similar"
      }
    />
  );
}

export default function MapScreen() {
  const { fieldId } = useParams<{ fieldId: string }>();
  const [searchParams] = useSearchParams();
  const { openPaper } = usePaperPanel();
  const isDark = useIsDarkTheme();
  const [view, setView] = useState<"citation" | "embedding">("citation");

  const [cmpAKind, setCmpAKind] = useState<MapEntityKind>("topic");
  const [cmpBKind, setCmpBKind] = useState<MapEntityKind>("topic");
  const [cmpAId, setCmpAId] = useState("");
  const [cmpBId, setCmpBId] = useState("");
  const [similarity, setSimilarity] = useState<MapSimilarityOut | null>(null);
  const [similarityError, setSimilarityError] = useState<string | null>(null);
  const [comparing, setComparing] = useState(false);

  const highlightA = searchParams.get("a");
  const highlightB = searchParams.get("b");
  const highlightCids = highlightA && highlightB ? [Number(highlightA), Number(highlightB)] : null;

  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ["map", fieldId],
    queryFn: () => api.map(fieldId!),
    enabled: !!fieldId,
  });

  useEffect(() => {
    if (highlightA && highlightB) {
      setCmpAKind("topic");
      setCmpBKind("topic");
      setCmpAId(highlightA);
      setCmpBId(highlightB);
    }
    // Only react to the URL params changing, not every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [highlightA, highlightB]);

  if (isLoading) return <LoadingState label="Loading the map…" />;
  if (isError || !data) {
    return (
      <ErrorState title="Couldn't load the map" onRetry={() => refetch()}>
        This field's map didn't load. Try again.
      </ErrorState>
    );
  }
  if (data.nodes.length === 0) {
    return <EmptyState title="Nothing to map yet">This field has no embedded papers yet.</EmptyState>;
  }

  const colorForCid = buildColorScale(data.topics, isDark);
  const paperOptions = data.nodes.filter((n) => n.title);
  const topicLabel = (cid: number) => data.topics.find((t) => t.cid === cid)?.label ?? `Topic ${cid}`;

  async function compare() {
    if (!fieldId || !cmpAId || !cmpBId) return;
    setComparing(true);
    setSimilarityError(null);
    try {
      const result = await api.mapSimilarity(fieldId, { kind: cmpAKind, id: cmpAId }, { kind: cmpBKind, id: cmpBId });
      setSimilarity(result);
    } catch (err) {
      setSimilarityError((err as Error).message);
      setSimilarity(null);
    } finally {
      setComparing(false);
    }
  }

  return (
    <>
      <div className="page-head">
        <h1>Map</h1>
        <p className="lede">
          A spatial view of this field's papers — the same citation graph and topic communities used everywhere else
          in Axiom, laid out instead of listed. Not part of the primary flow; a way to look at the structure
          directly.
        </p>
      </div>

      {data.capped && (
        <p className="muted">
          Showing the {data.nodes.length} most influential of {data.total_papers} papers, capped so the 3D view
          stays smooth.
        </p>
      )}
      {highlightCids && (
        <p className="muted">
          Highlighting the gap between <strong style={{ color: "var(--text)" }}>{topicLabel(highlightCids[0])}</strong>{" "}
          and <strong style={{ color: "var(--text)" }}>{topicLabel(highlightCids[1])}</strong> in both views below.
        </p>
      )}

      <div className="map-view-toggle" role="group" aria-label="Map view">
        <button
          type="button"
          className={view === "citation" ? "btn btn-primary" : "btn-quiet"}
          aria-pressed={view === "citation"}
          onClick={() => setView("citation")}
        >
          Citation graph
        </button>
        <button
          type="button"
          className={view === "embedding" ? "btn btn-primary" : "btn-quiet"}
          aria-pressed={view === "embedding"}
          onClick={() => setView("embedding")}
        >
          Embedding map
        </button>
      </div>

      <div className="map-layout">
        <div className="map-canvas-wrap">
          <GraphView
            nodes={data.nodes}
            edges={data.edges}
            mode={view}
            colorForCid={colorForCid}
            highlightCids={highlightCids}
            onNodeClick={openPaper}
          />
          <p className="muted" style={{ fontSize: "var(--small)" }}>
            {view === "citation"
              ? "Nodes are papers, colored by topic community; lines are in-corpus citations. Click a node to open it. Drag to rotate, scroll to zoom."
              : "Each point is a paper, placed by a PCA projection of its embedding — papers with similar text sit closer together. This 3D view isn't keyboard-navigable; use “Compare similarity” below for an exact, accessible number."}
          </p>
        </div>

        <aside className="map-sidebar">
          <h3>Topics</h3>
          <ul className="map-legend">
            {[...data.topics]
              .sort((a, b) => b.size - a.size)
              .slice(0, 8)
              .map((t) => (
                <li key={t.cid}>
                  <span className="swatch" style={{ background: colorForCid(t.cid) }} />
                  {t.label} <span className="muted">({t.size})</span>
                </li>
              ))}
            {data.topics.length > 8 && (
              <li className="muted">
                <span className="swatch" style={{ background: OTHER_COLOR }} />
                {data.topics.length - 8} more topics
              </li>
            )}
          </ul>

          <h3>Compare similarity</h3>
          <p className="muted" style={{ fontSize: "var(--small)" }}>
            Pick two papers or two topics to see how close they are in embedding space, with a plain-language read.
          </p>

          <div className="map-compare-row">
            <label>
              <span className="visually-hidden">First item type</span>
              <select
                value={cmpAKind}
                onChange={(e) => {
                  setCmpAKind(e.target.value as MapEntityKind);
                  setCmpAId("");
                }}
              >
                <option value="topic">Topic</option>
                <option value="paper">Paper</option>
              </select>
            </label>
            <label style={{ flex: 1 }}>
              <span className="visually-hidden">First item</span>
              <select value={cmpAId} onChange={(e) => setCmpAId(e.target.value)} style={{ width: "100%" }}>
                <option value="" disabled>
                  Select…
                </option>
                {cmpAKind === "topic"
                  ? data.topics.map((t) => (
                      <option key={t.cid} value={t.cid}>
                        {t.label}
                      </option>
                    ))
                  : paperOptions.map((p) => (
                      <option key={p.paper_id} value={p.paper_id}>
                        {p.title}
                      </option>
                    ))}
              </select>
            </label>
          </div>

          <div className="map-compare-row">
            <label>
              <span className="visually-hidden">Second item type</span>
              <select
                value={cmpBKind}
                onChange={(e) => {
                  setCmpBKind(e.target.value as MapEntityKind);
                  setCmpBId("");
                }}
              >
                <option value="topic">Topic</option>
                <option value="paper">Paper</option>
              </select>
            </label>
            <label style={{ flex: 1 }}>
              <span className="visually-hidden">Second item</span>
              <select value={cmpBId} onChange={(e) => setCmpBId(e.target.value)} style={{ width: "100%" }}>
                <option value="" disabled>
                  Select…
                </option>
                {cmpBKind === "topic"
                  ? data.topics.map((t) => (
                      <option key={t.cid} value={t.cid}>
                        {t.label}
                      </option>
                    ))
                  : paperOptions.map((p) => (
                      <option key={p.paper_id} value={p.paper_id}>
                        {p.title}
                      </option>
                    ))}
              </select>
            </label>
          </div>

          <button type="button" className="btn btn-primary" onClick={compare} disabled={!cmpAId || !cmpBId || comparing}>
            {comparing ? "Comparing…" : "Compare"}
          </button>

          {similarityError && <p style={{ color: "var(--action)" }}>Couldn't compare: {similarityError}</p>}

          {similarity && (
            <div
              className="card"
              style={{ border: "1px solid var(--border)", background: "var(--surface)", padding: "var(--space-4)", marginTop: "var(--space-4)" }}
            >
              <p style={{ margin: 0, fontWeight: 700 }}>
                {similarity.a_label} ↔ {similarity.b_label}
              </p>
              <p className="muted" style={{ margin: "4px 0 0" }}>
                {similarity.explanation}
              </p>
            </div>
          )}
        </aside>
      </div>
    </>
  );
}
