// Typed client for the FastAPI /fields/* surface (api/main.py). Mirrors the
// Pydantic response models 1:1 -- see docs/product/ARCHITECTURE_NOTE.md.
const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) {
    return undefined as unknown as T;
  }
  if (res.headers.get("content-type")?.includes("text/markdown")) {
    return (await res.text()) as unknown as T;
  }
  return res.json() as Promise<T>;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

// --- Types (mirror api/main.py's response models) ------------------------------

export interface FieldOut {
  id: string;
  topic: string;
  status: "queued" | "building" | "ready" | "failed";
  papers_total: number | null;
  last_opened_at: string | null;
}

export interface CorpusManifestOut {
  papers_count: number;
  query: string;
  from_year: number;
  to_year: number;
  updated_at: string;
  spend_total_usd: number;
}

export interface FieldCreateOut {
  field_id: string;
  status: "ready" | "thin";
  papers_total: number;
  total_relevant: number;
  extend_command: string | null;
}

export interface FieldStatusOut {
  id: string;
  topic: string;
  status: "queued" | "building" | "ready" | "failed";
  step: "fetching" | "embedding" | "extracting" | null;
  papers_total: number | null;
  papers_done: number;
  cost_so_far: number;
  error: string | null;
  created_at: string;
}

export interface TopicOut {
  cid: number;
  labels: string[];
  size: number;
}

export interface KeywordVelocityOut {
  concept: string;
  recent_count: number;
  prior_count: number;
  recent_share: number;
  prior_share: number;
  velocity: number;
  low_confidence: boolean;
  year_counts: Record<string, number>;
}

export interface PaperRankOut {
  paper_id: string;
  title: string | null;
  year: number | null;
  venue: string | null;
  cited_by_count: number;
  local_in_degree: number;
  pagerank: number;
}

export interface OverviewOut {
  field_id: string;
  topic: string;
  papers: number;
  in_corpus_citations: number;
  topic_communities: number;
  candidate_gaps: number;
  topics: TopicOut[];
  rising: KeywordVelocityOut[];
  key_papers: PaperRankOut[];
}

export interface CommunityOut {
  cid: number;
  members: string[];
  labels: string[];
  size: number;
}

export interface GapCandidateOut {
  a: CommunityOut;
  b: CommunityOut;
  semantic_similarity: number;
  inter_citations: number;
  gap_score: number;
  g_score: number;
  components: Record<string, number>;
  meets_threshold: boolean;
}

export interface GapAnalysisOut {
  node2c: Record<string, number>;
  communities: Record<string, CommunityOut>;
  gaps: GapCandidateOut[];
}

export interface SideDetailOut {
  cid: number;
  labels: string[];
  size: number;
  top_papers: PaperRankOut[];
  methods: string[];
  datasets: string[];
}

export interface GapDetailOut {
  index: number;
  a: SideDetailOut;
  b: SideDetailOut;
  semantic_similarity: number;
  inter_citations: number;
  total_in_corpus_citations: number;
  gap_score: number;
  g_score: number;
  components: Record<string, number>;
  meets_threshold: boolean;
}

export interface EntityOut {
  name: string;
  kind: "method" | "task" | "dataset" | "metric";
  canonical: string;
  evidence_span: string;
  confidence: number;
}

export interface FieldPaperOut {
  paper_id: string;
  title: string | null;
  abstract: string | null;
  year: number | null;
  venue: string | null;
  cited_by_count: number;
  doi: string | null;
  entities: EntityOut[];
}

export type ShortlistStatus = "draft" | "kept" | "discarded";

export interface ShortlistItemOut {
  id: number;
  gap_a_label: string;
  gap_b_label: string;
  title: string;
  claim: string;
  method_sketch: string;
  datasets: string[];
  supporting_paper_ids: string[];
  status: ShortlistStatus;
  notes: string | null;
  papers_to_read: string[];
  created_at: string;
}

export interface MapNodeOut {
  paper_id: string;
  title: string | null;
  cid: number;
  x: number;
  y: number;
  z: number;
}

export interface MapTopicOut {
  cid: number;
  label: string;
  size: number;
}

export interface MapOut {
  nodes: MapNodeOut[];
  edges: [string, string][];
  topics: MapTopicOut[];
  total_papers: number;
  capped: boolean;
}

export type MapEntityKind = "paper" | "topic";

export interface MapSimilarityOut {
  a_label: string;
  b_label: string;
  cosine: number;
  explanation: string;
}

// --- Calls -----------------------------------------------------------------------

export const api = {
  corpusManifest: () => req<CorpusManifestOut>("/corpus"),
  listFields: () => req<FieldOut[]>("/fields"),
  createField: (topic: string, target?: number) =>
    req<FieldCreateOut>("/fields", {
      method: "POST",
      body: JSON.stringify({ topic, ...(target ? { target } : {}) }),
    }),
  fieldStatus: (fieldId: string) => req<FieldStatusOut>(`/fields/${fieldId}`),
  openField: (fieldId: string) => req<void>(`/fields/${fieldId}/open`, { method: "POST" }),
  deleteField: (fieldId: string) => req<void>(`/fields/${fieldId}`, { method: "DELETE" }),
  overview: (fieldId: string) => req<OverviewOut>(`/fields/${fieldId}/overview`),
  gaps: (fieldId: string) => req<GapAnalysisOut>(`/fields/${fieldId}/gaps`),
  gapDetail: (fieldId: string, gapIndex: number) =>
    req<GapDetailOut>(`/fields/${fieldId}/gaps/${gapIndex}`),
  paper: (fieldId: string, paperId: string) =>
    req<FieldPaperOut>(`/fields/${fieldId}/papers/${encodeURIComponent(paperId)}`),
  shortlist: (fieldId: string, status?: ShortlistStatus) =>
    req<ShortlistItemOut[]>(`/fields/${fieldId}/shortlist${status ? `?status=${status}` : ""}`),
  draftDirection: (fieldId: string, gapIndex: number) =>
    req<ShortlistItemOut>(`/fields/${fieldId}/shortlist`, {
      method: "POST",
      body: JSON.stringify({ gap_index: gapIndex }),
    }),
  patchShortlist: (
    fieldId: string,
    itemId: number,
    patch: Partial<Pick<ShortlistItemOut, "status" | "notes" | "papers_to_read">>,
  ) =>
    req<ShortlistItemOut>(`/fields/${fieldId}/shortlist/${itemId}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),
  exportMarkdown: (fieldId: string) => req<string>(`/fields/${fieldId}/shortlist/export.md`),
  map: (fieldId: string) => req<MapOut>(`/fields/${fieldId}/map`),
  mapSimilarity: (
    fieldId: string,
    a: { kind: MapEntityKind; id: string },
    b: { kind: MapEntityKind; id: string },
  ) =>
    req<MapSimilarityOut>(
      `/fields/${fieldId}/map/similarity?a_kind=${a.kind}&a_id=${encodeURIComponent(a.id)}` +
        `&b_kind=${b.kind}&b_id=${encodeURIComponent(b.id)}`,
    ),
};
