// Stage 4 B8: static demo data for a `VITE_DEMO=1` build (GitHub Pages --
// see .github/workflows/pages.yml). Same shape and the SAME real values as
// `web/e2e/fixtures.ts` (a real 220-paper "retrieval-augmented generation"
// slice of the live corpus, pulled 2026-11-01) -- kept as a separate file
// rather than importing e2e/fixtures.ts, since that lives outside `src/`
// and pulls in `@playwright/test` as a dependency of the production bundle.
// If you refresh one, refresh the other the same way (re-pull real field
// data, don't hand-edit either into drifting from the live corpus).
import type {
  CorpusManifestOut, FieldOut, FieldCreateOut, FieldStatusOut, GapAnalysisOut, GapDetailOut,
  GapExternalCheckOut, FieldPaperOut, OverviewOut, MapOut, MapSimilarityOut, ShortlistItemOut,
} from "./api";

export const DEMO_FIELD_ID = "default";
export const DEMO_GAP_ID = "0370b0c1baed144d";

export const DEMO_CORPUS_MANIFEST: CorpusManifestOut = {
  papers_count: 1000, query: "retrieval augmented generation question answering large language model knowledge",
  from_year: 2019, to_year: 2026, updated_at: "2026-11-01T17:30:18Z", spend_total_usd: 0.4133,
  budget_usd: 3.0, extraction_complete: true,
};

export const DEMO_FIELDS_LIST: FieldOut[] = [
  { id: DEMO_FIELD_ID, topic: "retrieval-augmented generation", status: "ready", papers_total: 220, last_opened_at: "2026-11-01T17:40:00Z" },
];

export const DEMO_FIELD_CREATE: FieldCreateOut = {
  field_id: DEMO_FIELD_ID, status: "ready", papers_total: 220, total_relevant: 650, extend_command: null,
};

export const DEMO_FIELD_STATUS: FieldStatusOut = {
  id: DEMO_FIELD_ID, topic: "retrieval-augmented generation", status: "ready", step: null,
  papers_total: 220, papers_done: 220, cost_so_far: 0, error: null, created_at: "2026-11-01T17:30:18Z",
};

const A = { cid: 0, labels: ["retrieve-and-read", "Question answering", "large language models", "Generative model"], size: 11 };
const B = { cid: 1, labels: ["full open-domain question answering", "open question answering", "re-ranking method", "top-20 passage retrieval accuracy"], size: 10 };

export const DEMO_GAPS: GapAnalysisOut = {
  node2c: {}, communities: {},
  gaps: [{
    gap_id: DEMO_GAP_ID, a: { ...A, members: ["W3039017601"] }, b: { ...B, members: ["W4412158322"] },
    semantic_similarity: 0.991, similarity_percentile: 83.3, inter_citations: 7, gap_score: 0.124, g_score: 0.551,
    components: { similarity: 0.991, disconnection: 0.125, velocity: 0.305, authority: 0.559 }, meets_threshold: false,
  }],
  total_in_corpus_citations: 79,
};

export const DEMO_GAP_DETAIL: GapDetailOut = {
  gap_id: DEMO_GAP_ID,
  a: {
    cid: A.cid, labels: A.labels, size: A.size,
    top_papers: [
      { paper_id: "W3039017601", title: "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering", year: 2021, venue: "Conference of the European Chapter of the Association for Computational Linguistics", cited_by_count: 1092, local_in_degree: 19, pagerank: 0 },
      { paper_id: "W4389519118", title: "Active Retrieval Augmented Generation", year: 2023, venue: "Conference on Empirical Methods in Natural Language Processing (EMNLP)", cited_by_count: 498, local_in_degree: 3, pagerank: 0 },
      { paper_id: "W4393147129", title: "Benchmarking Large Language Models in Retrieval-Augmented Generation", year: 2024, venue: "Proceedings of the AAAI Conference on Artificial Intelligence", cited_by_count: 455, local_in_degree: 0, pagerank: 0 },
    ],
    methods: ["1-pager", "Generative model", "Transformer", "chain-of-thought", "fusion-in-decoder", "passage retrieval", "retrieval-augmented generation", "retrieve-and-read"],
    datasets: ["2wikimultihopqa", "hotpotqa", "natural questions", "triviaqa"],
  },
  b: {
    cid: B.cid, labels: B.labels, size: B.size,
    top_papers: [
      { paper_id: "W4412158322", title: "Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models", year: 2025, venue: "Computational Linguistics", cited_by_count: 536, local_in_degree: 1, pagerank: 0 },
      { paper_id: "W4388778348", title: "In-Context Retrieval-Augmented Language Models", year: 2023, venue: "Transactions of the Association for Computational Linguistics", cited_by_count: 449, local_in_degree: 9, pagerank: 0 },
      { paper_id: "W4401042773", title: "REPLUG: Retrieval-Augmented Black-Box Language Models", year: 2024, venue: "Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)", cited_by_count: 209, local_in_degree: 4, pagerank: 0 },
    ],
    methods: ["fine-tuning", "large language models", "re-ranking method", "retrieval-augmented generation", "retrieval-augmented language models"],
    datasets: ["ms marco", "nq", "wizard of wikipedia"],
  },
  semantic_similarity: 0.991, similarity_percentile: 83.3, inter_citations: 7, total_in_corpus_citations: 79,
  gap_score: 0.124, g_score: 0.551,
  components: { similarity: 0.991, disconnection: 0.125, velocity: 0.305, authority: 0.559 }, meets_threshold: false,
};

export const DEMO_EXTERNAL_CHECK: GapExternalCheckOut = {
  query: "retrieve-and-read full open-domain question answering", works_count: 12, is_likely_artifact: false,
};

export const DEMO_OVERVIEW: OverviewOut = {
  field_id: DEMO_FIELD_ID, topic: "retrieval-augmented generation", papers: 220, in_corpus_citations: 79,
  topic_communities: 188, candidate_gaps: 6,
  topics: [
    { cid: 0, labels: A.labels, size: 11 },
    { cid: 1, labels: B.labels, size: 10 },
    { cid: 2, labels: ["open-domain QA", "dense passage retrieval", "knowledge-enhanced language model pre-training"], size: 8 },
    { cid: 3, labels: ["GPT-3", "controllable story generation", "mixture-of-experts (moe) ensemble"], size: 6 },
  ],
  rising: [
    { concept: "Key (lock)", recent_count: 30, prior_count: 0, recent_share: 0.156, prior_share: 0, velocity: 10.61, low_confidence: false, year_counts: { "2024": 4, "2025": 9, "2026": 17 } },
    { concept: "Feature (linguistics)", recent_count: 23, prior_count: 0, recent_share: 0.12, prior_share: 0, velocity: 10.23, low_confidence: false, year_counts: { "2024": 4, "2025": 10, "2026": 9 } },
  ],
  key_papers: [
    { paper_id: "W3015883388", title: "Dense Passage Retrieval for Open-Domain Question Answering", year: 2020, venue: "Conference on Empirical Methods in Natural Language Processing (EMNLP)", cited_by_count: 3591, local_in_degree: 25, pagerank: 0.2153 },
    { paper_id: "W3039017601", title: "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering", year: 2021, venue: "Conference of the European Chapter of the Association for Computational Linguistics", cited_by_count: 1092, local_in_degree: 19, pagerank: 0.2065 },
  ],
};

export const DEMO_MAP: MapOut = {
  nodes: [
    { paper_id: "W3039017601", title: "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering", cid: 0, x: -20, y: 0, z: 0 },
    { paper_id: "W4412158322", title: "Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models", cid: 1, x: 20, y: 0, z: 0 },
    { paper_id: "W3015883388", title: "Dense Passage Retrieval for Open-Domain Question Answering", cid: 0, x: -15, y: 5, z: 0 },
    { paper_id: "W4388778348", title: "In-Context Retrieval-Augmented Language Models", cid: 1, x: 15, y: -5, z: 0 },
  ],
  edges: [["W3039017601", "W4412158322"]],
  topics: [
    { cid: 0, label: A.labels.join(" · "), size: 11 },
    { cid: 1, label: B.labels.join(" · "), size: 10 },
  ],
  total_papers: 220, capped: false,
};

export const DEMO_MAP_SIMILARITY: MapSimilarityOut = {
  a_label: A.labels.join(" · "), b_label: B.labels.join(" · "), cosine: 0.991,
  explanation: "99% cosine similarity — near-identical in topic — expect heavy vocabulary and method overlap.",
};

const DEMO_PAPERS: Record<string, FieldPaperOut> = {
  W3039017601: {
    paper_id: "W3039017601", title: "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
    abstract: "Generative models for open domain question answering have proven to be competitive, without resorting to external knowledge. While promising, this approach requires to use models with billions of parameters, which are expensive to train and query. In this paper, we investigate how much these models can benefit from retrieving text passages, potentially containing evidence. We obtain state-of-the-art results on the Natural Questions and TriviaQA open benchmarks. Interestingly, we observe that the performance of this method significantly improves when increasing the number of retrieved passages. This is evidence that generative models are good at aggregating and combining evidence from multiple passages.",
    year: 2021, venue: "Conference of the European Chapter of the Association for Computational Linguistics", cited_by_count: 1092,
    doi: "https://doi.org/10.18653/v1/2021.eacl-main.74",
    entities: [
      { name: "open domain question answering", kind: "task", canonical: "open domain question answering", evidence_span: "Generative models for open domain question answering have proven to be competitive", confidence: 0.95 },
      { name: "Natural Questions", kind: "dataset", canonical: "natural questions", evidence_span: "We obtain state-of-the-art results on the Natural Questions and TriviaQA open benchmarks", confidence: 0.95 },
      { name: "TriviaQA", kind: "dataset", canonical: "triviaqa", evidence_span: "We obtain state-of-the-art results on the Natural Questions and TriviaQA open benchmarks", confidence: 0.95 },
      { name: "Generative models", kind: "method", canonical: "Generative model", evidence_span: "Generative models for open domain question answering have proven to be competitive", confidence: 0.85 },
      { name: "passage retrieval", kind: "method", canonical: "passage retrieval", evidence_span: "Leveraging Passage Retrieval with Generative Models", confidence: 0.8 },
    ],
  },
};

const DEMO_PAPER_FALLBACK = (paperId: string): FieldPaperOut => ({
  paper_id: paperId, title: null, abstract: null, year: null, venue: null, cited_by_count: 0, doi: null, entities: [],
});

export function demoPaper(paperId: string): FieldPaperOut {
  return DEMO_PAPERS[paperId] ?? DEMO_PAPER_FALLBACK(paperId);
}

export function demoDraftedItem(): ShortlistItemOut {
  return {
    id: 1, gap_a_label: "retrieve-and-read", gap_b_label: "full open-domain question answering",
    title: "Bridging retrieve-and-read methods and re-ranking-based open-domain QA",
    claim: "The two clusters share foundational retrieval-augmented generation techniques but rarely cite each other's re-ranking-specific contributions.",
    method_sketch: "Apply Side A's fusion-in-decoder approach to Side B's re-ranking pipeline and measure transfer.",
    datasets: ["natural questions"], supporting_paper_ids: ["W3039017601", "W4412158322"],
    status: "draft", notes: null, papers_to_read: [], created_at: "2026-11-01T18:00:00Z",
  };
}
