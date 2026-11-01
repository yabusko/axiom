// Fixture data shaped exactly like api/main.py's real response models, values
// drawn from the real 300-paper "retrieval-augmented generation" field used
// throughout docs/product/ (same gap, same evidence papers) so the e2e test
// exercises realistic content, not placeholder strings.
import type { Page } from "@playwright/test";

const FIELD_ID = "default";

const FIELDS_LIST = [
  { id: FIELD_ID, topic: "retrieval-augmented generation", status: "ready", papers_total: 300, last_opened_at: "2026-10-31T20:50:21Z" },
];

const FIELD_STATUS = {
  id: FIELD_ID, topic: "retrieval-augmented generation", status: "ready", step: null,
  papers_total: 300, papers_done: 300, cost_so_far: 0.11, error: null, created_at: "2026-10-31T19:00:00Z",
};

export const GAP_ID = "fixture-gap-0123456789ab";

const GAP = {
  gap_id: GAP_ID,
  a: { cid: 0, members: ["W1"], labels: ["Natural language processing", "Philosophy", "Machine learning"], size: 43 },
  b: { cid: 1, members: ["W2"], labels: ["Natural language processing", "Question answering"], size: 20 },
  semantic_similarity: 0.988, similarity_percentile: 95, inter_citations: 32, gap_score: 0.03, g_score: 0.618,
  components: { similarity: 0.988, disconnection: 0.9, velocity: 0.1, authority: 0.5 },
  meets_threshold: true,
};

const GAPS_LIST = { node2c: {}, communities: {}, gaps: [GAP], total_in_corpus_citations: 1699 };

const GAP_DETAIL = {
  gap_id: GAP_ID,
  a: {
    cid: GAP.a.cid, labels: GAP.a.labels, size: GAP.a.size,
    top_papers: [{ paper_id: "W_SQUAD", title: "SQuAD: 100,000+ Questions for Machine Comprehension of Text", year: 2016, venue: "EMNLP", cited_by_count: 6208, local_in_degree: 10, pagerank: 0.0829 }],
    methods: ["semantic parser"], datasets: ["freebase"],
  },
  b: {
    cid: GAP.b.cid, labels: GAP.b.labels, size: GAP.b.size,
    top_papers: [{ paper_id: "W_FREEBASE", title: "Semantic Parsing on Freebase from Question-Answer Pairs", year: 2013, venue: "EMNLP", cited_by_count: 1583, local_in_degree: 5, pagerank: 0.0431 }],
    methods: [], datasets: [],
  },
  semantic_similarity: GAP.semantic_similarity, similarity_percentile: GAP.similarity_percentile, inter_citations: GAP.inter_citations,
  total_in_corpus_citations: 1699, gap_score: GAP.gap_score, g_score: GAP.g_score,
  components: GAP.components, meets_threshold: GAP.meets_threshold,
};

const OVERVIEW = {
  field_id: FIELD_ID, topic: "retrieval-augmented generation", papers: 300, in_corpus_citations: 1699,
  topic_communities: 31, candidate_gaps: 15,
  topics: [
    { cid: 0, labels: GAP.a.labels, size: 43 },
    { cid: 1, labels: GAP.b.labels, size: 20 },
  ],
  rising: [
    { concept: "Machine translation", recent_count: 11, prior_count: 0, recent_share: 0.03, prior_share: 0, velocity: 8.68, low_confidence: false, year_counts: { "2020": 3, "2023": 8 } },
  ],
  key_papers: [
    { paper_id: "W_SQUAD", title: "SQuAD: 100,000+ Questions for Machine Comprehension of Text", year: 2016, venue: "EMNLP", cited_by_count: 6208, local_in_degree: 10, pagerank: 0.0829 },
  ],
};

const MAP = {
  nodes: [
    { paper_id: "W1", title: GAP.a.labels[0] + " paper", cid: 0, x: -20, y: 0, z: 0 },
    { paper_id: "W2", title: GAP.b.labels[0] + " paper", cid: 1, x: 20, y: 0, z: 0 },
    { paper_id: "W_SQUAD", title: "SQuAD: 100,000+ Questions for Machine Comprehension of Text", cid: 0, x: -15, y: 5, z: 0 },
    { paper_id: "W_FREEBASE", title: "Semantic Parsing on Freebase from Question-Answer Pairs", cid: 1, x: 15, y: -5, z: 0 },
  ],
  edges: [["W1", "W2"]],
  topics: [
    { cid: 0, label: GAP.a.labels.join(" · "), size: 2 },
    { cid: 1, label: GAP.b.labels.join(" · "), size: 2 },
  ],
  total_papers: 300,
  capped: false,
};

const MAP_SIMILARITY = {
  a_label: GAP.a.labels.join(" · "),
  b_label: GAP.b.labels.join(" · "),
  cosine: 0.988,
  explanation: "99% cosine similarity — near-identical in topic — expect heavy vocabulary and method overlap.",
};

let shortlistItems: Array<Record<string, unknown>> = [];

const DRAFTED_ITEM = {
  id: 1, gap_a_label: "NLP/ML methods", gap_b_label: "Question answering",
  title: "Bridging NLP methods and QA-focused work",
  claim: "The two clusters share foundational NLP techniques but rarely cite each other's QA-specific contributions.",
  method_sketch: "Apply Side A's methods to Side B's QA benchmarks and measure transfer.",
  datasets: ["freebase"], supporting_paper_ids: ["W_SQUAD", "W_FREEBASE"],
  status: "draft", notes: null, papers_to_read: [], created_at: "2026-10-31T21:00:00Z",
};

const CORPUS_MANIFEST = {
  papers_count: 1000, query: "retrieval augmented generation question answering large language model knowledge",
  from_year: 2019, to_year: 2026, updated_at: "2026-11-01T12:00:00Z", spend_total_usd: 0.42,
  budget_usd: 3.0, extraction_complete: true,
};

export async function mockApi(page: Page) {
  shortlistItems = [];

  await page.route("http://localhost:8000/corpus", (route) => route.fulfill({ json: CORPUS_MANIFEST }));
  await page.route("http://localhost:8000/fields", (route) => {
    if (route.request().method() === "GET") {
      return route.fulfill({ json: FIELDS_LIST });
    }
    if (route.request().method() === "POST") {
      return route.fulfill({
        json: { field_id: FIELD_ID, status: "ready", papers_total: 300, total_relevant: 300, extend_command: null },
      });
    }
    return route.continue();
  });
  await page.route(`http://localhost:8000/fields/${FIELD_ID}`, (route) => route.fulfill({ json: FIELD_STATUS }));
  await page.route(`http://localhost:8000/fields/${FIELD_ID}/open`, (route) => route.fulfill({ status: 204 }));
  await page.route(`http://localhost:8000/fields/${FIELD_ID}/overview`, (route) => route.fulfill({ json: OVERVIEW }));
  await page.route(`http://localhost:8000/fields/${FIELD_ID}/gaps`, (route) => {
    if (route.request().method() === "GET" && !route.request().url().includes("/gaps/")) {
      return route.fulfill({ json: GAPS_LIST });
    }
    return route.continue();
  });
  await page.route(`http://localhost:8000/fields/${FIELD_ID}/gaps/${GAP_ID}`, (route) => route.fulfill({ json: GAP_DETAIL }));
  await page.route(`http://localhost:8000/fields/${FIELD_ID}/gaps/${GAP_ID}/external-check`, (route) =>
    route.fulfill({ json: { query: "Natural language processing Question answering", works_count: 12, is_likely_artifact: false } }),
  );
  await page.route(`http://localhost:8000/fields/${FIELD_ID}/papers/*`, (route) =>
    route.fulfill({
      json: {
        paper_id: "W_FREEBASE", title: "Semantic Parsing on Freebase from Question-Answer Pairs",
        abstract: "In this paper, we train a semantic parser that scales up to Freebase.",
        year: 2013, venue: "EMNLP", cited_by_count: 1583, doi: "https://doi.org/10.18653/v1/d13-1160",
        entities: [{ name: "semantic parser", kind: "method", canonical: "semantic parser", evidence_span: "we train a semantic parser that scales up to Freebase.", confidence: 0.7 }],
      },
    }),
  );
  await page.route(`http://localhost:8000/fields/${FIELD_ID}/shortlist`, (route) => {
    const method = route.request().method();
    if (method === "POST") {
      shortlistItems = [{ ...DRAFTED_ITEM }];
      return route.fulfill({ json: DRAFTED_ITEM });
    }
    return route.fulfill({ json: shortlistItems });
  });
  await page.route(`http://localhost:8000/fields/${FIELD_ID}/shortlist/1`, (route) => {
    const body = route.request().postDataJSON() as Record<string, unknown>;
    shortlistItems = shortlistItems.map((i) => (i.id === 1 ? { ...i, ...body } : i));
    return route.fulfill({ json: shortlistItems[0] });
  });
  await page.route(`http://localhost:8000/fields/${FIELD_ID}/map`, (route) => route.fulfill({ json: MAP }));
  await page.route(`http://localhost:8000/fields/${FIELD_ID}/map/similarity**`, (route) =>
    route.fulfill({ json: MAP_SIMILARITY }),
  );
  await page.route(`http://localhost:8000/fields/${FIELD_ID}/shortlist/export.md`, (route) =>
    route.fulfill({
      contentType: "text/markdown",
      body: `# retrieval-augmented generation — shortlist\n\n## ${DRAFTED_ITEM.title}\n\n${DRAFTED_ITEM.claim}\n`,
    }),
  );
}
