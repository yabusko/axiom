// Fixture data shaped exactly like api/main.py's real response models, values
// drawn from a real 220-paper "retrieval-augmented generation" slice of the
// live 1000-paper base corpus (pulled 2026-11-01, post the gap-labeling/
// junk-filter/casing fixes) -- real gap_id, real agent-extracted concept
// labels, real evidence papers/abstracts/entities, real citation counts.
// Refreshed from the stale pre-OD34 fixtures (raw OpenAlex tags like
// "Philosophy" on a RAG gap) per B3's review finding.
import type { Page } from "@playwright/test";

const FIELD_ID = "default";

const FIELDS_LIST = [
  { id: FIELD_ID, topic: "retrieval-augmented generation", status: "ready", papers_total: 220, last_opened_at: "2026-11-01T17:40:00Z" },
];

const FIELD_STATUS = {
  id: FIELD_ID, topic: "retrieval-augmented generation", status: "ready", step: null,
  papers_total: 220, papers_done: 220, cost_so_far: 0, error: null, created_at: "2026-11-01T17:30:18Z",
};

export const GAP_ID = "0370b0c1baed144d";

const GAP = {
  gap_id: GAP_ID,
  a: { cid: 0, members: ["W3039017601"], labels: ["retrieve-and-read", "Question answering", "large language models", "Generative model"], size: 11 },
  b: { cid: 1, members: ["W4412158322"], labels: ["full open-domain question answering", "open question answering", "re-ranking method", "top-20 passage retrieval accuracy"], size: 10 },
  semantic_similarity: 0.991, similarity_percentile: 83.3, inter_citations: 7, gap_score: 0.124, g_score: 0.551,
  components: { similarity: 0.991, disconnection: 0.125, velocity: 0.305, authority: 0.559 },
  meets_threshold: false,
};

const GAPS_LIST = { node2c: {}, communities: {}, gaps: [GAP], total_in_corpus_citations: 79 };

const GAP_DETAIL = {
  gap_id: GAP_ID,
  a: {
    cid: GAP.a.cid, labels: GAP.a.labels, size: GAP.a.size,
    top_papers: [
      { paper_id: "W3039017601", title: "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering", year: 2021, venue: "Conference of the European Chapter of the Association for Computational Linguistics", cited_by_count: 1092, local_in_degree: 19, pagerank: 0 },
      { paper_id: "W4389519118", title: "Active Retrieval Augmented Generation", year: 2023, venue: "Conference on Empirical Methods in Natural Language Processing (EMNLP)", cited_by_count: 498, local_in_degree: 3, pagerank: 0 },
      { paper_id: "W4393147129", title: "Benchmarking Large Language Models in Retrieval-Augmented Generation", year: 2024, venue: "Proceedings of the AAAI Conference on Artificial Intelligence", cited_by_count: 455, local_in_degree: 0, pagerank: 0 },
    ],
    methods: ["1-pager", "Generative model", "Transformer", "chain-of-thought", "fusion-in-decoder", "passage retrieval", "retrieval-augmented generation", "retrieve-and-read"],
    datasets: ["2wikimultihopqa", "hotpotqa", "natural questions", "triviaqa"],
  },
  b: {
    cid: GAP.b.cid, labels: GAP.b.labels, size: GAP.b.size,
    top_papers: [
      { paper_id: "W4412158322", title: "Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models", year: 2025, venue: "Computational Linguistics", cited_by_count: 536, local_in_degree: 1, pagerank: 0 },
      { paper_id: "W4388778348", title: "In-Context Retrieval-Augmented Language Models", year: 2023, venue: "Transactions of the Association for Computational Linguistics", cited_by_count: 449, local_in_degree: 9, pagerank: 0 },
      { paper_id: "W4401042773", title: "REPLUG: Retrieval-Augmented Black-Box Language Models", year: 2024, venue: "Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)", cited_by_count: 209, local_in_degree: 4, pagerank: 0 },
    ],
    methods: ["fine-tuning", "large language models", "re-ranking method", "retrieval-augmented generation", "retrieval-augmented language models"],
    datasets: ["ms marco", "nq", "wizard of wikipedia"],
  },
  semantic_similarity: GAP.semantic_similarity, similarity_percentile: GAP.similarity_percentile, inter_citations: GAP.inter_citations,
  total_in_corpus_citations: 79, gap_score: GAP.gap_score, g_score: GAP.g_score,
  components: GAP.components, meets_threshold: GAP.meets_threshold,
};

const OVERVIEW = {
  field_id: FIELD_ID, topic: "retrieval-augmented generation", papers: 220, in_corpus_citations: 79,
  topic_communities: 188, candidate_gaps: 6,
  topics: [
    { cid: 0, labels: GAP.a.labels, size: 11 },
    { cid: 1, labels: GAP.b.labels, size: 10 },
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

const MAP = {
  nodes: [
    { paper_id: "W3039017601", title: "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering", cid: 0, x: -20, y: 0, z: 0 },
    { paper_id: "W4412158322", title: "Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models", cid: 1, x: 20, y: 0, z: 0 },
    { paper_id: "W3015883388", title: "Dense Passage Retrieval for Open-Domain Question Answering", cid: 0, x: -15, y: 5, z: 0 },
    { paper_id: "W4388778348", title: "In-Context Retrieval-Augmented Language Models", cid: 1, x: 15, y: -5, z: 0 },
  ],
  edges: [["W3039017601", "W4412158322"]],
  topics: [
    { cid: 0, label: GAP.a.labels.join(" · "), size: 11 },
    { cid: 1, label: GAP.b.labels.join(" · "), size: 10 },
  ],
  total_papers: 220,
  capped: false,
};

const MAP_SIMILARITY = {
  a_label: GAP.a.labels.join(" · "),
  b_label: GAP.b.labels.join(" · "),
  cosine: 0.991,
  explanation: "99% cosine similarity — near-identical in topic — expect heavy vocabulary and method overlap.",
};

let shortlistItems: Array<Record<string, unknown>> = [];

const DRAFTED_ITEM = {
  id: 1, gap_a_label: "retrieve-and-read", gap_b_label: "full open-domain question answering",
  title: "Bridging retrieve-and-read methods and re-ranking-based open-domain QA",
  claim: "The two clusters share foundational retrieval-augmented generation techniques but rarely cite each other's re-ranking-specific contributions.",
  method_sketch: "Apply Side A's fusion-in-decoder approach to Side B's re-ranking pipeline and measure transfer.",
  datasets: ["natural questions"], supporting_paper_ids: ["W3039017601", "W4412158322"],
  status: "draft", notes: null, papers_to_read: [], created_at: "2026-11-01T18:00:00Z",
};

const CORPUS_MANIFEST = {
  papers_count: 1000, query: "retrieval augmented generation question answering large language model knowledge",
  from_year: 2019, to_year: 2026, updated_at: "2026-11-01T17:30:18Z", spend_total_usd: 0.4133,
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
        json: { field_id: FIELD_ID, status: "ready", papers_total: 220, total_relevant: 650, extend_command: null },
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
    route.fulfill({ json: { query: "retrieve-and-read full open-domain question answering", works_count: 12, is_likely_artifact: false } }),
  );
  await page.route(`http://localhost:8000/fields/${FIELD_ID}/papers/*`, (route) =>
    route.fulfill({
      json: {
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
