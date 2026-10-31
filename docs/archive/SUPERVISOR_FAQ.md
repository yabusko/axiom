# Supervisor Q&A: Axiom P1 Technical Defense

This document anticipates technical and architectural questions regarding the P1 Data Foundation Seam.

---

### I. Architectural & Strategic ("The Why")

**Q: Why did you choose a Hybrid Search approach instead of just using a modern LLM-based vector search?**
*   **A:** Pure semantic (dense) search often struggles with "out-of-vocabulary" technical terms, specific acronyms (e.g., "LoRA", "QLoRA"), and version numbers. By combining it with a sparse (BM25) arm, we ensure we don't lose exact-match precision while still benefiting from semantic conceptual mapping.

**Q: Why is SQLite the center of your architecture rather than a more scalable database like PostgreSQL or a dedicated Graph DB?**
*   **A:** Per decision **OD6**, SQLite serves as our "Shared Data Contract." It is lightweight, file-based (making it easy to share between team members), and more than capable of handling the P1/P2 scale. We use it as the durable source of truth; specialized stores like Qdrant (vectors) and NetworkX (graphs) are populated *from* this SQLite core.

**Q: What is the "P1 Seam" and why is it important?**
*   **A:** It represents the minimum viable end-to-end integration. It proves that the data flow—from raw paper metadata to searchable vector points to a functional UI—is stable. This "unblocks" other tracks (like the Ingestion or Graph teams) because they now have a stable "seam" to plug their work into.

---

### II. Technical Deep-Dive ("The How")

**Q: How does the "Hashing Trick" in your sparse encoder work, and why not use a standard vocabulary?**
*   **A:** We use `blake2b` to hash tokens into a fixed 2^20 index space. This makes the encoder **stateless**. We don't need to save or sync a "dictionary" file between the bootstrap process and the web app. As long as both use the same hashing function, they will always agree on where a word like "Transformer" belongs in the vector.

**Q: Why use Reciprocal Rank Fusion (RRF) instead of just adding the scores together?**
*   **A:** Dense scores (Cosine similarity, usually 0.0 to 1.0) and Sparse scores (BM25, can be any positive number) are on completely different scales. Adding them is like adding meters to gallons. RRF merges them based on their **rank** (1st, 2nd, 3rd...), which is a robust, parameter-free way to combine disparate search engines.

**Q: Why SPECTER2? Why not a general-purpose model like OpenAI's `text-embedding-3-small`?**
*   **A:** SPECTER2 was specifically trained by AllenAI on scientific papers and citation graphs. It understands the specific nuances of academic language (e.g., the relationship between a "Methodology" section and a "Result") better than a model trained on general internet text.

---

### III. Constraints & Trade-offs ("The So What")

**Q: I see you are using "Query Expansion" templates. Why not just use the official SPECTER2 query adapter?**
*   **A:** We encountered a dependency conflict: the official `adapters` library requires a newer version of `transformers` than what is currently pinned for the project. To maintain environment stability and respect the team's versioning pins, we implemented a "Query Expansion" template (`This paper presents {query}...`) which achieves similar semantic alignment without changing the project's dependencies.

**Q: Your current graph implementation uses NetworkX. Will that scale to millions of papers?**
*   **A:** Likely not for millions in real-time, but for the P1/P2 scope (thousands of papers), an in-memory NetworkX graph is significantly faster and simpler than setting up a Neo4j cluster. We have designed the SQLite `citation_edges` table so that we can migrate to a dedicated graph database later without changing our data ingestion logic.

**Q: What happens if Qdrant goes down? Does the app break?**
*   **A:** The app is designed with "Guard Rails." If the connection to Qdrant fails, the UI displays a clear error message instructing the user to check Docker. It fails gracefully rather than crashing the Python process.

---

### IV. Future Directions (P2 and beyond)

**Q: How will the "Gap Pipeline" use this search infrastructure?**
*   **A:** The Gap Pipeline will look for "sparse" regions in the vector space—areas where search queries return low-score results or where there is a conceptual distance between two dense clusters. This infrastructure provides the "map" that allows the P2 track to find those unexplored territories.
