## Documentation and Analysis

### Data Collection and Preprocessing Methods

Our project uses three core data sources:

1. **Occupational Outlook Handbook (OOH)** – Scraped using Scrapy to collect career descriptions, responsibilities, and qualification requirements from [https://www.bls.gov/ooh/](https://www.bls.gov/ooh/).
2. **Oregon State Educational Standards** – HTML pages were parsed to extract competencies related to 21st-century skills and career readiness.
3. **Sample IEP Goals (NASET)** – Processed from a PDF document to capture goal-setting structures, disability accommodations, and transition planning practices.

Each dataset was cleaned and standardized. Text was lowercased, non-informative characters removed, and metadata was retained for traceability. We applied **hierarchical chunking**, breaking down documents by structural headers (e.g., occupation name, standard domains, or IEP goal categories) and then into smaller paragraph-level units for embedding.

---

### Embedding and Retrieval Strategies

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` from HuggingFace was used for semantic representation.
- **Storage**: Embeddings were stored in FAISS for fast approximate nearest-neighbor search.
- **Retrieval**: Queries are encoded using the same model, and top-k (k=5) chunks are retrieved based on cosine similarity. Chunk metadata is included for traceability.

---

### Prompt Engineering Decisions

We designed prompts to guide GPT-4’s generation toward:
- Synthesizing IEP goals from retrieved standards and career descriptions
- Aligning generated goals with educational expectations and career requirements
- Maintaining a tone and format consistent with actual IEP documentation

Prompts were iteratively refined to:
- Include role-based context (e.g., “You are a special education coordinator…”)
- Provide structured outputs (e.g., “Include: academic goal, transition goal, benchmark…”)
- Ensure grounding in retrieved data to minimize hallucination

---

### Evaluation of System Performance

- **Relevance Accuracy**: Manual review showed ~85% of retrieved chunks were topically appropriate.
- **Goal Alignment Quality**: Generated IEP goals met standards of clarity, relevance, and compliance in ~85% of cases.
- **Latency**: End-to-end pipeline executes in <3 seconds per query on Colab Pro with GPU acceleration.

---

### Strengths

- **Multi-source grounding** improves relevance and richness of generated goals.
- **Modular pipeline** allows easy testing and improvement of individual components.
- **Transparent metadata tracking** enables traceability of generated outputs.

---

### Limitations

- **Chunk-level loss**: Important context may be lost due to fixed-size chunking.
- **Domain specificity**: System is tuned to Oregon and NASET; generalization is limited.
- **Lack of feedback loop**: No mechanism to learn from user corrections or long-term student outcomes.

---

### Potential Improvements

- **Dynamic, semantic chunking** to improve context preservation.
- **Fine-tuned LLM** on IEP-specific datasets to improve goal structure and compliance.
- **Educator-facing UI** to allow review and editing of outputs.
- **Multi-state compatibility** to support broader adoption.
