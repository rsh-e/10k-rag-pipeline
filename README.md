## The Project: "RAG System with Retrieval Evaluation Harness"

A domain-specific Q&A system (pick something you know — your old coursework, a niche topic, company docs, whatever) where the differentiator is that you **benchmark your own retrieval choices** instead of just wiring up a demo. This is what separates a script-kiddie project from something that reads "this person understands tradeoffs."

**Stack (all $0):**
- ChromaDB (you already know it) — vector store
- `sentence-transformers` (local, free) — embeddings
- **Groq API** (free tier, generous limits, very fast) — LLM inference, no local GPU needed
- Streamlit — UI, deployed free on **Streamlit Community Cloud**
- RAGAS or a hand-rolled eval script — retrieval/answer quality metrics

## Day 1 — Data + core pipeline

1. Pick a corpus (20–50 documents is plenty — PDFs, markdown, whatever). Something specific reads better than generic ("SEC 10-K filings for 5 companies", "a niche open-source project's docs", "your own study notes from the AWS cert" — meta and relatable).
2. Build ingestion: chunk with **two different strategies** (fixed-size vs semantic/recursive chunking) — you'll compare these later, this is your differentiator.
3. Embed with `sentence-transformers` (e.g. `all-MiniLM-L6-v2`), store in Chroma with metadata (source, chunk strategy, chunk id).
4. Basic retrieval + Groq LLM call for generation. Get a working end-to-end query → answer flow.

## Day 2 — The differentiator: evaluation

This is the part that makes it "engineer" not "tutorial follower."

1. Write ~15-20 test questions with known correct answers/sources from your corpus.
2. Build a small eval script that measures:
   - **Retrieval precision/recall** — did the right chunk get retrieved in top-k?
   - **Answer faithfulness** — does the generated answer actually come from retrieved context (use an LLM-as-judge call via Groq, cheap and fast)?
3. Run this eval against your two chunking strategies + maybe two `k` values. Produce a small results table/chart (matplotlib, save as PNG).
4. Optionally add a simple reranker (cross-encoder from `sentence-transformers`, still free/local) and show it improves your metrics.

This gives you a concrete, quantified claim for your writeup: *"Semantic chunking improved retrieval precision by X% over fixed-size chunking on my eval set."* That sentence alone signals more competence than 90% of portfolio RAG projects.

## Day 3 — Polish, deploy, write up

1. Wrap it in a clean Streamlit app: chat interface + a toggle to show retrieved chunks/sources (transparency is impressive to reviewers) + maybe a small "eval results" tab showing your benchmark chart.
2. Deploy free on Streamlit Community Cloud (connect your GitHub repo, one click).
3. Clean up the GitHub repo: solid README with **architecture diagram** (even a simple one), setup instructions, and your eval results table front and center.
4. Write the LinkedIn post — lead with the result/insight, not the tech stack:
   - Bad: "Built a RAG chatbot using ChromaDB and Streamlit!"
   - Good: "I benchmarked 2 chunking strategies + reranking on a RAG pipeline and found semantic chunking gave a 23% precision lift — here's the eval harness and what I learned."
