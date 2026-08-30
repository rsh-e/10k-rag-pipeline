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



1
Finalize chunking with full citation metadata
Build the markdown-table statement chunks and narrative chunks exactly as planned, but make sure every chunk carries company, statement_name, period, source_type, filing_url, and document_content_hash in metadata. Do this before anything else — every later step (eval, citation, routing) depends on this being right.
2
Add hybrid search (vector + keyword)
Layer a keyword search (e.g. SQLite FTS5, or Chroma's built-in keyword support) alongside your existing vector similarity search, and merge the two result sets. This fixes the common case where a query names an exact term (like a line-item label) that semantic search alone sometimes misses.
3
Add reranking on top of retrieval
Retrieve a wider set (top 15-20) with your current retriever, then use a cross-encoder reranker to reorder and cut down to the final top 3-5 before generation. This is the single highest-leverage accuracy improvement for the effort involved, and it's a well-known technique worth naming explicitly in your README.
4
Build a hand-labeled eval set with real metrics
Write 15-20 question/answer pairs across your 6 companies, mixing single-fact lookups and cross-company comparisons. For each, record which chunk(s) should be retrieved. Measure precision@k, recall@k, and MRR before and after step 2 and step 3 — this turns 'I improved retrieval' into a specific, quotable number.
5
Add a grounding check on generated answers
After the LLM generates an answer, run a simple check (regex or a second small LLM call) confirming any numbers or claims in the answer actually appear in the retrieved chunks. Reject or flag answers that don't pass. This matters specifically for financial data, where a wrong number is a credibility problem, not just noise.
6
Add query routing for structured vs. narrative questions
Detect whether a question is a direct numeric lookup (route to a structured query against your extracted statement data) versus a narrative question (route to your existing vector+keyword retrieval). Even a simple heuristic or a cheap classification prompt here demonstrates you understand your data has two distinct shapes and shouldn't be queried the same way.
7
Write the README with before/after numbers and citations
Document each design decision (the ones you already worked through, plus the new ones), and lead with your eval metrics showing quantified improvement from hybrid search and reranking. Include a worked example showing a question, the retrieved chunks with their citation metadata, and the final cited answer.s

Create a self correcting citation system

The fix isn't to try to guarantee complete metadata for every table — it's to accept sparse metadata as normal, and add a lower-confidence fallback field instead of overloading the existing ones.

Make section/heading genuinely optional in the schema, so a citation with empty metadata isn't a schema violation, just a citation with less context:
python
class Citation(BaseModel):
    item: str
    section: str = ""
    heading: str = ""
    text: str
    company: str
    year: str
    is_table: bool
    source: str
    file_path: str
    nearby_text: str = ""   # new: best-effort fallback, see below

This alone already stops "some tables miss context" from being a problem you have to solve perfectly — a citation with section="" is just retrievable on item/company/text instead, which is normal and fine for RAG.

Add a rolling "last seen text regardless of classification" buffer, separate from current_section/current_heading. This captures raw text your classifier couldn't confidently bucket, and only gets used as a fallback right when you hit a table — it never overwrites your working section/heading logic for prose citations:
python
last_seen_raw_text = ""

# inside the span loop, at the very top after computing span_text:
if span_text.strip():
    last_seen_raw_text = span_text.strip()

# in the table branch:
elif child_tag.name == "table":
    table_as_markdown = extract_table(child_tag)
    if table_as_markdown:
        citations.append(Citation(
            item=current_item, 
            section=current_section, 
            heading=current_heading, 
            nearby_text=last_seen_raw_text,   # best-effort context, whatever it is
            text=table_as_markdown,
            company=company,
            year=year,
            is_table=True,
            source=file_name,
            file_path=file_path))

Now every table citation gets section/heading when your classifier worked, and nearby_text as an unconditional fallback (even if it's just "(in millions)" or some unclassified span) — without touching how prose citations are built at all. Tables that were already working (like your GOOG cash-flow example) keep clean metadata; tables that weren't just get a weaker but non-empty signal instead of nothing.

For embedding, prepend whatever context exists rather than requiring it be structured. When you build the text you actually embed, do something like:
python
def build_embed_text(c: Citation) -> str:
    context_parts = [p for p in [c.section, c.heading, c.nearby_text] if p]
    context = " | ".join(dict.fromkeys(context_parts))  # dedupe, preserve order
    return f"{context}\n{c.text}" if context else c.text

This way each table chunk gets some contextual prefix when anything is available, gracefully degrades to just the table text when nothing is, and you never have to force every citation into the same metadata shape to make the pipeline work.

Bottom line: the reliability of section/heading genuinely varies per table depending on how that specific filer styled their caption — that's a real, permanent source of heterogeneity in 11 different companies' HTML, not a bug you can code your way out of completely. The practical move is to stop trying to guarantee uniform rich metadata, add a cheap always-populated fallback field, and let downstream retrieval/embedding handle graceful degradation instead.