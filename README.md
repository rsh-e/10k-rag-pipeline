
# Overview
A citation-backed RAG system for querying SEC 10-K filings using hybrid retrieval, Reciprocal Rank Fusion (RRF), cross-encoder reranking, and domain-specific routing.

The RAG approach combines hybrid search using semantic search and lexical search which are fused with RRF (Reciprocal Ranked Fusion). The results from the search are then passed into a cross encoder which reranks the results to provide the most relevant information to the model. Additional strategies were used to improve retrieval quality. 

Retrieval qualtiy was measured using an extensive eval set (containing 95 answerable questions) which was developed with the help of a LLM (and manually cross-checked). A summary of retrieval quality is as follows:

| Retrieval configuration           |    Recall | Completeness |       MRR |
| --------------------------------- | --------: | -----------: | --------: |
| Embeddings only                   |     63.2% |        58.9% |     0.392 |
| RRF                               |     68.2% |        64.2% |     0.477 |
| Cross-encoder                     |     75.1% |        70.5% |     0.531 |
| **RRF + company + table routing** | **83.5%** |    **77.9%** | **0.607** |

The full retrieval pipeline improved recall by **20.3 percentage points** over embeddings-only retrieval.

ChromaDB has been used for semantic search and SQLite FTS5 has been used for lexical search. The embedding model used is ```nomic-ai/nomic-embed-text-v1.5```, the encoder model used is ```cross-encoder/ms-marco-MiniLM-L-6-v2``` and LLM model used is ```groq/compound``` provided by Groq. Model selection was based on embedding size, model size, and costs which were kept low.

## Architecture

```text
                         SEC 10-K filings
                                │
                          Parse + clean
                                │
                    ┌───────────┴───────────┐
                    │                       │
                Text chunks             Tables
                    │                       │
              Retrieval text     Linearised and converted to .md
                    │                       │
              ┌─────┴─────┐           ┌─────┴─────┐
              │           │           │           │
          Embeddings     FTS5      Embeddings    FTS5
              │           │           │           │
              └─────┬─────┘           └─────┬─────┘
                    │                       │
                    └───────────┬───────────┘
                                │
                           Query analysis
                                │
                   ┌────────────┼────────────┐
                   │            │            │
              Company       Table intent   Stopwords
              routing        routing       filtering
                   │            │            │
                   └────────────┼────────────┘
                                │
                    Dense + lexical candidates
                                │
                               RRF
                                │
                     Cross-encoder reranking
                                │
                              Top-K
                                │
                           Groq Compound
                                │
                    Citation-backed response
```


# Run the project
To run the project:
1) Clone the repo
2) ```uv run install```
3) Set your groq api key as ```GROQ_API_KEY="gsk-..."``` in ```src/.env```
4) ```uv run main.py``` for the chunking and querying
5) ```uv run streamlit run lit.py``` to view the streamlit app

# Project Structure
#### ```src/```
- ```main.py```: The main entry point, it checks if the documents have been chunked, chunks them if they haven't and then runs the query workflow which allows the user to ask the model questions about the filings.
- ```chunking.py```: Collects the filings available in the ```data/``` directory and chunks each of them. Documents being chunked are kept track of using a database and if the program stops, the chunking resumes from where it stopped.
- ```clean.py```: Given a file path, it will generate citations for the respective filing. These citations are chunked in chunking.py
- ```retrieval.py```: Contains the entire retrieval workflow from getting the vector embeddings and FTS results, fusing them and then reranking them with a cross encoder.
- ```query.py```: Connects to the Groq client and allows you to ask a query to it
- ```eval.py```: Runs the eval set and produces a summary of the results in eval_results.json which can be explored
- ```eval_results.json```: Results from the eval set are stored in this file
- ```eval_analysis.ipynb```: A notebook which can be used used to explore eval_results.json. The code in the notebook provides interesting summary statistics
- ```constants.py```: All constants (described in Upper case) used throughout the project
- ```prompt.py```: The system prompt
- ```questions.py```: The eval set, consists of questions with the necessary answers (chunks)
- ```lit.py```: Streamlit UI for the demo, completely vibecoded
#### ```data/```
Consists of the 10K filings
#### ```storage/```
Where the Chroma and SQLite databases live

# Strategies
## Parsing and Citations
The 10K documents used have all been filed using the workiva platform. Although the formatting is not identical to accurateley parse, they contain reasonable heuristics to generate citations. ```clean.py``` strips the file of images and xbrl tags and then runs an algorithm which identifies the Item, Section and Heading a text belongs to. Each citation is thus self contained as to what it describes making retrieval easier.
Tables are transformed into markdown for LLM use and linearised for retrieval, ```row_label, column_label: data```. This text is used for embedding whereas the markdown is the document. This is because LLMs work well with markdown.
Only 10 SEC filings have been used because each extra document would require more questions in the eval set to test.

## Chunking
The citations are then passed on to be chunked. In Chroma, the text is embedded with the item, section and heading. The same citation data + content hash is encoded as metadata. The FTS5 table also contains the same text and metadata. If a citation's text is too large, it's split and the metadata is still preserved.

## Retrieval
When a user asks a question to the model, the top 50 semantic results and top 50 from the lexical results are retrieved. These results are then fused with RRF and passed into an encoder which reranks the questions and returns the Top K results as context. Top K is 10 for the eval set as the model performs the best on it. For the Streamlit demo, top K 5 to stay within the limited context window on the free tier.
To optimise the quality of retrieval, 3 improvements were used:

1. **Stopwords**: The NLTK stopwords were used to make sure common words weren't ranked high in FTS. Along with those words, extra ```FINANCIAL_STOP_WORDS``` (found in ```constants.py```) were used to make common financial terms rank lower to minimise chunks with similar language.
2. **Company Routing**: Since the filings involved companies within the same sector (often competitors), it was observed that chunks from rival companies were being retrieved more often that the desired company because of the similar sector language used. To circumvent this, the required company (or companies) are identified from the prompt which are used to fetch results from those desired companies.
3. **Table Routing**: The encoder seemed to prefer the prose over the markdown tables or transformed tables which led to irrelevant chunks being ranked much higher. To ensure that a query be answered via tables, if the prompt contained 'table' or 'financial statement' it would collect only tables. ```TABLE_WORDS``` in ```constants.py``` contains the words used to check if the prompt contains a table.

## Querying
Groq provides the models for the query. The system prompt lives in ```prompt.py```. There is no memory, every query is self contained due to the context window.

## Eval
The eval set runs several different retrieval pipelines in order to identify improvement in retrieval quality. The eval set only measures retrieval. Answers could not be measured using LLM-as-judge due to rate limits. The metrics measured are precision (of which there is little practical use), recall, mrr and completeness (whether all the required chunks for a complete answer were retrieved).

## Findings
1. Retrieval with the stopwords, company and table routing is clearly the superior strategy with recall at 83.5% and completeness at 77.9%
2. Most chunks appear in rank 1 with 77.8% of chunks appearing within the first 5 ranks. Chunks for 10 questions (out of 95 answerable) do not appear in the top 10 at all.
3. Top K at 10 provides the best tradeoff for recall and amount being passed in to the LLM. Top K at 20 has recall at 90%
4. Retrieval quality is often dependent on the specific language used within the annual report, if the specific style of langauge is not present within the prompt, it hurts retrieval.


# Notes
1. You get a warning from chroma that says one of the tables is too large to embed and will be truncated. The said table is an exhibit index from ```ko-20251231.html``` and has no value in retrieval. While I could split tables down further to smaller chunks, it would hurt recall and precision because you lose the structure of the table as well as context. It would also lead to chunk changing, requiring revision of the eval set.
2. Table citations on Streamlit don't render well because the embedding was passed in as grid format. While this is a simple fix, it would completely change the hash of the text requiring me to completely change the hashes on the eval set.

# Potential future improvements
1. Make ingestion and chunking multithreaded.
2. Retrieval strategy where if the first Top K chunks dont provide enough context, retrieve the next Top K.
3. Retrieval strategy where it identifies the number of companies required to be retrieved and retrieves a pool for each. The chunks can then be reranked and provided to the LLM.
4. Figure out a better way to parse the filings instead of relying on formatting heuristics.

# AI Usage
AI was used for the following:
1. Generating the Streamlit UI
2. Generating the questions and right chunks (via hashes), although half were wrong and I had to fix them
3. Writing the specific functions which promoted the first row of the dataframe to the column labels and collapsed the dollar symbol into a single row
4. Explain certain concepts
5. Generate the markdown diagram for the README

Everything else was programmed and strategised by myself.
