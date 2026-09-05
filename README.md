
# Overview
The following project describes a RAG pipeline built to ingest, chunk and retrieve information from SEC 10K filings to provide citable context to an LLM Query. The RAG approach combines hybrid search using vector embeddings and FTS which are fused RRF. The results from the search are then passed into a cross encoder which reranks the chunks to provide the most relevant information to the model. Other optimisations have also been used to improve retrieval quality. To test the quality of retrieval, an eval set has been developed with the help of a LLM (and cross checked by myself). A summary of the retrieval quality is as follows:

| config                        |   recall |   completeness |   mrr |
|:------------------------------|---------:|---------------:|------:|
| embed only                    |    0.632 |          0.589 | 0.392 |
| rrf only                      |    0.682 |          0.642 | 0.477 |
| encoder only                  |    0.751 |          0.705 | 0.531 |
| rrf + company + table routing |    0.835 |          0.779 | 0.607 |

ChromaDB has been used for vector embeddings and SQLite FTS5 has been used for FTS. The embedding model used is ```nomic-ai/nomic-embed-text-v1.5```, the encoder model used is ```cross-encoder/ms-marco-MiniLM-L-6-v2``` and LLM model used is ```groq/compound``` provided by Groq. Model selection was based on embedding size, model size, and costs which were kept low.

# Run the project
To run the download and run project:
1) Clone the repo
2) ```uv run install```
3) Set your groq api key
4) ```uv run main.py```
(CHANGE SO THAT YOU JUST HAVE TO RUN MAIN AND IT WILL RUN CHUNKING FOR YOU IF NEEDS BE)

# Project Structure
#### ```src/```
- ```main.py```: The main entry point, it checks if the documents have been chunked, chunks them if they haven't and then runs the query workflow which allows the user to ask the model questions about the filings.
- ```chunking.py```: Collects the filings availble in the data/ directory and chunks each of them. Documents being chunked are kept track of using a database and if the program stops, the chunking resumes from where it stopped.
- ```clean.py```: Given a file path, it will generate citations for the respective filing. These citations are chunked in chunking.py
- ```retrieval.py```: Contains the entire retrieval workflow from getting the vector embeddings and FTS results, fusing them and then reranking them with a cross encoder.
- ```query.py```: Connects to the Groq client and allows you to ask a query to it
- ```eval.py```: Runs the eval set and produces a summary of the results in eval_results.json which can be explored
- ```eval_results.json```: Results from the eval set are stored in this file
- ```eval_analysis.ipynb```: A notebook which can be used used to explore eval_results.json. The code in the notebook provides interesting summary statistics
- ```constansts.py```: All constants (described in Upper case) used throughout the project
- ```prompt.py```: The system prompt
- ```questions.py```: The eval set, consists of questions with the necessary answers (chunks)
#### ```data/```
Consists of the 10K filings
#### ```storage/```
Where the Chroma and SQLite databases live

# Strategies
## Parsing and Citations
## Chunking
## Retrieval
### Vector Embedding
### FTS
### RRF
### Encoder
### Stopwords
### Company routing
### Table routing
## Querying
## Eval
## Findings

# AI Usage

