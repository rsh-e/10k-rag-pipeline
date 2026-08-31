from importlib import metadata
import json
import os
import re
import sqlite3
from typing import List

from cv2 import inpaint
from groq import Groq
import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer, cross_encoder

from clean import Citation

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

N = 50

def print_results(question: str, context: dict) -> None:
    print("=" * 100)
    print(f"QUESTION: {question}")
    print("=" * 100)
    documents = context["documents"][0]
    metadatas = context["metadatas"][0]
    distances = context["distances"][0]
    for rank, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances), start=1):
        print(f"\n--- Rank {rank} | distance={distance:.4f} | "
              f"company={metadata.get('company')} | source={metadata.get('source')} | "
              f"section={metadata.get('section')} ---")
        print(document.strip())
        print("-" * 100)
    print("\n")


def build_fts_query(question: str) -> str:
    tokens = word_tokenize(question.lower())
    stop_words = set(stopwords.words("english"))
    terms = [word for word in tokens if word not in stop_words]

    # split on non-word characters, drop empty strings, lowercase not required (FTS5 handles case)
    if not terms:
        return '""'  # empty/degenerate query, will just match nothing

    quoted_terms = [f'"{term}"' for term in terms]
    return " OR ".join(quoted_terms)


def get_fts_results(conn: sqlite3.Connection, question: str) -> List[any]:
    fts_query = build_fts_query(question)
    cursor = conn.execute(
        """
        SELECT citation_text_hash, item, section, heading, text, is_table, flatten_table, nearby_text, company, year, source, file_path
        FROM citations
        WHERE citations MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (fts_query, N)
    )
    results = cursor.fetchall()
    # for i in results:
    #     print(i)
    #     print()
    return results


def embedded_citation_normaliser(text, metadata, rank):
    metadata = metadata[rank]
    return Citation(
        item=metadata["item"],
        section=metadata["section"],
        heading=metadata["heading"],
        text=text,
        is_table=metadata["is_table"],
        flatten_table=metadata["flatten_table"],
        nearby_text=metadata["nearby_text"],
        company=metadata["company"],
        year=metadata["year"],
        source=metadata["source"],
        file_path=metadata["file_path"]
    )

def fts_citation_normaliser(result):
    return Citation(
        item=result[1],
        section=result[2],
        heading=result[3],
        text=result[4],
        is_table=result[5],
        flatten_table=result[6], 
        nearby_text=result[7],
        company=result[8],
        year=result[9],
        source=result[10],
        file_path=result[11]
    )
    

def get_rrf_results(fts_results: List[tuple], embedded_results: chromadb.QueryResult) -> List[tuple]:
    # Shape of weights looks liek this:
    # weights = {hash: {Citation:, score:}}
    # Go through the fts_results add in the hash, text and rrf to the weights
    # Go through the embedded_results, add in the hash, text and rrf to the weights
    # If you see that a hash already exists, just add to that
    # sort and then get the reranked results?

    weights = {}
    # Go through the fts_results add in the hash, text and rrf to the weights
    for rank, result in enumerate(fts_results):
        text_hash =  result[0]
        citation: Citation = fts_citation_normaliser(result)
        score = 1 / (60 + rank)
        weights[text_hash] = {"citation": citation, "score": score}

    # Go through the embedded_results, add in the hash, text and rrf to the weights
    text_hashs = embedded_results["ids"][0]
    documents = embedded_results["documents"][0]
    metadata = embedded_results["metadatas"][0]
    for rank, text_hash in enumerate(text_hashs):
        text = documents[rank]
        citation: Citation = embedded_citation_normaliser(text, metadata, rank)
        current_score = 1 / (60 + rank)
        if text_hash in weights:
            old_score = weights[text_hash]["score"]
            combined_score = old_score + current_score
            weights[text_hash]["score"] = combined_score
        else:
            weights[text_hash] = {"citation": citation, "score": current_score} 

    ranked = sorted(weights.items(), key=lambda item: item[1]["score"], reverse=True)
    return ranked

def get_cross_encoder_results(cross_encoder: CrossEncoder, question, rrf_results: tuple):
    pairs = []
    for result in rrf_results:
        citation: Citation = result[1]["citation"]
        # print(citation)
        # if citation.is_table:
        #     text = citation.flatten_table
        #     # print(text)
        # else:
        text = citation.text
        pairs.append((question, text))

    scores = cross_encoder.predict(pairs)
    results_with_scores = list(zip(rrf_results, scores))
    ranked = sorted(results_with_scores, key=lambda x: x[1], reverse=True)
    return ranked



def main():
    N = 50
    chroma_client = chromadb.PersistentClient(path="../storage")
    collection = chroma_client.get_collection(name="data_store") 
    model: SentenceTransformer = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    conn = sqlite3.connect("../storage/document_ledger.db")
    cross_encoder: CrossEncoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


    all_data = collection.get(
    include=["embeddings", "metadatas", "documents"],
    where={"company": "amd"}
    )
    
    
    # print(len(all_data["ids"]))  # sanity check — should match however many chunks you ingested

    # for id_, metadata, document in zip(all_data["ids"], all_data["metadatas"], all_data["documents"]):
    #     print(f"id={id_}")
    #     print(f"company={metadata.get('company')} item={metadata.get('item')} section={metadata.get('section')} heading={metadata.get('heading')} source={metadata.get('source')}")
    #     print(f"is_table={metadata.get('is_table')} flatten_table={metadata.get('flatten_table')}")
    #     print(document)
    #     print("§" * 100)

    # data_directory = "../data"
    # files = os.listdir(data_directory)    
    # companies = sorted({file_name.split("-")[0] for file_name in files})

    # for company in companies:
    #     all_data = collection.get(
    #         include=["documents", "metadatas"],
    #         where={"company": company}
    #     )
    #     dump = [
    #         {"hash": id_, "text": doc, **meta}
    #         for id_, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"])
    #     ]
    #     with open(company + ".json", "w") as f:
    #         json.dump(dump, f, indent=2, ensure_ascii=False)


    # write to json


    while True: 
        question = input("Ask your question:")
        query_embedding = model.encode("search_query: " + question)

        embedded_results = collection.query(
            query_embeddings=[query_embedding],
            include=["documents", "metadatas", "distances"],
            n_results=N
        )

        
        fts_results = get_fts_results(conn, question)
        print(question)

        rrf_results = get_rrf_results(fts_results, embedded_results)
        cross_encoder_results = get_cross_encoder_results(cross_encoder, question, rrf_results)

        # for i in cross_encoder_results:
        #     print(i)
        # print("\n")
        # print("=" * 10)

    # print("len: ", len(cross_encoder_results[]))  
        # print_results(question, context)

    # Calling Groq
        template = f"""
        SYSTEM:
        You are a financial research assistant specializing in SEC filings (primarily 10-Ks) for the following companies: AMD, American Express (AXP), ConocoPhillips (COP), Chevron (CVX), Alphabet (GOOG), Johnson & Johnson (JNJ), Coca-Cola (KO), Meta (META), NVIDIA (NVDA), and PepsiCo (PEP).

        You will be given a user question and a set of retrieved context chunks pulled from these companies' filings. Each chunk may be prose or a flattened table, and includes metadata such as company, source, section, and item.

        Answer using ONLY the information in the provided context. Do not use outside knowledge, and do not guess or extrapolate beyond what the context states. If the context does not contain enough information to answer the question, say so directly — do not speculate.

        Response guidelines:
        - Be concise and direct. Lead with the answer, not preamble.
        - Default to plain prose or short bullet points. Only use a table when the data is genuinely tabular (e.g. comparing the same metric across multiple companies or years) — most answers do not need one.
        - Never fabricate figures, dates, or company names not present in the context.
        - When citing a source, refer to it naturally by company and section/item (e.g. "per AMD's 10-K, Item 1A") — never reference internal IDs, hashes, or chunk numbers.
        - If multiple companies are relevant, organize the answer by company using short headers or bold labels, not a table, unless comparing a single shared metric.
        - Avoid restating the question back to the user.
        - Keep formatting clean: use bold for key terms/figures, bullet points for lists, and short paragraphs for explanations. Avoid nested formatting or excessive headers for short answers.
        
        The question provided to you is: {question}
        The context provided to you is: {cross_encoder_results[0:5]}
        """


        print(cross_encoder_results[0:5])

        # Prompt 
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        chat_completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
            {
                "role": "user",
                "content": template
            }
            ],
            temperature=1,
            max_completion_tokens=1000,
            top_p=1,
            stream=False,
            stop=None
        )
        
        answer = chat_completion.choices[0].message.content
        print(answer)
        print(chat_completion.usage)          # prompt/completion/total tokens
        print(chat_completion.choices[0].finish_reason)  # "stop", "length", etc.
        print("//" * 10)

if __name__ == "__main__":
    main()
