from re import I
import sqlite3
from typing import List

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
from main import get_cross_encoder_results, get_fts_results, get_rrf_results
import questions
from chromadb import IDs, QueryResult

K = 50

def print_results(question: str, context: QueryResult) -> None:
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

def calculate_metrics(actual_chunk_ids, retrieved_chunk_ids) -> int:
    found = []
    # print("actual: \n", actual_chunk_ids, "\nretrieved: \n", retrieved_chunk_ids)
    recall, precision, mrr = 0,0, 0 
    if len(actual_chunk_ids) == 0: print("Abstain")
    else:
        for retrieved in retrieved_chunk_ids:
            if retrieved in actual_chunk_ids:
                found.append(retrieved)

        precision = len(found) / K
        recall = len(found) / len(actual_chunk_ids)

        mrr = 0.0
        for rank, retrieved in enumerate(retrieved_chunk_ids, start=1):
            if retrieved in actual_chunk_ids:
                mrr = 1 / rank
                break
        
        print("precision: ", precision, "recall: ", recall, "mrr: ", mrr)

    return recall, precision, mrr

def main():
    chroma_client = chromadb.PersistentClient(path="../storage")
    collection = chroma_client.get_collection(name="data_store") 
    model: SentenceTransformer = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    conn = sqlite3.connect("../storage/document_ledger.db")
    cross_encoder: CrossEncoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    recall = 0
    precision = 0
    mrr = 0
    for item in questions.eval_set:
        print()
        question = item["question"]
        print(question)
        query_embedding = model.encode("search_query: " + question)

        embedded_results: QueryResult = collection.query(
            query_embeddings=[query_embedding],
            include=["documents", "metadatas", "distances"],
            n_results=K
        )

        # print_results(question, context)
        fts_results = get_fts_results(conn, question)
        rrf_results = get_rrf_results(fts_results, embedded_results)
        cross_encoder_results = get_cross_encoder_results(cross_encoder, question, rrf_results)

        retrieved_chunk_ids = []

        for result in cross_encoder_results:
            content_hash = result[0][0]
            print(content_hash)
            retrieved_chunk_ids.append(content_hash)

        retrieved_chunk_ids = retrieved_chunk_ids[0:10]
        actual_chunk_ids = item["relevant_chunks"]

        re_recall, re_precision, re_mrr =  calculate_metrics(actual_chunk_ids, retrieved_chunk_ids)
        recall += re_recall
        precision += re_precision
        mrr += re_mrr

    print("recall:", recall/100, "precision:", precision/100, "mrr:", mrr/100)


if __name__ == "__main__":
    main()