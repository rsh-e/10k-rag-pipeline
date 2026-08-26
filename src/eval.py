from typing import List

import chromadb
from sentence_transformers import SentenceTransformer
import questions
from chromadb import IDs, QueryResult

K = 10

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
    print("actual: \n", actual_chunk_ids, "\nretrieved: \n", retrieved_chunk_ids)
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

    

def calculate_recall(question: str, context: QueryResult) -> int:
    pass

def calculate_mrr(question: str, context: QueryResult) -> int:
    pass

def main():
    print("hi")
    chroma_client = chromadb.PersistentClient(path="../storage")
    collection = chroma_client.get_collection(name="data_store") 
    model: SentenceTransformer = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    for item in questions.eval_set:
        print()
        question = item["question"]
        print(question)
        query_embedding = model.encode("search_query: " + question)

        context: QueryResult = collection.query(
            query_embeddings=[query_embedding],
            include=["documents", "metadatas", "distances"],
            n_results=K
        )

        # print_results(question, context)

        retrieved_chunk_ids = context["ids"][0]
        actual_chunk_ids = item["relevant_chunks"]
        # print("here")

        calculate_metrics(actual_chunk_ids, retrieved_chunk_ids)
        # precision = calculate_precision(actual_chunk_ids, retrieved_chunk_ids)
        # recall = calculate_recall(actual_chunk_ids, retrieved_chunk_ids)
        # mrr = calculate_mrr(actual_chunk_ids, retrieved_chunk_ids)

    # print("Precision@K:", precision)
    # print("Recall@K:", recall)
    # print("MRR@K:", mrr)



if __name__ == "__main__":
    main()