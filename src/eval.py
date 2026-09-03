from re import I
import sqlite3
from typing import List

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
from sympy import comp
from main import check_question_uses_tables, get_cross_encoder_results, get_fts_results, get_retrieved_chunks, get_rrf_results, identify_companies
import questions
from chromadb import IDs, QueryResult



K = 50
SHOW_RECALL_MISSES = False

def get_type(type: str, results: List[dict]):
    for item in results:
        if item["type"] == type:
            print(item)
            print()

def get_difficulty(difficulty: str, results: List[dict]):
    for item in results:
        if item["difficulty"] == difficulty:
            print(item)
            print()

def get_company(company: str, results: List[dict]):
    for item in results:
        if company in item["companies"]:
            print(item)
            print()

def get_chunk_text(conn: sqlite3.Connection, chunk_id: str) -> str:
    row = conn.execute(
        """
        SELECT company, section, heading, is_table, text, flatten_table
        FROM citations
        WHERE citation_text_hash = ?
        LIMIT 1
        """,
        (chunk_id,),
    ).fetchone()
    if not row:
        return "(hash not in store)\n"
    company, section, heading, is_table, text, flatten_table = row
    body = flatten_table if is_table and flatten_table else text
    return (
        f"company={company} section={section} heading={heading} is_table={is_table}\n"
        f"{body}\n"
    )

def calculate_metrics(primary_chunk_ids, equivalent_chunk_ids, retrieved_chunk_ids) -> tuple[int, int, int]:
    primary_found = []
    equivalent_found = []
    recall, precision, mrr, completeness = 0, 0, 0, 0
    if len(primary_chunk_ids) == 0: 
        print("Abstain")
    else:
        for retrieved in retrieved_chunk_ids:
            if retrieved in primary_chunk_ids:
                primary_found.append(retrieved)
            if retrieved in equivalent_chunk_ids:
                equivalent_found.append(retrieved)

        if len(equivalent_chunk_ids) > 0:
            recall = max(len(primary_found)/len(primary_chunk_ids), len(equivalent_found)/len(equivalent_chunk_ids))
        else:
            recall = len(primary_found)/len(primary_chunk_ids)

        primary_complete = set(primary_found) == set(primary_chunk_ids)
        equivalent_complete = (
            len(equivalent_chunk_ids) > 0
            and set(equivalent_found) == set(equivalent_chunk_ids)
        )
        completeness = 1 if primary_complete or equivalent_complete else 0

        precision = (len(primary_found) + len(equivalent_found)) / len(retrieved_chunk_ids)

        mrr = 0.0
        for rank, retrieved in enumerate(retrieved_chunk_ids, start=1):
            if retrieved in primary_chunk_ids or retrieved in equivalent_chunk_ids:
                mrr = 1 / rank
                break
        
        print("precision: ", precision, "recall: ", recall, "mrr: ", mrr)

    return recall, precision, mrr, completeness

def main():
    recall = 0
    precision = 0
    mrr = 0
    completeness = 0

    results = []

    for item in questions.eval_set:
        question = item["question"]
        retrieved_chunks = get_retrieved_chunks(question=question, top_k=10)
       
        retrieved_chunk_ids = []
        for result in retrieved_chunks:
            content_hash = result[0][0]
            # print(content_hash)
            retrieved_chunk_ids.append(content_hash)

        retrieved_chunk_ids = retrieved_chunk_ids[0:10]
        primary_chunk_ids = item["primary_chunks"]
        equivalent_chunk_ids = item["equivalent_chunks"]

        re_recall, re_precision, re_mrr, re_completeness =  calculate_metrics(primary_chunk_ids, equivalent_chunk_ids, retrieved_chunk_ids)
        # add to the new results
        item["results"] = {"recall":re_recall, "precision":re_precision, "mrr":re_mrr, "completeness":re_completeness}
        item["retrieved_chunks"] = retrieved_chunk_ids
        results.append(item)
        
        recall += re_recall
        precision += re_precision
        mrr += re_mrr
        completeness += re_completeness

        # Calculate new metric where it sees what percentage of chunks recieved was from the correct doc

    # n = len(questions.eval_set)
    answerable = [q for q in questions.eval_set if not q.get("abstain")]
    n = len(answerable)
    print(n)
    print("recall:", recall/n, "precision:", precision/n, "mrr:", mrr/n, "completeness:", completeness/n)


if __name__ == "__main__":
    main()