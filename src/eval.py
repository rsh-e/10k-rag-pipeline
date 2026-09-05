import json
import sqlite3
from pathlib import Path
from typing import List

from sentence_transformers import CrossEncoder

from constants import DOCUMENT_DB_NAME, ENCODER_MODEL, NUMBER_OF_CITATIONS, TOP_K
import questions

from retrieval import check_question_uses_tables, get_cross_encoder_results, get_fts_results, get_embedded_results, get_rrf_results, identify_companies


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


def calculate_metrics(
    primary_chunk_ids, equivalent_chunk_ids, retrieved_chunk_ids
) -> tuple[int, int, int]:
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
            recall = max(
                len(primary_found) / len(primary_chunk_ids),
                len(equivalent_found) / len(equivalent_chunk_ids),
            )
        else:
            recall = len(primary_found) / len(primary_chunk_ids)

        primary_complete = set(primary_found) == set(primary_chunk_ids)
        equivalent_complete = len(equivalent_chunk_ids) > 0 and set(equivalent_found) == set(equivalent_chunk_ids)
        completeness = 1 if primary_complete or equivalent_complete else 0

        precision = (len(primary_found) + len(equivalent_found)) / len(
            retrieved_chunk_ids
        )

        mrr = 0.0
        for rank, retrieved in enumerate(retrieved_chunk_ids, start=1):
            if retrieved in primary_chunk_ids or retrieved in equivalent_chunk_ids:
                mrr = 1 / rank
                break

        # print("precision: ", precision, "recall: ", recall, "mrr: ", mrr)

    return recall, precision, mrr, completeness


def main():
    results = []

    conn = sqlite3.connect(DOCUMENT_DB_NAME)
    cross_encoder: CrossEncoder = CrossEncoder(ENCODER_MODEL)

    for item in questions.eval_set:
        question: str = item["question"]
        companies: List[str] = identify_companies(question)
        needs_table: bool = check_question_uses_tables(question)

        # Chunks from vector
        embedded_chunks_no_companies_no_table = get_embedded_results(question=question, companies=[], needs_table=False)
        embedded_chunks = get_embedded_results(question=question, companies=companies, needs_table=needs_table)

        # Chunks from fts
        fts_chunks_no_companies_no_table = get_fts_results(conn=conn, question=question, companies=[], needs_table=False)
        fts_chunks = get_fts_results(conn=conn, question=question, companies=companies, needs_table=needs_table)

        # Chunks from vector + fts
        rrf_chunks_no_companies_no_table = get_rrf_results(fts_chunks_no_companies_no_table, embedded_chunks_no_companies_no_table)
        rrf_chunks = get_rrf_results(fts_chunks, embedded_chunks)

        # Chunks from vector + fts + cross_encoder
        encoder_chunks_no_companies_no_table = get_cross_encoder_results(cross_encoder=cross_encoder, question=question, rrf_results=rrf_chunks_no_companies_no_table)
        encoder_chunks = get_cross_encoder_results(cross_encoder=cross_encoder, question=question, rrf_results=rrf_chunks)

        configs = [
            ("embed only", embedded_chunks_no_companies_no_table, "embed"),
            ("rrf only", rrf_chunks_no_companies_no_table, "rrf"),
            ("encoder only", encoder_chunks_no_companies_no_table, "encoder"),
            ("rrf + company + table routing", encoder_chunks, "encoder"),
        ]

        primary_chunk_ids = item["primary_chunks"]
        equivalent_chunk_ids = item["equivalent_chunks"]

        for config_name, chunks, chunk_type in configs:
            retrieved_chunk_ids = []

            if chunk_type == "embed":
                for text_hash in chunks["ids"][0]:
                    retrieved_chunk_ids.append(text_hash)
            elif chunk_type == "rrf":
                for result in chunks:
                    retrieved_chunk_ids.append(result[0])
            else:
                for result in chunks:
                    retrieved_chunk_ids.append(result[0][0])

            retrieved_chunk_ids = retrieved_chunk_ids[0:TOP_K]

            re_recall, re_precision, re_mrr, re_completeness = calculate_metrics(
                primary_chunk_ids, equivalent_chunk_ids, retrieved_chunk_ids
            )

            row = {
                "id": item["id"],
                "config": config_name,
                "question": question,
                "type": item["type"],
                "difficulty": item["difficulty"],
                "hop": item["hop"],
                "companies": item["companies"],
                "abstain": item.get("abstain", False),
                "recall": re_recall,
                "precision": re_precision,
                "mrr": re_mrr,
                "completeness": re_completeness,
            }
            results.append(row)

        print(item["id"], "done")

    answerable = [q for q in questions.eval_set if not q.get("abstain")]
    n = len(answerable)

    output = {
        "n_answerable": n,
        "n_total": len(questions.eval_set),
        "top k": TOP_K,
        "citations retrieved": NUMBER_OF_CITATIONS,
        "results": results,
    }

    output_path = Path(__file__).resolve().parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("wrote", output_path)


if __name__ == "__main__":
    main()
