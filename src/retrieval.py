import sqlite3
from typing import List
import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer, cross_encoder

from clean import Citation

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from constants import COLLECTION_NAME, COMPANIES, DOCUMENT_DB_NAME, EMBEDDING_MODEL, ENCODER_MODEL, FINANCIAL_STOP_WORDS, NLTK_MODULES, NUMBER_OF_CITATIONS, STORAGE_PATH, TOP_K

chroma_client = chromadb.PersistentClient(path=STORAGE_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)
model: SentenceTransformer = SentenceTransformer(
    EMBEDDING_MODEL, trust_remote_code=True
)
conn = sqlite3.connect(DOCUMENT_DB_NAME)
cross_encoder: CrossEncoder = CrossEncoder(ENCODER_MODEL)


def download_nltk_modules():
    for path, module in NLTK_MODULES.items():
        try:
            nltk.data.find(path)
        except:
            nltk.download(module)
    
def build_fts_query(question: str) -> str:
    tokens = word_tokenize(question.lower())
    stop_words = set(stopwords.words("english"))
    for i in FINANCIAL_STOP_WORDS:
        stop_words.add(i)
    terms = [word for word in tokens if word not in stop_words]

    if not terms:
        return '""'  

    quoted_terms = [f'"{term}"' for term in terms]
    return " OR ".join(quoted_terms)


def get_fts_results(
    conn: sqlite3.Connection, question: str, companies: List[str], needs_table: bool
) -> List[any]:
    fts_query = build_fts_query(question)
    if len(companies) == 0:
        cursor = conn.execute(
            """
            SELECT citation_text_hash, item, section, heading, text, is_table, flatten_table, nearby_text, company, year, source, file_path
            FROM citations
            WHERE citations MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, NUMBER_OF_CITATIONS),
        )
        results = cursor.fetchall()
    elif len(companies) > 0 and needs_table:
        needs_table = 1
        placeholders = ", ".join("?" for i in companies)
        cursor = conn.execute(
            f"""
            SELECT citation_text_hash, item, section, heading, text, is_table, flatten_table, nearby_text, company, year, source, file_path
            FROM citations
            WHERE citations MATCH ? AND company in ({placeholders}) AND is_table = ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, *companies, needs_table, NUMBER_OF_CITATIONS),
        )
        results = cursor.fetchall()
    else:
        placeholders = ", ".join("?" for i in companies)
        cursor = conn.execute(
            f"""
            SELECT citation_text_hash, item, section, heading, text, is_table, flatten_table, nearby_text, company, year, source, file_path
            FROM citations
            WHERE citations MATCH ? AND company in ({placeholders})
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, *companies, NUMBER_OF_CITATIONS),
        )
        results = cursor.fetchall()

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
        file_path=metadata["file_path"],
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
        file_path=result[11],
    )


def get_rrf_results(
    fts_results: List[tuple], embedded_results: chromadb.QueryResult
) -> List[tuple]:
    # Shape of weights looks like this:
    # weights = {hash: {Citation:, score:}}
    
    weights = {}

    # Go through the fts_results add in the hash, text and rrf to the weights
    for rank, result in enumerate(fts_results):
        text_hash = result[0]
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


def get_cross_encoder_results(
    cross_encoder: CrossEncoder, question, rrf_results: tuple
):
    pairs = []
    for result in rrf_results:
        citation: Citation = result[1]["citation"]
        if citation.is_table and citation.flatten_table:
            text = citation.flatten_table
        else:
            text = citation.text
        pairs.append((question, text))

    scores = cross_encoder.predict(pairs)
    results_with_scores = list(zip(rrf_results, scores))
    ranked = sorted(results_with_scores, key=lambda x: x[1], reverse=True)
    return ranked


def identify_companies(question: str) -> List[str]:
    companies_in_question = []
    question = question.lower()
    for company in COMPANIES:
        for alias in COMPANIES[company]:
            if (alias in question) and (company not in companies_in_question):
                companies_in_question.append(company)

    return companies_in_question


def check_question_uses_tables(question: str) -> bool:
    question = question.lower()
    table_synonms = ["table", "statement", "statements", "tables"]
    for synonm in table_synonms:
        if synonm in question:
            return True

    return False

def get_retrieved_chunks(question: str):
    query_embedding = model.encode("search_query: " + question)
    companies = identify_companies(question)
    needs_table = check_question_uses_tables(question)
    if len(companies) > 0 and needs_table:
        embedded_results = collection.query(
            query_embeddings=[query_embedding],
            include=["documents", "metadatas", "distances"],
            where={
                "$and": [{"company": {"$in": companies}}, {"is_table": needs_table}]
            },
            n_results=NUMBER_OF_CITATIONS,
        )
    elif len(companies) > 0 and not needs_table:
        embedded_results = collection.query(
            query_embeddings=[query_embedding],
            include=["documents", "metadatas", "distances"],
            where={"company": {"$in": companies}},
            n_results=NUMBER_OF_CITATIONS,
        )
    else:
        embedded_results = collection.query(
            query_embeddings=[query_embedding],
            include=["documents", "metadatas", "distances"],
            n_results=NUMBER_OF_CITATIONS,
        )
    fts_results = get_fts_results(conn, question, companies, needs_table)
    rrf_results = get_rrf_results(fts_results, embedded_results)
    cross_encoder_results = get_cross_encoder_results(
        cross_encoder, question, rrf_results
    )

    return cross_encoder_results[0:TOP_K]