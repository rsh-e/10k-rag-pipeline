from typing import Any, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sqlalchemy import table
from transformers import AutoTokenizer
import chromadb
from hashlib import sha256
import sqlite3
import os

from clean import get_citations, Citation


model: SentenceTransformer = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
MAX_TOKEN_SIZE = 800
chroma_client = chromadb.PersistentClient(path="../storage")
collection = chroma_client.get_or_create_collection(name="data_store")
conn = sqlite3.connect("../storage/document_ledger.db")


def create_table(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_name TEXT NOT NULL UNIQUE,
            path  TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL DEFAULT 'pending',
            ingested_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def create_fts5_table(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE VIRTUAL TABLE citations USING fts5(
            citation_text_hash UNINDEXED,
            item UNINDEXED,
            section UNINDEXED,
            heading UNINDEXED,
            text,
            is_table UNINDEXED,
            flatten_table,
            nearby_text, 
            company UNINDEXED,
            year UNINDEXED,
            source UNINDEXED,
            file_path UNINDEXED,
            tokenize='porter'
            );
        """
    )
    conn.commit()


def add_document(content_hash: str, conn: sqlite3.Connection):
    document_exists = check_document_exists(content_hash, conn)
    if not document_exists:
        conn.execute(
            """
            INSERT INTO documents(
                document_name,
                path, 
                content_hash
            ) VALUES (
            ?, ?, ?
            )
            """,
            (file_name.split(".")[0], file_name, content_hash),
        )
        conn.commit()
    else:
        print("Document exists")


def check_document_exists(content_hash: str, conn: sqlite3.Connection):
    cursor = conn.execute(
        """
        SELECT 1
        FROM documents
        WHERE content_hash = ?
        """,
        (content_hash,),
    )
    return cursor.fetchone() is not None


# Embedding
def insert_to_chroma(embedding, citation: Citation, citation_text_hash: str):
    collection.upsert(
        ids=[citation_text_hash],
        embeddings=[embedding],
        documents=[citation.text],
        metadatas=[
            {
                "item": citation.item,
                "section": citation.section,
                "heading": citation.heading,
                "company": citation.company,
                "flatten_table": citation.flatten_table,
                "is_table": citation.is_table,
                "nearby_text": citation.nearby_text,
                "year": citation.year,
                "source": citation.source,
                "file_path": citation.file_path,
            }
        ],
    )
    return collection


def insert_to_fts5(
    conn: sqlite3.Connection, citation: Citation, citation_text_hash: str
):
    if citation.nearby_text is None:
        citation.nearby_text = ""
    if citation.flatten_table is None:
        citation.flatten_table = ""
    conn.execute(
        """
        INSERT INTO citations(
            citation_text_hash,
            item,
            section,
            heading,
            text,
            is_table,
            flatten_table,
            nearby_text,
            company,
            year,
            source,
            file_path
        ) VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            citation_text_hash,
            citation.item,
            citation.section,
            citation.heading,
            citation.text,
            citation.is_table,
            citation.flatten_table,
            citation.nearby_text,
            citation.company,
            citation.year,
            citation.source,
            citation.file_path,
        ),
    )
    conn.commit()


def get_tokens(text: str) -> List[str]:
    # prepared_text: tuple[str, dict[str, Any]] = tokenizer.prepare_for_tokenization(text, is_split_into_words=False)
    tokens: List[str] = tokenizer.tokenize(text)
    return tokens


def get_token_count(text: str) -> int:
    return len(get_tokens(text))


def chunk_processing(conn: sqlite3.Connection, citation: Citation):
    # print(citation)
    # print()
    if citation.is_table:
        table_text = citation.flatten_table
        text_to_embed = (
            "search document: "
            + citation.heading
            + " "
            + citation.nearby_text
            + " "
            + table_text
        )
    else:
        text_to_embed = (
            "search document: "
            + citation.item
            + " "
            + citation.section
            + " "
            + citation.heading
            + " "
            + citation.text
        )
    embedding = model.encode(text_to_embed)
    citation_text_hash: str = sha256(citation.text.encode()).hexdigest()
    insert_to_chroma(embedding, citation, citation_text_hash)
    insert_to_fts5(conn, citation, citation_text_hash)


def recursive_chunking(conn: sqlite3.Connection, citation: Citation):
    chunks: List = []

    chunk = ""
    chunk_size: int = 0
    paragraphs = citation.text.split("\n")
    for paragraph in paragraphs:
        paragraph_token_count = get_token_count(paragraph)
        chunk_size += paragraph_token_count
        if chunk_size <= MAX_TOKEN_SIZE:
            chunk = chunk + paragraph + "\n"
        else:
            chunks.append(chunk)
            chunk_size = paragraph_token_count
            chunk = paragraph

    if chunk:
        chunks.append(chunk)

    for chunk in chunks:
        new_citation = citation
        new_citation.text = chunk
        chunk_processing(conn, new_citation)


def tokenise(conn: sqlite3.Connection, citations: List[Citation]):
    for citation in citations:
        if citation.is_table:
            chunk_processing(conn, citation)
        else:
            token_count = get_token_count(citation.text)
            if token_count > MAX_TOKEN_SIZE:
                # calculate the min number of chunks needed
                recursive_chunking(conn, citation)
            else:
                chunk_processing(conn, citation)


if __name__ == "__main__":
    create_table(conn)
    create_fts5_table(conn)

    path = "../data/amd-20251227.html"
    citations = get_citations(path)
    tokenise(conn, citations)

    # You want to uncomment this to chunk all files
    data_directory = "../data"
    files = os.listdir(data_directory)
    for file_name in files:
        full_path = os.path.join(data_directory, file_name)
        # print("full path", full_path)
        citations = get_citations(full_path)
        tokenise(conn, citations)
        print("done", file_name)

        # Make an entry on the table
        # Get the content hash during that time asw.
        # try:
        # content = str(from_path("../data/"+file_name).best())
        # content_hash = sha256(content.encode()).hexdigest()
        # add_document(content_hash, conn)

        # # chunk the file using the chunk file function
        # chunks, chunks_keys = chunk_file(content)

        # # embed the file in a collection
        # collection = get_collection(chunks=chunks, chunks_keys=chunks_keys)
        # print("done", file_name)

        # chunk the file using the chunk file function
        # embed the file in a collection
        # change the status in the table to 'done'

    print("done chunking")
