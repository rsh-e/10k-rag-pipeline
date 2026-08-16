from langchain_text_splitters import RecursiveCharacterTextSplitter
from charset_normalizer import from_path
import chromadb
from hashlib import sha256
import sqlite3
import os
from bs4 import BeautifulSoup

def chunk_file(content: str):
    soup = BeautifulSoup(content, 'html.parser')
    text_only = soup.get_text()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "! ", "? "],
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = text_splitter.split_text(text_only)
    chunks_keys = [(str(sha256(str(chunk).encode()).hexdigest())+str(i)) for i, chunk in enumerate(chunks)]

    # embed using index to make sure there's no repetition i have to 
    return chunks, chunks_keys

# Embedding
def get_collection(chunks, chunks_keys):
    collection = chroma_client.get_or_create_collection(name="my_collection")
    collection.upsert(ids=chunks_keys, documents=[str(c) for c in chunks])
    return collection

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
            (file_name.split(".")[0], file_name, content_hash)
        )
        conn.commit()
    else:
        print("Document exists")

def check_document_exists(content_hash: str, conn:sqlite3.Connection):
    cursor = conn.execute(
        """
        SELECT 1
        FROM documents
        WHERE content_hash = ?
        """,
        (content_hash,)
    )
    return cursor.fetchone() is not None


if __name__ == "__main__":
    chroma_client = chromadb.PersistentClient(path="../storage")
    collection = chroma_client.get_or_create_collection(name="my_collection")
    conn = sqlite3.connect("../storage/document_ledger.db")
    create_table(conn)

    files = os.listdir("../data")    
    for file_name in files:
        # Make an entry on the table
            # Get the content hash during that time asw.
        # try: 
        content = str(from_path("../data/"+file_name).best())
        content_hash = sha256(content.encode()).hexdigest()
        add_document(content_hash, conn)

        # chunk the file using the chunk file function
        chunks, chunks_keys = chunk_file(content)

        # embed the file in a collection
        collection = get_collection(chunks=chunks, chunks_keys=chunks_keys)
        print("done", file_name)


        # chunk the file using the chunk file function
        # embed the file in a collection
        # change the status in the table to 'done'
        pass

    print("done")