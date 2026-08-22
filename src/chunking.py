
from typing import Any, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import chromadb
from hashlib import sha256
import sqlite3
import os

from clean import get_citations, Citation


model: SentenceTransformer = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
MAX_TOKEN_SIZE = 5000
chroma_client = chromadb.PersistentClient(path="../storage")
collection = chroma_client.get_or_create_collection(name="data_store")

# def chunk_file(citation: Citation):
#     # soup = BeautifulSoup(citation, 'html.parser')
#     # text_only = soup.get_text()

#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=["\n\n", "\n", ". ", "! ", "? "],
#         chunk_size=500,
#         chunk_overlap=50,
#     )

#     chunks = text_splitter.split_text(text_only)
#     chunks_keys = [(str(sha256(str(chunk).encode()).hexdigest())+str(i)) for i, chunk in enumerate(chunks)]

#     # embed using index to make sure there's no repetition i have to 
#     return chunks, chunks_keys

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

# Embedding
def insert_to_chroma(embedding, citation: Citation):
    citation_text_hash: str = sha256(citation.text.encode()).hexdigest()
    collection.upsert(
        ids=[citation_text_hash],
        embeddings=[embedding],
        documents=[citation.text],
        metadatas=[{
            "item": citation.item,
            "section": citation.section,
            "heading": citation.heading,
            "company": citation.company,
            "year": citation.year,
            "source:": citation.source,
            "file_path": citation.file_path
        }]
    )
    return collection

def get_tokens(text: str) -> List[str]:
    # prepared_text: tuple[str, dict[str, Any]] = tokenizer.prepare_for_tokenization(text, is_split_into_words=False)
    tokens: List[str] = tokenizer.tokenize(text)
    return tokens

def get_token_count(text: str) -> int:
    return len(get_tokens(text))

def chunk_processing(citation):
    text_to_embed = "search document: " + citation.item + " " + citation.section + " " + citation.heading + " " + citation.text
    embedding = model.encode(text_to_embed)
    insert_to_chroma(embedding, citation)

def recursive_chunking(token_count: int, citation: Citation):
    print("hit a point")
    # Decrease MAX_TOKEN_SIZE to see this play out and perfect the algo
    min_chunks_required = (token_count / MAX_TOKEN_SIZE).__ceil__()
    chunks : List = []

    chunk = ""
    chunk_size: int = 0
    paragraphs = citation.text.split("\n")
    for paragraph in paragraphs:
        paragraph_token_count = get_token_count(paragraph)
        if paragraph_token_count + chunk_size <= MAX_TOKEN_SIZE:
            chunk = chunk + "\n" + paragraph
            # chunk.join(paragraph, "\n") # is this syntax correct?
        else:
            chunks.append(chunks)
            chunk_size = paragraph_token_count
            chunk = paragraph

    for chunk in chunks:
        new_citation = citation
        new_citation.text = chunk
        chunk_processing(new_citation)


def tokenise(citations: List[Citation]):
    for citation in citations:
        token_count = get_token_count(citation.text)
        if token_count > MAX_TOKEN_SIZE:
            # calculate the min number of chunks needed
            recursive_chunking(token_count, citation)
        else:
            chunk_processing(citation)


if __name__ == "__main__":
    # chroma_client = chromadb.PersistentClient(path="../storage")
    # collection = chroma_client.get_or_create_collection(name="my_collection")
    conn = sqlite3.connect("../storage/document_ledger.db")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
    create_table(conn)

    data_directory = "../data"
    files = os.listdir(data_directory)    
    for file_name in files:
        full_path = os.path.join(data_directory, file_name)
        # print("full path", full_path)
        citations = get_citations(full_path)

        tokenise(citations)

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