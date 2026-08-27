from importlib import metadata
import re
import sqlite3
from typing import List

from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

from clean import Citation
import questions

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
    # split on non-word characters, drop empty strings, lowercase not required (FTS5 handles case)
    terms = re.findall(r"\w+", question)
    if not terms:
        return '""'  # empty/degenerate query, will just match nothing
    quoted_terms = [f'"{term}"' for term in terms]
    return " OR ".join(quoted_terms)


def get_fts_results(conn: sqlite3.Connection, question: str) -> List[any]:
    fts_query = build_fts_query(question)
    cursor = conn.execute(
        """
        SELECT citation_text_hash, item, section, heading, text, company, year, source, file_path
        FROM citations
        WHERE citations MATCH ?
        ORDER BY rank
        LIMIT 10
        """,
        (fts_query,)
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
        company=result[5],
        year=result[6],
        source=result[7],
        file_path=result[8]
    )
    

def get_reranked_results(fts_results: List[tuple], embedded_results: chromadb.QueryResult):
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


def main():
    N = 10
    chroma_client = chromadb.PersistentClient(path="../storage")
    collection = chroma_client.get_collection(name="data_store") 
    model: SentenceTransformer = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    conn = sqlite3.connect("../storage/document_ledger.db")

    all_data = collection.get(
    include=["embeddings", "metadatas", "documents"],
    where={"company": "amd"}
    )
    
    
    print(len(all_data["ids"]))  # sanity check — should match however many chunks you ingested

    for id_, metadata, document in zip(all_data["ids"], all_data["metadatas"], all_data["documents"]):
        print(f"id={id_}")
        print(f"company={metadata.get('company')} item={metadata.get('item')} section={metadata.get('section')} heading={metadata.get('heading')} source={metadata.get('source')}")
        print(document)
        print("-" * 100)

    for item in questions.eval_set:
        question = item["question"]
        query_embedding = model.encode("search_query: " + question)

        embedded_results = collection.query(
            query_embeddings=[query_embedding],
            include=["documents", "metadatas", "distances"],
            n_results=N
        )

        
        fts_results = get_fts_results(conn, question)
        # print(question)

        # print_results(question, embedded_results)

        # for i in fts_results:
        #     print(i)

        # ranked_results = get_reranked_results(fts_results, embedded_results)
        # for text_hash, data in ranked_results:
        #     # print(i["documents"][0])
        #     print(data["score"], data["citation"])

        # print("\n")
        # print("=" * 10)
        
        # print_results(question, context)

    # Calling Groq
    # template = f"""SYSTEM: You are a chatbot that specialises in providing information about AI Chip Companies. 
    # Information can include financial information, management, risks etc. All your responses must be factual. If you 
    # don't know the answer, say you don't know.
    # Respond to the following question: {question} only from the below context: {context}
    # """


    # Prompt 
    # client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    # chat_completion = client.chat.completions.create(
    #     model="llama-3.3-70b-versatile",
    #     messages=[
    #     {
    #         "role": "user",
    #         "content": template
    #     }
    #     ],
    #     temperature=1,
    #     max_completion_tokens=2048,
    #     top_p=1,
    #     stream=False,
    #     stop=None
    # )

if __name__ == "__main__":
    main()
