from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

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

def main():
    chroma_client = chromadb.PersistentClient(path="../storage")
    collection = chroma_client.get_collection(name="data_store") 
    model: SentenceTransformer = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    all_data = collection.get(
    include=["embeddings", "metadatas", "documents"],
    where={"company": "ko"}
    )
    
    
    # print(len(all_data["ids"]))  # sanity check — should match however many chunks you ingested

    for id_, metadata, document in zip(all_data["ids"], all_data["metadatas"], all_data["documents"]):
        print(f"id={id_}")
        print(f"company={metadata.get('company')} item={metadata.get('item')} section={metadata.get('section')} heading={metadata.get('heading')} source={metadata.get('source')}")
        print(document)
        print("-" * 100)

    for item in questions.eval_set:
        question = item["question"]
        query_embedding = model.encode("search_query: " + question)

        context = collection.query(
            query_embeddings=[query_embedding],
            include=["documents", "metadatas", "distances"],
            n_results=10
        )
        print("hi")
        print_results(question, context)

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


    # print(chat_completion.choices[0].message.content)
    # print("Hello from ai-chips-report-rag!")

if __name__ == "__main__":
    main()
