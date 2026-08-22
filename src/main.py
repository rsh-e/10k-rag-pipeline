from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

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

    questions = [
        "What countries does chevron operate in",
        "What risks does nvidia face",
        "Who are the board of directors in amd",
        "How does coca-cola distribution work",
        "What does conoco philips do"
    ]

    for question in questions:
        query_embedding = model.encode("search_query: " + question)

        context = collection.query(
            query_embeddings=[query_embedding],
            include=["documents", "metadatas", "distances"],
            n_results=10
        )

        print_results(question, context)

    # Calling Groq
    template = f"""SYSTEM: You are a chatbot that specialises in providing information about AI Chip Companies. 
    Information can include financial information, management, risks etc. All your responses must be factual. If you 
    don't know the answer, say you don't know.
    Respond to the following question: {question} only from the below context: {context}
    """


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
