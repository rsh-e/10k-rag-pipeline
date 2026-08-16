from groq import Groq
import chromadb

def main():
    chroma_client = chromadb.PersistentClient(path="../storage")
    collection = chroma_client.get_collection(name="my_collection") 

    #retriever (use collection.query)

    question = input("Type in your question: ")
    context = collection.query(
        query_texts=question, 
        include=["documents"],
        n_results=10
        )

    for embedding in  context["documents"]:
        for i in embedding:
            print(i)
            print()

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
