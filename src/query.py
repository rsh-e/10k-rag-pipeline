
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from constants import MAX_COMPLETION_TOKENS, MODEL, ROLE, TEMPERATURE, TOP_P
from prompt import TEMPLATE
from retrieval import download_nltk_modules, get_retrieved_chunks


def construct_query(question, retrieved):
    query = TEMPLATE + f"""
    The question provided to you is {question} \
    The context provided to you is {retrieved}                 
    """

    return query

def query_model(client, question, retrieved):
    query = construct_query(question, retrieved)
    chat_completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": ROLE, "content": query}],
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        top_p=TOP_P,
        stream=False,
        stop=None,
    )

    return chat_completion

def query_workflow():
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path, override=True)
    api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit(
            f"GROQ_API_KEY is not set. Add it to {env_path} (see console.groq.com/keys)."
        )
    download_nltk_modules()

    client = Groq(api_key=api_key)
    while True:
        question = input("Ask your question: ")
        retrieved_chunks = get_retrieved_chunks(question)
        chat_completion = query_model(client, question, retrieved_chunks)
        answer = chat_completion.choices[0].message.content
        print(answer)
        print(chat_completion.usage)
        print(chat_completion.choices[0].finish_reason)  
        print("//" * 100)

if __name__ == "__main__":
    query_workflow()
