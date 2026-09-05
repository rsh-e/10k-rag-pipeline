import os
import re
from pathlib import Path

# retrieval/constants use ../storage relative to src/
os.chdir(Path(__file__).resolve().parent)

import streamlit as st
from dotenv import load_dotenv
from groq import APIStatusError, RateLimitError, Groq

from query import query_model
from prompt import format_context
from retrieval import download_nltk_modules, get_retrieved_chunks

# ASCII [1] and fullwidth 【1】 / ［1］
CITATION_RE = re.compile(r"(?:\[(\d+)\]|【(\d+)】|［(\d+)］)")


def citation_number(match: re.Match) -> int:
    return int(next(group for group in match.groups() if group is not None))


def get_cited_indices_in_order(answer: str, n_chunks: int) -> list[int]:
    cited = []
    for match in CITATION_RE.finditer(answer):
        index = citation_number(match)
        if 1 <= index <= n_chunks and index not in cited:
            cited.append(index)
    return cited


def renumber_citations(answer: str, n_chunks: int) -> tuple[str, list[int]]:
    cited = get_cited_indices_in_order(answer, n_chunks)
    display_map = {
        retrieval_index: display_index
        for display_index, retrieval_index in enumerate(cited, start=1)
    }

    def replace_match(match: re.Match) -> str:
        retrieval_index = citation_number(match)
        if retrieval_index in display_map:
            return f"[{display_map[retrieval_index]}]"
        return match.group(0)

    renumbered = CITATION_RE.sub(replace_match, answer)
    return renumbered, cited


def strip_trailing_sources_block(answer: str) -> str:
    answer = re.sub(
        r"\nSources:?\s*(?:\n\s*(?:\[\d+\]|【\d+】|［\d+］).*)*$",
        "",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return answer.strip()


def format_cited_sources(cited: list[int], indexed: dict) -> str:
    if not cited:
        return ""

    blocks = []
    for display_index, retrieval_index in enumerate(cited, start=1):
        source = indexed[retrieval_index]
        blocks.append(f"**[{display_index}] {source['header']}**\n\n{source['body']}")
    return "\n\n---\n\n".join(blocks)


def groq_error_message(error: APIStatusError) -> str:
    if isinstance(error, RateLimitError) or error.status_code == 429:
        return (
            "Groq rate limit reached. Wait a minute and try again, "
            "or check your usage at console.groq.com."
        )
    if error.status_code == 413:
        return (
            "The retrieved context is too large for the API. "
            "Try a narrower question — for example, one company and one metric."
        )
    return f"Groq API error ({error.status_code}): {error.message}"


@st.cache_resource
def setup():
    download_nltk_modules()
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path, override=True)
    api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not api_key:
        return None
    return Groq(api_key=api_key)


st.set_page_config(page_title="10-K RAG", page_icon="📄", layout="wide")

st.title("10-K Research Assistant")
st.caption(
    "Ask questions about AMD, AXP, COP, CVX, GOOG, JNJ, KO, META, NVDA, and PEP filings."
)
st.caption("Because this is a demo on model with a smaller context window, top K is limited to 5, avoid questions using large tables")
st.caption("You can tell the model to use tables as its source of information by specifying table in the prompt")

client = setup()

if client is None:
    st.error("GROQ_API_KEY is not set. Add it to src/.env")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    show_sources = st.checkbox("Show cited sources", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources") and show_sources:
            with st.expander("Sources"):
                st.markdown(message["sources"])

if prompt := st.chat_input("Ask a question about the 10-K filings..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving..."):
                retrieved_chunks = get_retrieved_chunks(prompt)
                context, indexed = format_context(retrieved_chunks)

            with st.spinner("Generating..."):
                chat_completion = query_model(client, prompt, context)
                raw_answer = chat_completion.choices[0].message.content

            cleaned = strip_trailing_sources_block(raw_answer)
            answer, cited = renumber_citations(cleaned, len(indexed))
            cited_sources = format_cited_sources(cited, indexed)

            st.markdown(answer)

            if show_sources and cited_sources:
                with st.expander("Sources"):
                    st.markdown(cited_sources)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": cited_sources,
                }
            )
        except RateLimitError as e:
            st.warning(groq_error_message(e))
        except APIStatusError as e:
            if e.status_code == 413:
                st.warning(groq_error_message(e))
            else:
                st.error(groq_error_message(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
