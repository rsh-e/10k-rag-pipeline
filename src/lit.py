import os
import re
import tarfile
import urllib.request
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
STORAGE_DIR = REPO_ROOT / "storage"
DOCUMENT_DB = STORAGE_DIR / "document_ledger.db"

# After you upload storage-v1.tar.gz to a GitHub Release, set this URL.
# Example: https://github.com/rsh-e/10k-rag-pipeline/releases/download/v1.0.0/storage-v1.tar.gz
DEFAULT_INDEX_URL = "https://github.com/rsh-e/10k-rag-pipeline/releases/download/Storage/storage-v1.tar.gz"

# retrieval/constants use ../storage relative to src/
os.chdir(SRC_DIR)

import streamlit as st
from dotenv import load_dotenv
from groq import APIStatusError, RateLimitError, Groq

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


def missing_storage_parts() -> list[str]:
    missing = []
    if not STORAGE_DIR.exists():
        missing.append("`storage/` folder")
        return missing
    if not DOCUMENT_DB.exists():
        missing.append("`storage/document_ledger.db`")
    if not any(STORAGE_DIR.iterdir()):
        missing.append("files inside `storage/`")
    return missing


def storage_setup_message(missing: list[str], detail: str = "") -> str:
    lines = [
        "**Search index not found.** The app needs the pre-built `storage/` folder "
        "(Chroma collection `data_store` + SQLite ledger).",
        "",
        "For local dev: run the ingest pipeline, or download `storage-v1.tar.gz` from GitHub Releases.",
        "",
    ]
    if missing:
        lines.append("Missing: " + ", ".join(missing) + ".")
    if detail:
        lines.append(f"Detail: {detail}")
    return "\n".join(lines)


def index_download_url() -> str:
    try:
        return st.secrets["STORAGE_INDEX_URL"]
    except Exception:
        return DEFAULT_INDEX_URL


def download_storage_index() -> None:
    url = index_download_url()
    archive_path = REPO_ROOT / "storage-v1.tar.gz"

    with st.spinner("Downloading search index (first run only, ~100MB)..."):
        urllib.request.urlretrieve(url, archive_path)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=REPO_ROOT)

    archive_path.unlink(missing_ok=True)


def ensure_storage_index() -> None:
    if not missing_storage_parts():
        return
    try:
        download_storage_index()
    except Exception as e:
        raise RuntimeError(
            f"Could not download index from {index_download_url()}. "
            f"Upload storage-v1.tar.gz to GitHub Releases first. ({e})"
        ) from e
    if missing_storage_parts():
        raise RuntimeError(
            "Download finished but storage/ is still incomplete. "
            "Check that the archive contains a top-level storage/ folder."
        )


@st.cache_resource
def load_pipeline():
    try:
        ensure_storage_index()
    except RuntimeError as e:
        return {"ok": False, "message": storage_setup_message(missing=missing_storage_parts(), detail=str(e))}

    missing = missing_storage_parts()
    if missing:
        return {"ok": False, "message": storage_setup_message(missing)}

    try:
        from query import query_model
        from prompt import format_context
        from retrieval import download_nltk_modules, get_retrieved_chunks

        download_nltk_modules()
        return {
            "ok": True,
            "query_model": query_model,
            "format_context": format_context,
            "get_retrieved_chunks": get_retrieved_chunks,
        }
    except Exception as e:
        return {
            "ok": False,
            "message": storage_setup_message(missing=[], detail=str(e)),
        }


@st.cache_resource
def setup():
    env_path = SRC_DIR / ".env"
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
st.caption(
    "Because this is a demo on model with a smaller context window, top K is limited to 5, "
    "avoid questions using large tables"
)
st.caption(
    "You can tell the model to use tables as its source of information by specifying table in the prompt"
)

pipeline = load_pipeline()
if not pipeline["ok"]:
    st.error(pipeline["message"])
    st.stop()

query_model = pipeline["query_model"]
format_context = pipeline["format_context"]
get_retrieved_chunks = pipeline["get_retrieved_chunks"]

client = setup()

if client is None:
    st.error(
        "GROQ_API_KEY is not set. Add it in Streamlit Cloud **Secrets** "
        "(or in `src/.env` locally)."
    )
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
