"""Filings AI — Streamlit front end for the 10-K retrieval pipeline."""

from __future__ import annotations

import json
import os
import re
import tarfile
import urllib.request
from dataclasses import dataclass
from html import escape
from pathlib import Path
from string import Template

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
STORAGE_DIR = REPO_ROOT / "storage"
DOCUMENT_DB = STORAGE_DIR / "document_ledger.db"

DEFAULT_INDEX_URL = (
    "https://github.com/rsh-e/10k-rag-pipeline/releases/download/Storage/storage-v1.tar.gz"
)

# retrieval.py resolves storage paths relative to src/
os.chdir(SRC_DIR)

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import APIStatusError, Groq, RateLimitError
from streamlit import config as st_config

from constants import COMPANIES, MAX_COMPLETION_TOKENS, MODEL, TOP_K

CITATION_RE = re.compile(r"(?:\[(\d+)\]|【(\d+)】|［(\d+)］)")

COMPANY_LABELS = {
    "amd": "AMD",
    "axp": "American Express",
    "cop": "ConocoPhillips",
    "cvx": "Chevron",
    "goog": "Alphabet",
    "jnj": "Johnson & Johnson",
    "ko": "Coca-Cola",
    "meta": "Meta",
    "nvda": "NVIDIA",
    "pep": "PepsiCo",
}

EXAMPLE_QUESTIONS = [
    ("Revenue lookup", "What was NVIDIA's total revenue in fiscal 2025?"),
    ("Comparison", "Compare Coca-Cola and PepsiCo net revenue."),
    ("Risk factors", "What risk factors does AMD highlight in Item 1A?"),
    ("Segments", "How did Meta's Reality Labs segment perform?"),
]

CONFIG_LABELS = {
    "embed only": "Embeddings",
    "rrf only": "Hybrid (RRF)",
    "encoder only": "Cross-encoder",
    "rrf + company + table routing": "Full pipeline",
}
CONFIG_ORDER = list(CONFIG_LABELS)

# What each configuration adds on top of the one before it.
STAGE_LABELS = {
    "rrf only": "adding lexical search",
    "encoder only": "cross-encoder reranking",
    "rrf + company + table routing": "company and table routing",
}

SLICE_LABELS = {"type": "Question type", "hop": "Reasoning hops", "difficulty": "Difficulty"}

SOURCES_PER_ROW = 3


# ──────────────────────────────────────────────────────────────────────────────
# Themes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Theme:
    """A complete look: Streamlit's own widget palette plus the stylesheet tokens."""

    label: str
    blurb: str
    font_import: str
    body_font: str
    heading_font: str
    base_font_size: int
    accent: str
    accent_soft: str
    bg: str
    surface: str
    surface_2: str
    border: str
    border_strong: str
    text: str
    text_2: str
    muted: str
    positive: str
    negative: str
    radius: str
    st_radius: str
    label_case: str
    label_tracking: str
    title_tracking: str
    title_weight: str
    chart_colors: list[str]
    extra_css: str = ""


PLEX = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');"
)
CMU = (
    "@import url('https://cdn.jsdelivr.net/gh/aaaakshat/cm-web-fonts@latest/font/Serif/cmun-serif.css');"
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');"
)
INTER = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600&display=swap');"
)

THEMES = {
    "Carbon": Theme(
        label="Carbon",
        blurb="IBM Plex, square edges, hairline rules.",
        font_import=PLEX,
        body_font="'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif",
        heading_font="'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif",
        base_font_size=18,
        accent="#0f62fe",
        accent_soft="#d0e2ff",
        bg="#f4f4f4",
        surface="#ffffff",
        surface_2="#f7f7f7",
        border="#e0e0e0",
        border_strong="#c6c6c6",
        text="#161616",
        text_2="#393939",
        muted="#6f6f6f",
        positive="#0e6027",
        negative="#da1e28",
        radius="0px",
        st_radius="none",
        label_case="uppercase",
        label_tracking="0.06em",
        title_tracking="-0.01em",
        title_weight="600",
        chart_colors=["#0f62fe", "#009d9a", "#005d5d", "#b28600", "#9f1853"],
        extra_css="""
.answer-eyebrow, .sources-bar, .side-label, .kpi-label { font-family: 'IBM Plex Mono', monospace; }
.chip, .spec span:last-child { font-family: 'IBM Plex Mono', monospace; }
[class*="st-key-answer-"] { border-left: 2px solid var(--accent); }
""",
    ),
    "Journal": Theme(
        label="Journal",
        blurb="Computer Modern on paper, LaTeX-style.",
        font_import=CMU,
        body_font="'Computer Modern Serif', 'Source Serif 4', Georgia, serif",
        heading_font="'Computer Modern Serif', 'Source Serif 4', Georgia, serif",
        base_font_size=19,
        accent="#8b2635",
        accent_soft="#f2e4e4",
        bg="#faf9f5",
        surface="#ffffff",
        surface_2="#f6f4ed",
        border="#e2ddd0",
        border_strong="#cbc4b2",
        text="#1a1a1a",
        text_2="#3c3934",
        muted="#6f6a5f",
        positive="#3f5c3a",
        negative="#8b2635",
        radius="2px",
        st_radius="none",
        label_case="none",
        label_tracking="0.02em",
        title_tracking="0",
        title_weight="600",
        chart_colors=["#8b2635", "#1f3a5f", "#5c6b3f", "#8a6d3b", "#4a4a4a"],
        extra_css="""
.answer-eyebrow, .sources-bar, .side-label, .kpi-label, .hero-kicker {
    font-variant-caps: small-caps; font-size: 0.92rem;
}
.hero-title, .page-title { font-style: normal; }
[class*="st-key-answer-"] [data-testid="stMarkdownContainer"] p { line-height: 1.7; }
.user-bubble { font-style: italic; }
""",
    ),
    "Slate": Theme(
        label="Slate",
        blurb="Monochrome sans, minimal colour.",
        font_import=INTER,
        body_font="'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
        heading_font="'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
        base_font_size=18,
        accent="#1c1f24",
        accent_soft="#ececee",
        bg="#fafafa",
        surface="#ffffff",
        surface_2="#f6f6f6",
        border="#e5e5e6",
        border_strong="#d0d0d2",
        text="#18181b",
        text_2="#3f3f46",
        muted="#71717a",
        positive="#15803d",
        negative="#b91c1c",
        radius="4px",
        st_radius="small",
        label_case="uppercase",
        label_tracking="0.07em",
        title_tracking="-0.02em",
        title_weight="600",
        chart_colors=["#1c1f24", "#5c6672", "#8d949e", "#b9bec5", "#dcdfe3"],
    ),
}

DEFAULT_THEME = "Carbon"


def active_theme() -> Theme:
    return THEMES.get(st.session_state.get("theme", DEFAULT_THEME), THEMES[DEFAULT_THEME])


def choose_theme() -> None:
    """Copy the picker into a plain state key.

    Applying a theme reruns before the sidebar is drawn, and Streamlit discards
    widget state for widgets a run never reached — so the choice cannot live in
    the widget's own key.
    """
    st.session_state.theme = st.session_state.theme_picker


def theme_config(theme: Theme) -> dict:
    """Streamlit renders dataframes and charts itself, so its palette has to be set
    through config — CSS cannot reach inside those widgets."""
    return {
        "base": "light",
        "primaryColor": theme.accent,
        "backgroundColor": theme.bg,
        "secondaryBackgroundColor": theme.surface,
        "textColor": theme.text,
        "linkColor": theme.accent,
        "borderColor": theme.border,
        "dataframeBorderColor": theme.border,
        "dataframeHeaderBackgroundColor": theme.surface_2,
        "codeBackgroundColor": theme.surface_2,
        "font": theme.body_font,
        "headingFont": theme.heading_font,
        "baseFontSize": theme.base_font_size,
        "baseRadius": theme.st_radius,
        "showWidgetBorder": True,
        "showSidebarBorder": True,
        "chartCategoricalColors": theme.chart_colors,
    }


def apply_theme(name: str) -> None:
    """Pin the Streamlit theme so native widgets match the stylesheet."""
    options = theme_config(THEMES[name])
    stale = any(st_config.get_option(f"theme.{k}") != v for k, v in options.items())
    for key, value in options.items():
        st_config.set_option(f"theme.{key}", value)

    # The theme travels on the message that starts a script run, so the run that
    # changes it is already painting with the previous one.
    if stale and st.session_state.get("theme_applied") != name:
        st.session_state.theme_applied = name
        st.rerun()


CSS_TEMPLATE = """
<style>
$font_import

:root {
    --bg: $bg;
    --surface: $surface;
    --surface-2: $surface_2;
    --border: $border;
    --border-strong: $border_strong;
    --text: $text;
    --text-2: $text_2;
    --muted: $muted;
    --accent: $accent;
    --accent-soft: $accent_soft;
    --positive: $positive;
    --negative: $negative;
    --radius: $radius;
    --body-font: $body_font;
    --heading-font: $heading_font;
    --label-case: $label_case;
    --label-tracking: $label_tracking;
}

html, body, .stApp, [class*="css"] {
    font-family: var(--body-font);
    color: var(--text);
}

.stApp { background: var(--bg); }

h1, h2, h3, h4, .hero-title, .page-title, .brand-name { font-family: var(--heading-font); }

[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    min-height: 3.25rem;
    visibility: visible !important;
}

header a, header [data-testid="stNavLink"],
[data-testid="stHeader"] a {
    font-size: 1.02rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
}

[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none; }
footer, #MainMenu { visibility: hidden; height: 0; }

[class*="st-key-fixed-eval"],
[class*="st-key-fixed-chat"] {
    position: fixed !important;
    top: 0.7rem;
    right: 1.4rem;
    z-index: 1000000;
    width: auto !important;
}

[class*="st-key-fixed-eval"] button,
[class*="st-key-fixed-chat"] button {
    background: var(--text) !important;
    color: #ffffff !important;
    border: none !important;
    min-height: 2.25rem !important;
    padding: 0 1rem !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* One page width for both views so the Chat / Evaluation control does not jump. */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    scroll-behavior: auto !important;
    overflow-anchor: none !important;
}

[data-testid="stMainBlockContainer"], .stMain .block-container {
    padding: 3.75rem 2rem 2.75rem 2rem;
    max-width: 78rem;
}

[class*="st-key-chat-pane"] {
    max-width: 50rem;
    margin: 0 auto;
}


/* ── Sidebar ─────────────────────────────────────────────────────────────── */

[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }

[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 1.5rem; }

.brand { border-bottom: 1px solid var(--border); padding-bottom: 0.9rem; margin-bottom: 1.1rem; }

.brand-name {
    font-size: 1.2rem; font-weight: 600; letter-spacing: $title_tracking;
    line-height: 1.2; color: var(--text);
}

.brand-sub { font-size: 0.88rem; color: var(--muted); line-height: 1.35; margin-top: 0.15rem; }

.side-label {
    font-size: 0.8rem; font-weight: 600; letter-spacing: var(--label-tracking);
    text-transform: var(--label-case); color: var(--muted);
    margin: 1.3rem 0 0.5rem 0;
}

.chip-wrap { display: flex; flex-wrap: wrap; gap: 0.25rem; }

.chip {
    font-size: 0.8rem; color: var(--text-2);
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 0.16rem 0.45rem; white-space: nowrap;
}

.spec { display: flex; justify-content: space-between; gap: 0.75rem; padding: 0.22rem 0; font-size: 0.88rem; }
.spec span:first-child { color: var(--muted); }
.spec span:last-child { color: var(--text-2); font-variant-numeric: tabular-nums; }

.side-note { font-size: 0.84rem; color: var(--muted); line-height: 1.4; margin: 0.4rem 0 0 0; }

.nav-label { margin-top: 0; }

[data-testid="stSidebar"] [data-testid="stSegmentedControl"] > div { width: 100%; }
[data-testid="stSidebar"] [data-testid="stSegmentedControl"] button { flex: 1; }

[class*="st-key-nav"] button { min-height: 2.55rem; }
[class*="st-key-nav"] button p { font-weight: 500; font-size: 0.95rem !important; }

/* ── Landing ─────────────────────────────────────────────────────────────── */

.hero { padding: 2.25rem 0 1.4rem 0; }

.hero-kicker {
    font-size: 0.84rem; font-weight: 600; letter-spacing: var(--label-tracking);
    text-transform: var(--label-case); color: var(--muted); margin: 0 0 0.7rem 0;
}

.hero-title {
    font-size: clamp(2.05rem, 3.8vw, 2.7rem); font-weight: $title_weight;
    letter-spacing: $title_tracking; line-height: 1.15; margin: 0 0 0.6rem 0;
}

.hero-lead { font-size: 1.12rem; line-height: 1.6; color: var(--text-2); max-width: 36rem; margin: 0; }

[class*="st-key-example-"] button {
    height: auto !important; min-height: 3.8rem;
    padding: 0.75rem 0.95rem !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    text-align: left !important; justify-content: flex-start !important;
    transition: border-color 0.12s ease, background 0.12s ease;
}

[class*="st-key-example-"] button p {
    font-size: 0.98rem !important; line-height: 1.45 !important;
    color: var(--text-2) !important; text-align: left !important; white-space: normal !important;
}

[class*="st-key-example-"] button:hover {
    border-color: var(--accent) !important; background: var(--surface-2) !important;
}

/* ── Conversation ────────────────────────────────────────────────────────── */

.turn-gap { height: 1.4rem; }

.user-bubble {
    width: fit-content; max-width: 84%; margin-left: auto;
    background: var(--accent); color: #ffffff;
    font-size: 1.05rem; line-height: 1.5;
    padding: 0.55rem 0.8rem; border-radius: var(--radius);
}

[class*="st-key-answer-"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 0.95rem 1.1rem 0.85rem 1.1rem;
}

.answer-eyebrow {
    font-size: 0.8rem; font-weight: 600; letter-spacing: var(--label-tracking);
    text-transform: var(--label-case); color: var(--muted); margin-bottom: 0.45rem;
}

[class*="st-key-answer-"] [data-testid="stMarkdownContainer"] p,
[class*="st-key-answer-"] [data-testid="stMarkdownContainer"] li {
    font-size: 1.05rem; line-height: 1.65; color: var(--text);
}

[class*="st-key-answer-"] table { font-size: 0.94rem; }

.cite-chip {
    display: inline-block; min-width: 1.1rem; text-align: center;
    font-size: 0.78rem; font-weight: 600; line-height: 1.1rem;
    color: var(--accent); background: var(--accent-soft);
    border-radius: var(--radius); padding: 0 0.18rem; margin: 0 0.06rem;
    vertical-align: 0.1rem;
}

.sources-bar {
    border-top: 1px solid var(--border);
    margin: 0.8rem -1.1rem 0.55rem -1.1rem; padding: 0.65rem 1.1rem 0 1.1rem;
    font-size: 0.8rem; font-weight: 600; letter-spacing: var(--label-tracking);
    text-transform: var(--label-case); color: var(--muted);
}

[class*="st-key-source-"] button {
    width: 100%; min-height: 2.3rem; padding: 0.32rem 0.55rem !important;
    border: 1px solid var(--border) !important;
    background: var(--surface-2) !important;
    transition: border-color 0.12s ease, background 0.12s ease;
}

[class*="st-key-source-"] button p {
    font-size: 0.86rem !important; color: var(--text-2) !important;
    white-space: nowrap !important; overflow: hidden; text-overflow: ellipsis;
}

[class*="st-key-source-"] button:hover {
    border-color: var(--accent) !important; background: var(--accent-soft) !important;
}

.thinking {
    font-size: 1.02rem; color: var(--muted);
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 0.8rem 1.1rem;
}

.thinking::after {
    content: ''; display: inline-block; width: 0.4rem; height: 0.4rem;
    margin-left: 0.4rem; background: var(--accent);
    animation: blink 1.1s steps(2, start) infinite;
}

@keyframes blink { 0%, 100% { opacity: 0.2; } 50% { opacity: 1; } }

/* ── Source dialog ───────────────────────────────────────────────────────── */

.dialog-meta {
    font-size: 0.92rem; color: var(--text-2);
    background: var(--surface-2); border-left: 2px solid var(--accent);
    padding: 0.5rem 0.7rem; margin-bottom: 0.7rem; word-break: break-word;
}

[class*="st-key-sourcebody-"] {
    border: 1px solid var(--border); border-radius: var(--radius);
    background: var(--surface); padding: 0.8rem 0.95rem;
    max-height: min(24rem, 45vh); overflow-y: auto; overscroll-behavior: contain;
}

[class*="st-key-sourcebody-"] [data-testid="stMarkdownContainer"] p,
[class*="st-key-sourcebody-"] [data-testid="stMarkdownContainer"] li {
    font-size: 0.98rem; line-height: 1.65; color: var(--text-2);
}

[class*="st-key-sourcebody-"] table { font-size: 0.88rem; }

/* ── Evaluation ──────────────────────────────────────────────────────────── */

.page-head { border-bottom: 1px solid var(--border); padding-bottom: 0.9rem; margin-bottom: 1.3rem; }

.page-title { font-size: 1.75rem; font-weight: $title_weight; letter-spacing: $title_tracking; margin: 0 0 0.25rem 0; }

.page-sub { font-size: 1rem; color: var(--muted); margin: 0; }

.kpi-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(11.5rem, 1fr));
    gap: 0.7rem; margin-bottom: 1.4rem;
}

.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 0.85rem 0.95rem; }

.kpi-label {
    font-size: 0.8rem; font-weight: 600; letter-spacing: var(--label-tracking);
    text-transform: var(--label-case); color: var(--muted); margin-bottom: 0.35rem;
}

.kpi-value { font-size: 1.85rem; font-weight: $title_weight; letter-spacing: $title_tracking; line-height: 1; font-variant-numeric: tabular-nums; }

.kpi-delta { font-size: 0.84rem; margin-top: 0.35rem; color: var(--muted); }
.kpi-delta.up { color: var(--positive); }
.kpi-delta.down { color: var(--negative); }

.summary { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 0.95rem 1.1rem; margin-bottom: 1.4rem; }

.summary-head {
    font-size: 0.8rem; font-weight: 600; letter-spacing: var(--label-tracking);
    text-transform: var(--label-case); color: var(--muted); margin-bottom: 0.6rem;
}

.summary ul { margin: 0; padding-left: 1.05rem; }

.summary li { font-size: 1.02rem; line-height: 1.65; color: var(--text-2); margin-bottom: 0.3rem; }

.summary li:last-child { margin-bottom: 0; }

.summary b { color: var(--text); font-variant-numeric: tabular-nums; }

.section-title { font-size: 1.1rem; font-weight: 600; letter-spacing: $title_tracking; margin: 0 0 0.15rem 0; }

.section-note { font-size: 0.92rem; color: var(--muted); margin: 0 0 0.55rem 0; }

/* ── Shared widgets ──────────────────────────────────────────────────────── */

[data-testid="stSegmentedControl"] button p { font-size: 0.95rem !important; }

[data-testid="stChatInput"] > div {
    background: var(--surface) !important;
    border: 2px solid var(--accent) !important;
    border-radius: var(--radius) !important;
    padding: 0.28rem 0.3rem 0.28rem 0.4rem !important;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--text) !important;
    box-shadow: 0 0 0 1px var(--text) !important;
}

.stChatInput textarea { font-size: 1.05rem !important; }

[data-testid="stChatInputSubmitButton"] button {
    background: var(--accent) !important; color: #ffffff !important; border: none !important;
}

[data-testid="stChatInputSubmitButton"] button:disabled { background: var(--border-strong) !important; }

[data-testid="stBottomBlockContainer"] {
    max-width: 50rem; margin: 0 auto;
    padding: 0 2rem 1.15rem 2rem;
    background: transparent !important;
}

[data-testid="stBottom"] > div { background: transparent !important; }

$extra_css
</style>
"""


def stylesheet(theme: Theme) -> str:
    return Template(CSS_TEMPLATE).substitute(
        font_import=theme.font_import,
        bg=theme.bg,
        surface=theme.surface,
        surface_2=theme.surface_2,
        border=theme.border,
        border_strong=theme.border_strong,
        text=theme.text,
        text_2=theme.text_2,
        muted=theme.muted,
        accent=theme.accent,
        accent_soft=theme.accent_soft,
        positive=theme.positive,
        negative=theme.negative,
        radius=theme.radius,
        body_font=theme.body_font,
        heading_font=theme.heading_font,
        label_case=theme.label_case,
        label_tracking=theme.label_tracking,
        title_tracking=theme.title_tracking,
        title_weight=theme.title_weight,
        extra_css=theme.extra_css,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Text helpers
# ──────────────────────────────────────────────────────────────────────────────


def citation_number(match: re.Match) -> int:
    return int(next(group for group in match.groups() if group is not None))


def get_cited_indices_in_order(answer: str, n_chunks: int) -> list[int]:
    cited: list[int] = []
    for match in CITATION_RE.finditer(answer):
        index = citation_number(match)
        if 1 <= index <= n_chunks and index not in cited:
            cited.append(index)
    return cited


def renumber_citations(answer: str, n_chunks: int) -> tuple[str, list[int]]:
    """Renumber retrieval indices to 1..n in order of first appearance."""
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

    return CITATION_RE.sub(replace_match, answer), cited


def strip_trailing_sources_block(answer: str) -> str:
    answer = re.sub(
        r"\nSources:?\s*(?:\n\s*(?:\[\d+\]|【\d+】|［\d+］).*)*$",
        "",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return answer.strip()


def sanitize_markdown(text: str) -> str:
    """Escape $ so Streamlit does not read dollar amounts as LaTeX math."""
    return text.replace("$", r"\$") if text else text


def render_text(text: str) -> None:
    st.markdown(sanitize_markdown(text))


def render_answer(text: str) -> None:
    """Render an answer with [n] markers styled as inline citation chips."""
    # Neutralise tags in the model output first: the only HTML that reaches the
    # renderer should be the citation chips added below.
    body = sanitize_markdown(text).replace("<", "&lt;").replace(">", "&gt;")
    body = CITATION_RE.sub(
        lambda m: f'<span class="cite-chip">{citation_number(m)}</span>', body
    )
    st.markdown(body, unsafe_allow_html=True)


def short_source_label(header: str, limit: int = 26) -> str:
    parts = [part.strip() for part in header.split("·") if part.strip()]
    label = " · ".join(parts[:2]) if len(parts) >= 2 else header
    return label if len(label) <= limit else label[: limit - 1] + "…"


def build_citation_records(cited: list[int], indexed: dict) -> list[dict]:
    records = []
    for display_index, retrieval_index in enumerate(cited, start=1):
        source = indexed[retrieval_index]
        records.append(
            {
                "index": display_index,
                "header": source["header"],
                "short": short_source_label(source["header"]),
                "body": source["body"],
            }
        )
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Backend wiring
# ──────────────────────────────────────────────────────────────────────────────


def groq_error_message(error: APIStatusError) -> str:
    if isinstance(error, RateLimitError) or error.status_code == 429:
        return (
            "Groq rate limit reached. Wait a minute and try again, "
            "or check your usage at console.groq.com."
        )
    if error.status_code == 413:
        return (
            "The retrieved context is too large for the API. "
            "Try a narrower question — one company, one metric, no large tables."
        )
    return f"Groq API error ({error.status_code}): {error.message}"


def missing_storage_parts() -> list[str]:
    if not STORAGE_DIR.exists():
        return ["`storage/` folder"]
    missing = []
    if not DOCUMENT_DB.exists():
        missing.append("`storage/document_ledger.db`")
    if not any(STORAGE_DIR.iterdir()):
        missing.append("files inside `storage/`")
    return missing


def storage_setup_message(missing: list[str], detail: str = "") -> str:
    lines = [
        (
            "**Search index not found.** The app needs the pre-built `storage/` folder "
            "(Chroma collection `data_store` + SQLite ledger)."
        ),
        "",
        (
            "For local dev: run the ingest pipeline, or download `storage-v1.tar.gz` "
            "from GitHub Releases."
        ),
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
    archive_path = REPO_ROOT / "storage-v1.tar.gz"
    urllib.request.urlretrieve(index_download_url(), archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=REPO_ROOT)
    archive_path.unlink(missing_ok=True)


@st.cache_resource
def ensure_storage_ready() -> dict:
    if not missing_storage_parts():
        return {"ok": True, "downloaded": False}

    try:
        download_storage_index()
    except Exception as e:
        return {
            "ok": False,
            "message": storage_setup_message(
                missing=missing_storage_parts(),
                detail=f"Could not download index from {index_download_url()}. ({e})",
            ),
        }

    if missing_storage_parts():
        return {
            "ok": False,
            "message": storage_setup_message(
                missing=missing_storage_parts(),
                detail="Download finished but storage/ is still incomplete.",
            ),
        }

    return {"ok": True, "downloaded": True}


@st.cache_resource
def load_retrieval_pipeline() -> dict:
    try:
        from prompt import format_context
        from query import query_model
        from retrieval import download_nltk_modules, get_retrieved_chunks

        download_nltk_modules()
        return {
            "ok": True,
            "query_model": query_model,
            "format_context": format_context,
            "get_retrieved_chunks": get_retrieved_chunks,
        }
    except Exception as e:
        return {"ok": False, "message": storage_setup_message(missing=[], detail=str(e))}


@st.cache_resource
def load_groq_client():
    load_dotenv(SRC_DIR / ".env", override=True)
    api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    return Groq(api_key=api_key) if api_key else None


def bootstrap_app() -> dict:
    storage = ensure_storage_ready()
    if not storage["ok"]:
        return {"ok": False, "message": storage["message"]}

    pipeline = load_retrieval_pipeline()
    if not pipeline["ok"]:
        return {"ok": False, "message": pipeline["message"]}

    client = load_groq_client()
    if client is None:
        return {
            "ok": False,
            "message": (
                "GROQ_API_KEY is not set. Add it in Streamlit Cloud **Secrets** "
                "(or in `src/.env` locally)."
            ),
        }

    return {
        "ok": True,
        "query_model": pipeline["query_model"],
        "format_context": pipeline["format_context"],
        "get_retrieved_chunks": pipeline["get_retrieved_chunks"],
        "client": client,
    }


def generate_response(prompt: str, app: dict) -> tuple[str, list[dict]]:
    retrieved_chunks = app["get_retrieved_chunks"](prompt)
    context, indexed = app["format_context"](retrieved_chunks)
    chat_completion = app["query_model"](app["client"], prompt, context)
    raw_answer = chat_completion.choices[0].message.content

    cleaned = strip_trailing_sources_block(raw_answer)
    answer, cited = renumber_citations(cleaned, len(indexed))
    return answer, build_citation_records(cited, indexed)


# ──────────────────────────────────────────────────────────────────────────────
# Sources
# ──────────────────────────────────────────────────────────────────────────────


@st.dialog("Cited sources", width="large")
def source_dialog(citations: list[dict], active: int, token: int) -> None:
    """Full chunk text in a modal with its own scroll area."""
    numbers = [citation["index"] for citation in citations]
    selected = active
    if len(numbers) > 1:
        selected = st.segmented_control(
            "Source",
            numbers,
            default=active,
            format_func=lambda n: f"[{n}]",
            label_visibility="collapsed",
            key=f"sourcepicker-{token}",
        )
        selected = selected if selected is not None else active

    citation = next(c for c in citations if c["index"] == selected)
    st.markdown(
        f'<div class="dialog-meta">{escape(citation["header"])}</div>',
        unsafe_allow_html=True,
    )
    # Capped in CSS rather than fixed-height, so short passages stay compact and
    # long ones scroll inside the dialog instead of stretching the page.
    with st.container(border=False, key=f"sourcebody-{token}-{selected}"):
        render_text(citation["body"])


def render_sources(citations: list[dict], message_index: int) -> None:
    if not citations:
        return

    st.markdown(
        f'<div class="sources-bar">Sources · {len(citations)}</div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(SOURCES_PER_ROW, gap="small")
    for position, citation in enumerate(citations):
        column = columns[position % SOURCES_PER_ROW]
        clicked = column.button(
            f"{citation['index']} · {citation['short']}",
            key=f"source-{message_index}-{citation['index']}",
            width="stretch",
            help=citation["header"],
        )
        if clicked:
            st.session_state.dialog_token += 1
            source_dialog(citations, citation["index"], st.session_state.dialog_token)


# ──────────────────────────────────────────────────────────────────────────────
# Chat view
# ──────────────────────────────────────────────────────────────────────────────


def render_landing() -> None:
    st.markdown(
        f"""
        <div class="hero">
          <p class="hero-kicker">SEC 10-K corpus · {len(COMPANIES)} issuers</p>
          <p class="hero-title">Ask questions about annual reports.</p>
          <p class="hero-lead">
            Every answer is drawn from retrieved passages and cites the filing it came from.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(2, gap="small")
    for index, (tag, question) in enumerate(EXAMPLE_QUESTIONS):
        if columns[index % 2].button(
            f"**{tag}**\n\n{question}",
            key=f"example-{index}",
            width="stretch",
        ):
            st.session_state.pending_prompt = question
            st.rerun()


def render_messages(messages: list[dict]) -> None:
    for index, message in enumerate(messages):
        if index:
            st.markdown('<div class="turn-gap"></div>', unsafe_allow_html=True)

        if message["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">{escape(message["content"])}</div>',
                unsafe_allow_html=True,
            )
            continue

        with st.container(key=f"answer-{index}"):
            st.markdown('<div class="answer-eyebrow">Answer</div>', unsafe_allow_html=True)
            render_answer(message["content"])
            render_sources(message.get("citations", []), index)


def render_chat(app: dict) -> None:
    messages = st.session_state.messages

    # The landing and conversation layouts emit different numbers of elements.
    # Wrapping them keeps the top-level count fixed, so a long answer run cannot
    # leave the previous run's trailing elements on screen beside the new ones.
    with st.container(key="chat-pane"):
        if not messages:
            render_landing()
        else:
            render_messages(messages)
        pending = st.empty()

    prompt = st.chat_input("Ask about revenue, risk factors, segments…")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    if not messages or messages[-1]["role"] != "user":
        return

    pending.markdown(
        '<div class="turn-gap"></div>'
        '<div class="thinking">Searching filings and drafting an answer…</div>',
        unsafe_allow_html=True,
    )

    try:
        answer, citations = generate_response(messages[-1]["content"], app)
    except APIStatusError as e:
        messages.pop()
        message = groq_error_message(e)
        if isinstance(e, RateLimitError) or e.status_code in (413, 429):
            pending.warning(message)
        else:
            pending.error(message)
        return
    except Exception as e:
        messages.pop()
        pending.error(f"Something went wrong: {e}")
        return

    messages.append({"role": "assistant", "content": answer, "citations": citations})
    st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation view
# ──────────────────────────────────────────────────────────────────────────────


@st.cache_data
def load_eval_results() -> dict | None:
    path = SRC_DIR / "eval_results.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def config_label(config: str) -> str:
    return CONFIG_LABELS.get(config, config)


def ordered_configs(configs: set[str]) -> list[str]:
    known = [config for config in CONFIG_ORDER if config in configs]
    return known + sorted(configs - set(CONFIG_ORDER))


def metric_columns(labels: dict[str, str]) -> dict:
    return {
        name: st.column_config.ProgressColumn(
            label, min_value=0, max_value=100, format="%.1f%%"
        )
        for name, label in labels.items()
    }


def render_kpis(summary: pd.DataFrame, n_questions: int, n_held_out: int) -> None:
    best = summary.iloc[-1]
    baseline = summary.iloc[0]

    cards = [
        ("Recall", best["recall"], best["recall"] - baseline["recall"]),
        ("MRR", best["mrr"], best["mrr"] - baseline["mrr"]),
        ("Completeness", best["completeness"], best["completeness"] - baseline["completeness"]),
    ]

    html = ['<div class="kpi-grid">']
    for label, value, delta in cards:
        direction = "up" if delta > 0.05 else "down" if delta < -0.05 else ""
        sign = "+" if delta >= 0 else "−"
        html.append(
            f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value:.1f}%</div>'
            f'<div class="kpi-delta {direction}">{sign}{abs(delta):.1f} pts vs '
            f'{escape(config_label(baseline["config"]).lower())}</div></div>'
        )
    html.append(
        f'<div class="kpi"><div class="kpi-label">Questions</div>'
        f'<div class="kpi-value">{n_questions}</div>'
        f'<div class="kpi-delta">{n_held_out} unanswerable held out</div></div>'
    )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def summary_points(frame: pd.DataFrame, summary: pd.DataFrame, top_k: str) -> list[str]:
    """Read the headline story out of the numbers rather than restating them."""
    best = summary.iloc[-1]
    baseline = summary.iloc[0]
    points = [
        (
            f"<b>{config_label(best['config'])}</b> retrieves <b>{best['recall']:.1f}%</b> of the "
            f"passages a question needs at top {top_k}, against "
            f"<b>{baseline['recall']:.1f}%</b> for {config_label(baseline['config']).lower()} alone."
        ),
    ]

    steps = [
        (summary.iloc[i]["config"], summary.iloc[i]["recall"] - summary.iloc[i - 1]["recall"])
        for i in range(1, len(summary))
    ]
    if steps:
        stage, gain = max(steps, key=lambda step: step[1])
        points.append(
            f"The largest single gain comes from "
            f"<b>{STAGE_LABELS.get(stage, config_label(stage).lower())}</b>, worth "
            f"<b>{gain:+.1f} pts</b> of recall over the stage before it."
        )

    if best["mrr"]:
        points.append(
            f"The first relevant passage lands at rank <b>{100 / best['mrr']:.1f}</b> on average "
            f"(MRR {best['mrr']:.1f}%), and <b>{best['completeness']:.1f}%</b> of questions get "
            "every passage they need."
        )

    by_type = frame.groupby("type")["recall"].mean().mul(100).sort_values()
    if len(by_type) > 1:
        points.append(
            f"Strongest on <b>{by_type.index[-1]}</b> questions ({by_type.iloc[-1]:.0f}%), "
            f"weakest on <b>{by_type.index[0]}</b> ({by_type.iloc[0]:.0f}%)."
        )

    by_hop = frame.groupby("hop")["recall"].mean().mul(100)
    if "single" in by_hop and len(by_hop) > 1:
        multi = by_hop.drop("single").mean()
        points.append(
            f"Single-hop questions reach <b>{by_hop['single']:.0f}%</b>; those needing evidence "
            f"from more than one chunk or document drop to <b>{multi:.0f}%</b>."
        )

    return points


def render_summary(points: list[str]) -> None:
    items = "".join(f"<li>{point}</li>" for point in points)
    st.markdown(
        f'<div class="summary"><div class="summary-head">Summary</div><ul>{items}</ul></div>',
        unsafe_allow_html=True,
    )


def render_config_table(summary: pd.DataFrame) -> None:
    st.markdown(
        '<p class="section-title">Retrieval configurations</p>'
        '<p class="section-note">Each stage of the pipeline, measured on the same question set.</p>',
        unsafe_allow_html=True,
    )
    table = pd.DataFrame(
        {
            "Configuration": summary["config"].map(config_label),
            "Recall": summary["recall"],
            "MRR": summary["mrr"],
            "Completeness": summary["completeness"],
            "N": summary["n"],
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Configuration": st.column_config.TextColumn(width="medium"),
            **metric_columns({"Recall": "Recall", "MRR": "MRR", "Completeness": "Completeness"}),
            "N": st.column_config.NumberColumn("Questions", width="small"),
        },
    )


def render_breakdown(frame: pd.DataFrame) -> None:
    st.markdown(
        '<p class="section-title">Where it wins and loses</p>'
        '<p class="section-note">Recall for the selected configuration, sliced by question metadata.</p>',
        unsafe_allow_html=True,
    )
    slice_by = st.segmented_control(
        "Slice by",
        options=list(SLICE_LABELS),
        default="type",
        format_func=lambda option: SLICE_LABELS[option],
        label_visibility="collapsed",
        key="slice-by",
    )
    slice_by = slice_by or "type"

    # Completeness stays in the configuration table above: four metric columns do
    # not fit beside the chart on a narrow screen.
    grouped = frame.groupby(slice_by)[["recall", "mrr"]].mean().mul(100).round(1)
    grouped["N"] = frame.groupby(slice_by).size()
    grouped = grouped.sort_values("recall", ascending=False).reset_index()
    grouped = grouped.rename(
        columns={slice_by: SLICE_LABELS[slice_by], "recall": "Recall", "mrr": "MRR"}
    )

    table_col, chart_col = st.columns([1.5, 1], gap="medium")
    with table_col:
        st.dataframe(
            grouped,
            hide_index=True,
            width="stretch",
            column_config={
                SLICE_LABELS[slice_by]: st.column_config.TextColumn(width="medium"),
                **metric_columns({"Recall": "Recall", "MRR": "MRR"}),
                "N": st.column_config.NumberColumn(width="small"),
            },
        )
    with chart_col:
        st.bar_chart(
            grouped.set_index(SLICE_LABELS[slice_by])[["Recall"]],
            horizontal=True,
            color=active_theme().accent,
            height=max(180, 42 * len(grouped)),
        )


def render_question_explorer(frame: pd.DataFrame) -> None:
    st.markdown(
        '<p class="section-title">Per-question results</p>'
        '<p class="section-note">Filter to inspect individual retrievals.</p>',
        unsafe_allow_html=True,
    )
    search = st.text_input(
        "Search",
        placeholder="Filter by question or id…",
        label_visibility="collapsed",
        key="eval-search",
    )
    rows = frame
    if search:
        needle = search.lower()
        rows = frame[
            frame["question"].str.lower().str.contains(needle, regex=False)
            | frame["id"].str.lower().str.contains(needle, regex=False)
        ]

    table = pd.DataFrame(
        {
            "ID": rows["id"],
            "Question": rows["question"],
            "Type": rows["type"],
            "Difficulty": rows["difficulty"],
            "Recall": rows["recall"] * 100,
            "MRR": rows["mrr"] * 100,
        }
    ).sort_values("Recall")

    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=340,
        column_config={
            "ID": st.column_config.TextColumn(width="small"),
            "Question": st.column_config.TextColumn(width="large"),
            "Type": st.column_config.TextColumn(width="small"),
            "Difficulty": st.column_config.TextColumn(width="small"),
            **metric_columns({"Recall": "Recall", "MRR": "MRR"}),
        },
    )


def render_eval() -> None:
    data = load_eval_results()
    if not data:
        st.markdown(
            '<p class="page-title">Evaluation</p>'
            '<p class="page-sub">No results file yet.</p>',
            unsafe_allow_html=True,
        )
        st.info("Run `python src/eval.py` to generate `eval_results.json`.")
        return

    frame = pd.DataFrame(data["results"])
    frame = frame[~frame["abstain"]]

    st.markdown(
        '<div class="page-head">'
        '<p class="page-title">Evaluation</p>'
        f'<p class="page-sub">{data["n_answerable"]} answerable questions · '
        f'retrieval @ {data.get("top k", "?")} of '
        f'{data.get("citations retrieved", "?")} candidates</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    configs = ordered_configs(set(frame["config"]))
    summary = (
        frame.groupby("config")[["recall", "mrr", "completeness"]]
        .mean()
        .mul(100)
        .round(1)
        .reindex(configs)
    )
    summary["n"] = frame.groupby("config").size().reindex(configs)
    summary = summary.reset_index()

    render_kpis(
        summary,
        n_questions=data["n_answerable"],
        n_held_out=data["n_total"] - data["n_answerable"],
    )
    best_rows = frame[frame["config"] == summary.iloc[-1]["config"]]
    render_summary(summary_points(best_rows, summary, str(data.get("top k", "k"))))
    render_config_table(summary)

    picker, _ = st.columns([1, 2.2], gap="medium")
    selected = picker.selectbox(
        "Configuration",
        configs,
        index=len(configs) - 1,
        format_func=config_label,
        key="eval-config",
    )
    filtered = frame[frame["config"] == selected]

    with st.expander("Where it wins and loses", expanded=False):
        render_breakdown(filtered)
    with st.expander("Per-question results", expanded=False):
        render_question_explorer(filtered)


# ──────────────────────────────────────────────────────────────────────────────
# Shell
# ──────────────────────────────────────────────────────────────────────────────


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="brand">'
            '<div class="brand-name">Filings AI</div>'
            '<div class="brand-sub">10-K retrieval with citations</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        if st.session_state.mode == "Chat":
            if st.button("Evaluation →", width="stretch", key="sidebar-go-eval"):
                st.switch_page(EVAL_PAGE)
            if st.button("Clear conversation", width="stretch", key="clear"):
                st.session_state.messages = []
                st.rerun()
        elif st.button("← Chat", width="stretch", key="sidebar-go-chat"):
            st.switch_page(CHAT_PAGE)

        chips = "".join(
            f'<span class="chip">{escape(COMPANY_LABELS.get(key, key.upper()))}</span>'
            for key in COMPANIES
        )
        st.markdown(
            f'<p class="side-label">Coverage</p><div class="chip-wrap">{chips}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p class="side-label">Configuration</p>'
            f'<div class="spec"><span>Model</span><span>{escape(MODEL)}</span></div>'
            f'<div class="spec"><span>Passages</span><span>top {TOP_K}</span></div>'
            f'<div class="spec"><span>Max output</span><span>{MAX_COMPLETION_TOKENS} tokens</span></div>'
            '<div class="spec"><span>Retrieval</span><span>hybrid + rerank</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<p class="side-label">Appearance</p>', unsafe_allow_html=True)
        names = list(THEMES)
        st.selectbox(
            "Theme",
            names,
            index=names.index(st.session_state.theme),
            label_visibility="collapsed",
            key="theme_picker",
            on_change=choose_theme,
        )
        st.markdown(
            f'<p class="side-note">{escape(active_theme().blurb)}</p>',
            unsafe_allow_html=True,
        )

        with st.expander("Tips"):
            st.markdown(
                "Context is shared between retrieved passages, your question and the "
                "answer — table-heavy questions fill it quickly.\n\n"
                "**Works well:** one company, one metric, a named section.\n\n"
                "**May fail:** broad multi-company comparisons or full financial statements.\n\n"
                "Add *table* or *income statement* to route to statement data."
            )


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_prompt", None)
    st.session_state.setdefault("dialog_token", 0)
    st.session_state.setdefault("mode", "Chat")
    st.session_state.setdefault("theme", DEFAULT_THEME)


def shell(mode: str) -> None:
    st.session_state.mode = mode
    apply_theme(st.session_state.theme)
    st.markdown(stylesheet(active_theme()), unsafe_allow_html=True)
    render_sidebar()
    marker = "mode-chat" if mode == "Chat" else "mode-eval"
    st.markdown(f'<span class="{marker}" aria-hidden="true"></span>', unsafe_allow_html=True)
    if mode == "Chat":
        if st.button("Evaluation", key="fixed-eval"):
            st.switch_page(EVAL_PAGE)
    elif st.button("Chat", key="fixed-chat"):
        st.switch_page(CHAT_PAGE)


def page_chat() -> None:
    init_state()
    shell("Chat")

    app = bootstrap_app()
    if not app["ok"]:
        st.error(app["message"])
        return

    if st.session_state.pending_prompt:
        st.session_state.messages.append(
            {"role": "user", "content": st.session_state.pending_prompt}
        )
        st.session_state.pending_prompt = None

    render_chat(app)


def page_eval() -> None:
    init_state()
    shell("Evaluation")
    render_eval()


CHAT_PAGE = st.Page(page_chat, title="Chat", url_path="chat", default=True)
EVAL_PAGE = st.Page(page_eval, title="Evaluation", url_path="evaluation")


def main() -> None:
    st.set_page_config(
        page_title="Filings AI",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    st.navigation([CHAT_PAGE, EVAL_PAGE], position="top").run()


main()
