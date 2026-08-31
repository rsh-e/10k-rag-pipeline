"""Rebuild src/questions.py from the live FTS store. Run from src/: python3 _rebuild_questions.py"""

from __future__ import annotations

import sqlite3
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STORE = ROOT.parent / "storage" / "document_ledger.db"

HEADER = '''\
"""
Retrieval + answer eval set for the 10-issuer 10-K corpus
(amd, axp, cop, cvx, goog, jnj, ko, meta, nvda, pep).

Question labels
  id            Stable slug for the item (issuer-or-cmp + topic).
  question      The user query as written.
  type          factoid | event | risk | comparison | underspecified | distractor | unanswerable
  difficulty    easy | medium | hard
  companies     Issuers whose 10-Ks the gold hashes come from.
  hop           single | multi_chunk | multi_doc
  abstain       True = corpus cannot answer; relevant_chunks is [].
  answer        Canonical short answer, grounded only in the gold hashes.
  must_have     Substrings a generated answer should contain.

Hash labels
  primary_chunks     Best sufficient passage(s). One unique primary per fact.
  equivalent_chunks  Other passages that also fully support the same answer.
  relevant_chunks    Union of primary + equivalent (eval.py scorer).
  citations[].role   "primary" or "equivalent".
  is_table           True when the gold passage is a markdown grid table.

Verify gold hashes (do not eyeball the whole set):
  python3 src/verify_eval_hashes.py
"""
'''

# Drop: Mastercard, same-deal date/price twins, same-hash topic clones.
DROP_IDS = {
    "amd-zt-date",
    "amd-zt-price",
    "amd-zt-underspec",
    "goog-wiz-price",
    "goog-intersect-price",
    "jnj-ict-price",
    "pep-employees-us",
    "meta-facebook-name",
    "cvx-hess-date",
    "cvx-hess-price",
    "cvx-hess-synergy",
    "cvx-loma-expiry",
    "cvx-ghg-oil-intensity",
    "cop-marathon-date",
    "ma-not-issuer",
    "ma-four-party",
    "ma-switching",
    "ma-recorded-future",
    "ma-interchange-lit",
    "ma-nrev-growth",
    "ma-vs-axp-distractor",
    "abs-visa-acquire-ma",
    "abs-ma-bitcoin-card",
    "cmp-hess-marathon",
    "goog-founders-underspec",
    "axp-closed-loop-underspec",
    "cop-marathon-synergy",
}

# id -> (primary prefixes, equivalent prefixes) to replace stale gold.
HASH_FIX = {
    "goog-rev-2025": (["51d6f1d5881092c6"], []),
    "goog-segments": (["e33caa8bab0c8e89"], []),
    "jnj-sales-2025": (["5feb6d1ce7d54b50"], []),
    "jnj-segments": (["9603bfbc7c387bd8"], []),
    "jnj-ict-date": (["3f8ee7f7528d2846"], []),
    "jnj-talc-reversal": (["df5bba0100478a95"], []),
    "ko-nrev-2025": (["0a3233b93296c7a3"], []),
    "ko-auditor": (["f7bd39d9b60595ab"], []),
    "cop-production": (["b97ca8ef4e47dc8a"], []),
    "cvx-clair": (["03890ccfd375a219"], []),
    "cvx-oil-price-risk": (["61505b567e4c6712"], []),
    "cmp-meta-goog-ads": (["a37338ec70761ec1", "89853a6b4adc83ba"], []),
    "cmp-cvx-cop-price": (["61505b567e4c6712", "e6d8431634114ba2"], []),
}

QUESTION_FIX = {
    "jnj-ict-date": "On what date did Johnson & Johnson close the Intra-Cellular Therapies deal?",
    "jnj-talc-reversal": "Does Johnson & Johnson still describe material product-liability litigation, including talc?",
    "jnj-sales-2025": "How much were Johnson & Johnson's worldwide sales in 2025, and how fast did they grow?",
    "goog-rev-2025": "How much revenue did Alphabet generate in 2025, and how fast did that grow versus 2024?",
    "goog-segments": "Besides Search ads, how does Alphabet break Google into reportable businesses?",
    "ko-nrev-2025": "What were Coca-Cola's net operating revenues in 2025, and how did that compare with 2024?",
    "ko-auditor": "Who is Coca-Cola's independent auditor?",
    "cop-production": "What full-year production did ConocoPhillips deliver in 2025, in MBOED?",
    "cvx-clair": "How much of the Clair Field does Chevron own, and where is that field?",
    "cvx-loma-interest": "What interest does Chevron hold in Loma Campana, and when does that concession expire?",
    "pep-employees-world": "How many people did PepsiCo employ worldwide and in the United States at year-end 2025?",
    "cmp-meta-goog-ads": "How central was advertising to Meta versus Alphabet in 2025?",
}

ANSWER_FIX = {
    "jnj-ict-date": "Johnson & Johnson acquired Intra-Cellular Therapies on April 2, 2025.",
    "jnj-talc-reversal": "Yes. The 10-K says J&J and its subsidiaries face numerous claims including product liability; talc is among the disclosed litigation matters.",
    "jnj-sales-2025": "Worldwide sales increased 6.0% to $94.2 billion in 2025.",
    "goog-rev-2025": "Revenues were $402.8 billion, up 15% year over year, driven by Google Services (+12%) and Google Cloud (+36%).",
    "goog-segments": "Google is reported as two segments: Google Services and Google Cloud.",
    "ko-nrev-2025": "Net operating revenues were $47,941 million in 2025, versus $47,061 million in 2024, up $880 million or 2%.",
    "ko-auditor": "Ernst & Young LLP, appointed by the Audit Committee.",
    "cop-production": "Full-year total company production was 2,375 MBOED (Lower 48 1,484 MBOED).",
    "cvx-clair": "Chevron holds a 19.4% nonoperated working interest in the Clair Field, west of the Shetland Islands.",
    "cvx-loma-interest": "Chevron has a 50% nonoperated interest in Loma Campana (Vaca Muerta). The Loma Campana concession expires in 2048.",
    "pep-employees-world": "About 306,000 people worldwide as of December 27, 2025, including about 125,000 in the United States.",
    "cmp-meta-goog-ads": "Meta generates substantially all revenue from advertising. Alphabet generated more than 70% of 2025 total revenues from online advertising.",
}

MUST_FIX = {
    "jnj-ict-date": ["April 2, 2025"],
    "jnj-talc-reversal": ["product liability"],
    "jnj-sales-2025": ["$94.2", "6.0%"],
    "goog-rev-2025": ["$402.8 billion", "15%"],
    "goog-segments": ["Google Services", "Google Cloud"],
    "ko-nrev-2025": ["$47,941", "2%"],
    "ko-auditor": ["Ernst & Young"],
    "cop-production": ["2,375"],
    "cvx-clair": ["19.4"],
    "cvx-loma-interest": ["50 percent", "2048"],
    "pep-employees-world": ["306,000", "125,000"],
    "nvda-china-foreclosed": ["foreclosed"],
    "cop-willow": ["Willow", "2029"],
    "pep-table-op-profit": ["11498", "(11)%"],
    "cmp-meta-goog-ads": ["substantially all", "70%"],
}


def resolve(conn: sqlite3.Connection, prefix: str) -> dict:
    row = conn.execute(
        """
        SELECT citation_text_hash, company, item, section, heading, text
        FROM citations WHERE citation_text_hash LIKE ?
        """,
        (prefix + "%",),
    ).fetchone()
    if not row:
        raise SystemExit(f"hash prefix not in store: {prefix}")
    return {
        "hash": row[0],
        "company": row[1],
        "item": row[2] or "",
        "section": row[3] or "",
        "heading": row[4] or "",
        "text": row[5] or "",
    }


def cite(meta: dict, role: str) -> dict:
    return {
        "hash": meta["hash"],
        "company": meta["company"],
        "item": meta["item"],
        "section": meta["section"],
        "heading": meta["heading"],
        "role": role,
        "is_table": meta["text"].lstrip().startswith("+--"),
    }


def fill(conn: sqlite3.Connection, item: dict, primaries: list[str], equivalents: list[str]) -> dict:
    p_meta = [resolve(conn, p) for p in primaries]
    e_meta = [resolve(conn, e) for e in equivalents]
    item["primary_chunks"] = [m["hash"] for m in p_meta]
    item["equivalent_chunks"] = [m["hash"] for m in e_meta]
    item["relevant_chunks"] = item["primary_chunks"] + item["equivalent_chunks"]
    item["citations"] = [cite(m, "primary") for m in p_meta] + [cite(m, "equivalent") for m in e_meta]
    return item


NEW_ITEMS = [
    {
        "id": "amd-zt-deal",
        "question": "When did AMD close ZT Systems, and what was the total purchase consideration?",
        "type": "event",
        "difficulty": "medium",
        "companies": ["amd"],
        "hop": "single",
        "abstain": False,
        "answer": "AMD completed the ZT Systems acquisition on March 31, 2025 for a total purchase consideration of $4.4 billion.",
        "must_have": ["March 31, 2025", "$4.4 billion"],
        "primary": ["90ad81cdcca16edd"],
        "equivalent": [],
    },
    {
        "id": "amd-cash-2025",
        "question": "How much cash and short-term investments did AMD hold at year-end 2025 versus 2024?",
        "type": "factoid",
        "difficulty": "easy",
        "companies": ["amd"],
        "hop": "single",
        "abstain": False,
        "answer": "Cash, cash equivalents and short-term investments were $10.6 billion as of December 27, 2025, compared with $5.1 billion as of December 28, 2024.",
        "must_have": ["$10.6 billion", "$5.1 billion"],
        "primary": ["87902b97214b5f0d"],
        "equivalent": [],
    },
    {
        "id": "amd-table-client-rev",
        "question": "From AMD's 2025 segment revenue table, how much net revenue did the Client line report?",
        "type": "factoid",
        "difficulty": "medium",
        "companies": ["amd"],
        "hop": "single",
        "abstain": False,
        "answer": "Client net revenue was $10,640 million for the year ended December 27, 2025 ($7,054 million in 2024).",
        "must_have": ["10640"],
        "primary": ["e9697e4b3e29997f"],
        "equivalent": [],
    },
    {
        "id": "nvda-table-gm",
        "question": "What gross margin did NVIDIA report for the fiscal year ended January 25, 2026 in its summary results table?",
        "type": "factoid",
        "difficulty": "medium",
        "companies": ["nvda"],
        "hop": "single",
        "abstain": False,
        "answer": "Gross margin was 71.1% for the year ended January 25, 2026, down 3.9 points from 75.0% the prior year.",
        "must_have": ["71.1%"],
        "primary": ["b6d874799b46220f"],
        "equivalent": [],
    },
    {
        "id": "goog-pending-deals",
        "question": "What did Alphabet agree to pay for Wiz and for Intersect, and when are those deals expected to close?",
        "type": "event",
        "difficulty": "medium",
        "companies": ["goog"],
        "hop": "single",
        "abstain": False,
        "answer": "Wiz: $32.0 billion all-cash, expected to close in 2026. Intersect: $4.8 billion in cash plus assumed debt, expected to close in the first half of 2026.",
        "must_have": ["$32.0", "$4.8"],
        "primary": ["6b6600f98222cbcf"],
        "equivalent": [],
    },
    {
        "id": "goog-table-opinc",
        "question": "In Alphabet's consolidated results table, what was 2025 operating income and operating margin?",
        "type": "factoid",
        "difficulty": "medium",
        "companies": ["goog"],
        "hop": "single",
        "abstain": False,
        "answer": "Operating income was $129,039 million in 2025 with a 32% operating margin (flat versus 2024).",
        "must_have": ["129039", "32%"],
        "primary": ["de4010929b5d608a"],
        "equivalent": [],
    },
    {
        "id": "meta-table-ad-rev",
        "question": "How much advertising revenue did Meta report for 2025 in its revenue-mix table?",
        "type": "factoid",
        "difficulty": "easy",
        "companies": ["meta"],
        "hop": "single",
        "abstain": False,
        "answer": "Advertising revenue was $196,175 million in 2025, up 22% from $160,633 million in 2024.",
        "must_have": ["196175"],
        "primary": ["9bec479106b6c86a"],
        "equivalent": [],
    },
    {
        "id": "jnj-im-sales",
        "question": "How large was Johnson & Johnson's Innovative Medicine segment in 2025?",
        "type": "factoid",
        "difficulty": "easy",
        "companies": ["jnj"],
        "hop": "single",
        "abstain": False,
        "answer": "Innovative Medicine segment sales were $60.4 billion in 2025, up 6.0% from 2024.",
        "must_have": ["$60.4 billion", "6.0%"],
        "primary": ["e8a0e3b4d93244f3"],
        "equivalent": [],
    },
    {
        "id": "jnj-table-darzalex",
        "question": "What were DARZALEX sales in 2025 according to J&J's Innovative Medicine therapeutic-area table?",
        "type": "factoid",
        "difficulty": "medium",
        "companies": ["jnj"],
        "hop": "single",
        "abstain": False,
        "answer": "DARZALEX sales were $14,351 million in 2025, up 23.0% from $11,670 million in 2024.",
        "must_have": ["14351"],
        "primary": ["00b80896ca96a2f3"],
        "equivalent": [],
    },
    {
        "id": "ko-table-opinc",
        "question": "What operating income did Coca-Cola report for 2025 on its consolidated income statement table?",
        "type": "factoid",
        "difficulty": "medium",
        "companies": ["ko"],
        "hop": "single",
        "abstain": False,
        "answer": "Operating income was $13,762 million in 2025, versus $9,992 million in 2024.",
        "must_have": ["13762"],
        "primary": ["80283730092952da"],
        "equivalent": [],
    },
    {
        "id": "pep-table-op-profit",
        "question": "Did PepsiCo's consolidated operating profit rise or fall in 2025, and to what level?",
        "type": "factoid",
        "difficulty": "medium",
        "companies": ["pep"],
        "hop": "single",
        "abstain": False,
        "answer": "Operating profit was $11,498 million in 2025, down 11% from $12,887 million in 2024. Operating margin was 12.2%.",
        "must_have": ["11498", "(11)%"],
        "primary": ["56bfb82d7b7c9c57"],
        "equivalent": [],
    },
    {
        "id": "axp-table-ni",
        "question": "What net income did American Express report for 2025 on its consolidated income statement?",
        "type": "factoid",
        "difficulty": "easy",
        "companies": ["axp"],
        "hop": "single",
        "abstain": False,
        "answer": "Net income was $10,833 million in 2025, versus $10,129 million in 2024.",
        "must_have": ["10833"],
        "primary": ["963f68202b410600"],
        "equivalent": [],
    },
    {
        "id": "cop-2026-guide",
        "question": "What 2026 production range did ConocoPhillips guide to?",
        "type": "factoid",
        "difficulty": "medium",
        "companies": ["cop"],
        "hop": "single",
        "abstain": False,
        "answer": "2026 production guidance is 2.33 to 2.36 MMBOED, with first-quarter 2026 expected at 2.30 to 2.34 MMBOED. Capex guided at about $12 billion.",
        "must_have": ["2.33", "2.36"],
        "primary": ["309cdfffb9abeb18"],
        "equivalent": [],
    },
    {
        "id": "cop-table-crude",
        "question": "From ConocoPhillips' 2025 operating-statistics table, what was total crude-oil production in MBD?",
        "type": "factoid",
        "difficulty": "medium",
        "companies": ["cop"],
        "hop": "single",
        "abstain": False,
        "answer": "Total crude oil was 1,145 MBD in 2025 (1,133 consolidated operations plus 12 equity affiliates).",
        "must_have": ["1145"],
        "primary": ["9cf3fa654d484e90"],
        "equivalent": [],
    },
    {
        "id": "cvx-hess-deal",
        "question": "When did Chevron close Hess, and what was the aggregate purchase price?",
        "type": "event",
        "difficulty": "medium",
        "companies": ["cvx"],
        "hop": "single",
        "abstain": False,
        "answer": "Chevron acquired Hess on July 18, 2025. The aggregate purchase price was about $48 billion; it also assumed $8.8 billion of Hess debt.",
        "must_have": ["July 18, 2025", "$48 billion"],
        "primary": ["f3bf71a8bc4a60d4"],
        "equivalent": [],
    },
    {
        "id": "cvx-table-ni",
        "question": "What net income attributable to Chevron did the 2025 key financial results table show, and what was diluted EPS?",
        "type": "factoid",
        "difficulty": "medium",
        "companies": ["cvx"],
        "hop": "single",
        "abstain": False,
        "answer": "Net income attributable to Chevron was $12,299 million in 2025. Diluted EPS was $6.63 ($9.72 in 2024).",
        "must_have": ["12299", "6.63"],
        "primary": ["40d56cf33ce6379f"],
        "equivalent": [],
    },
    {
        "id": "abs-ma-interchange",
        "question": "What interchange rate did Mastercard set for U.S. credit cards in 2025?",
        "type": "unanswerable",
        "difficulty": "easy",
        "companies": [],
        "hop": "single",
        "abstain": True,
        "answer": "Mastercard is not in this corpus. Abstain.",
        "must_have": [],
        "primary": [],
        "equivalent": [],
    },
    {
        "id": "abs-visa-volume",
        "question": "What was Visa's total payment volume in 2025?",
        "type": "unanswerable",
        "difficulty": "easy",
        "companies": [],
        "hop": "single",
        "abstain": True,
        "answer": "Visa is not in this corpus. Abstain.",
        "must_have": [],
        "primary": [],
        "equivalent": [],
    },
]


def load_old() -> list[dict]:
    ns: dict = {}
    exec(compile((ROOT / "questions.py").read_text(), "questions.py", "exec"), ns)
    return list(ns["eval_set"])


def py_str(value) -> str:
    return repr(value)


def emit_item(item: dict) -> str:
    lines = ["    {"]
    order = [
        "id",
        "question",
        "type",
        "difficulty",
        "companies",
        "hop",
        "abstain",
        "answer",
        "must_have",
        "primary_chunks",
        "equivalent_chunks",
        "relevant_chunks",
        "citations",
    ]
    for key in order:
        val = item[key]
        if key != "citations":
            lines.append(f"        {py_str(key)}: {py_str(val)},")
        else:
            lines.append('        "citations": [')
            for c in val:
                lines.append("        {")
                for ck in ("hash", "company", "item", "section", "heading", "role", "is_table"):
                    lines.append(f"            {py_str(ck)}: {py_str(c[ck])},")
                lines.append("        },")
            lines.append("    ],")
    lines.append("    },")
    return "\n".join(lines)


def main() -> None:
    conn = sqlite3.connect(STORE)
    old = load_old()
    out: list[dict] = []
    used_primary: dict[str, str] = {}

    for item in old:
        if item["id"] in DROP_IDS:
            continue
        item = dict(item)
        if item["id"] in QUESTION_FIX:
            item["question"] = QUESTION_FIX[item["id"]]
        if item["id"] in ANSWER_FIX:
            item["answer"] = ANSWER_FIX[item["id"]]
        if item["id"] in MUST_FIX:
            item["must_have"] = MUST_FIX[item["id"]]
        if item["abstain"]:
            item["primary_chunks"] = []
            item["equivalent_chunks"] = []
            item["relevant_chunks"] = []
            item["citations"] = []
            out.append(item)
            continue
        if item["id"] in HASH_FIX:
            p, e = HASH_FIX[item["id"]]
            fill(conn, item, p, e)
        else:
            if not item["id"].startswith("cmp-"):
                item["equivalent_chunks"] = []
                item["relevant_chunks"] = list(item["primary_chunks"])
                item["citations"] = [
                    c for c in item.get("citations") or [] if c.get("role") == "primary"
                ]
            for c in item.get("citations") or []:
                rec = resolve(conn, c["hash"])
                c["is_table"] = rec["text"].lstrip().startswith("+--")
        for h in item["primary_chunks"]:
            if h in used_primary and not item["id"].startswith("cmp-"):
                print(f"warn: primary {h[:12]} reused by {used_primary[h]} and {item['id']}")
            else:
                used_primary.setdefault(h, item["id"])
        out.append(item)

    existing_ids = {it["id"] for it in out}
    for spec in NEW_ITEMS:
        if spec["id"] in existing_ids:
            continue
        item = {k: spec[k] for k in spec if k not in ("primary", "equivalent")}
        if spec.get("abstain"):
            item["primary_chunks"] = []
            item["equivalent_chunks"] = []
            item["relevant_chunks"] = []
            item["citations"] = []
        else:
            fill(conn, item, spec["primary"], spec["equivalent"])
            for h in item["primary_chunks"]:
                if h in used_primary and not item["id"].startswith("cmp-"):
                    print(f"warn: new primary {h[:12]} reused by {used_primary[h]} and {item['id']}")
                used_primary.setdefault(h, item["id"])
        out.append(item)

    # cmp-meta-goog-ads now uses table hashes also used by meta-table-ad-rev and goog-table-opinc.
    # Point the comparison at those tables on purpose; strip equivalent reuse warnings for cmp-*.

    body = "\n".join(emit_item(it) for it in out)
    text = HEADER + "\n\neval_set = [\n" + body + "\n]\n"
    dest = ROOT / "questions.py"
    dest.write_text(text)
    n_abs = sum(1 for x in out if x["abstain"])
    print(f"wrote {dest} items={len(out)} answerable={len(out)-n_abs} abstain={n_abs}")


if __name__ == "__main__":
    main()
