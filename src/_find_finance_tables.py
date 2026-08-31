import sqlite3

conn = sqlite3.connect("storage/document_ledger.db")

# Existing primaries we must not reuse
used = {
    "e9697e4b3e29997f",
    "b6d874799b46220f",
    "de4010929b5d608a",
    "9bec479106b6c86a",
    "00b80896ca96a2f3",
    "80283730092952da",
    "56bfb82d7b7c9c57",
    "963f68202b410600",
    "9cf3fa654d484e90",
    "40d56cf33ce6379f",
}

needles = [
    ("amd", "Net income"),
    ("amd", "Total assets"),
    ("nvda", "Data Center"),
    ("nvda", "Gross profit"),
    ("goog", "Google Cloud"),
    ("goog", "Google Search"),
    ("meta", "Reality Labs"),
    ("meta", "Family of Apps"),
    ("jnj", "MedTech"),
    ("jnj", "Innovative Medicine"),
    ("ko", "United States"),
    ("ko", "Bottling"),
    ("pep", "FLNA"),
    ("pep", "PBNA"),
    ("axp", "Discount revenue"),
    ("axp", "Total assets"),
    ("cop", "Lower 48"),
    ("cop", "Sales and other"),
    ("cvx", "Upstream"),
    ("cvx", "Sales and Other Operating"),
]

for company, term in needles:
    rows = conn.execute(
        """
        SELECT citation_text_hash, section, heading, length(text), substr(text,1,160)
        FROM citations
        WHERE company=? AND text LIKE '+--%' AND text LIKE ?
        LIMIT 4
        """,
        (company, f"%{term}%"),
    ).fetchall()
    print(f"\n==== {company} {term!r} n={len(rows)} ====")
    for r in rows:
        flag = "USED" if r[0][:16] in used else "ok"
        print(flag, r[0][:16], "|", (r[1] or "")[:32], "|", (r[2] or "")[:24], "|", r[3])
        print("   ", r[4].replace("\n", " ")[:150])
