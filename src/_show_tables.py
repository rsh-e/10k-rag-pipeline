import sqlite3
import sys

conn = sqlite3.connect("storage/document_ledger.db")
for prefix in sys.argv[1:]:
    row = conn.execute(
        """
        SELECT citation_text_hash, company, item, section, heading, text
        FROM citations WHERE citation_text_hash LIKE ?
        """,
        (prefix + "%",),
    ).fetchone()
    print("=" * 80)
    if not row:
        print("MISSING", prefix)
        continue
    print(row[0])
    print(row[1], "|", (row[2] or "")[:50], "|", (row[3] or "")[:40], "|", (row[4] or "")[:30])
    print(row[5][:2200])
    print()
