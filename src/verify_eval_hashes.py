"""
Check that every eval gold hash exists in the live store and actually
supports the labeled answer. Run from repo root or src/:

    python3 src/verify_eval_hashes.py
    python3 src/verify_eval_hashes.py --show jnj-sales-2025
    python3 src/verify_eval_hashes.py --strict-unique
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import questions  # noqa: E402

STORE = ROOT.parent / "storage" / "document_ledger.db"


def load_store(conn: sqlite3.Connection) -> dict[str, dict]:
    rows = conn.execute(
        """
        SELECT citation_text_hash, company, item, section, heading, text
        FROM citations
        """
    ).fetchall()
    return {
        r[0]: {
            "company": r[1] or "",
            "item": r[2] or "",
            "section": r[3] or "",
            "heading": r[4] or "",
            "text": r[5] or "",
        }
        for r in rows
    }


def norm(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace())


def _compact(s: str) -> str:
    s = s.lower().replace("\u00a0", " ").replace("percent", "%")
    out = []
    for ch in s:
        if ch.isalnum() or ch in ".%":
            out.append(ch)
    return "".join(out)


def must_have_hit(needle: str, blob: str) -> bool:
    if needle.lower() in blob.lower().replace("\u00a0", " "):
        return True
    compact_n, compact_b = _compact(needle), _compact(blob)
    return bool(compact_n) and compact_n in compact_b


def check_item(item: dict, store: dict[str, dict]) -> list[str]:
    errors: list[str] = []
    gold = item.get("relevant_chunks") or []
    primary = item.get("primary_chunks") or []
    equivalent = item.get("equivalent_chunks") or []
    cites = item.get("citations") or []

    if item.get("abstain"):
        if gold or primary or equivalent:
            errors.append("abstain item still has gold hashes")
        return errors

    if not primary:
        errors.append("answerable item has no primary_chunks")
    if set(gold) != set(primary) | set(equivalent):
        errors.append("relevant_chunks != primary ∪ equivalent")

    cite_h = [c["hash"] for c in cites]
    if set(cite_h) != set(gold):
        errors.append("citation hashes != relevant_chunks")

    companies = set(item.get("companies") or [])
    gold_companies: set[str] = set()
    blob_parts: list[str] = []
    for h in gold:
        rec = store.get(h)
        if rec is None:
            errors.append(f"hash not in store: {h}")
            continue
        gold_companies.add(rec["company"])
        blob_parts.append(rec["text"])
        role = next((c["role"] for c in cites if c["hash"] == h), None)
        if h in primary and role != "primary":
            errors.append(f"primary {h[:12]} missing role=primary")
        if h in equivalent and role != "equivalent":
            errors.append(f"equivalent {h[:12]} missing role=equivalent")
        if rec["company"] and rec["company"] not in companies:
            errors.append(f"hash {h[:12]} is company={rec['company']} not in {sorted(companies)}")

    if companies and gold_companies and not companies <= gold_companies | companies:
        missing_co = companies - gold_companies
        if missing_co:
            errors.append(f"no gold chunk for companies {sorted(missing_co)}")

    blob = "\n".join(blob_parts)
    for needle in item.get("must_have") or []:
        if not must_have_hit(needle, blob):
            errors.append(f"must_have {needle!r} not found in gold text")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify eval gold hashes against the live store.")
    parser.add_argument("--show", metavar="ID", help="Print gold text for one item and exit")
    parser.add_argument(
        "--strict-unique",
        action="store_true",
        help="Fail if two non-comparison items share a primary hash",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(STORE)
    store = load_store(conn)

    if args.show:
        item = next((x for x in questions.eval_set if x["id"] == args.show), None)
        if not item:
            print("unknown id", args.show)
            return 1
        print(item["id"], "|", item["question"])
        print("answer:", item.get("answer"))
        print("must_have:", item.get("must_have"))
        for h in item.get("relevant_chunks") or []:
            rec = store.get(h)
            print("=" * 80)
            print(h)
            if not rec:
                print("MISSING FROM STORE")
                continue
            print(rec["company"], rec["item"], rec["section"], rec["heading"])
            print(rec["text"][:2500])
        return 0

    failures = 0
    primary_owners: dict[str, list[str]] = defaultdict(list)
    n_abs = 0
    n_table = 0
    for item in questions.eval_set:
        if item.get("abstain"):
            n_abs += 1
        errs = check_item(item, store)
        for h in item.get("primary_chunks") or []:
            if not item["id"].startswith("cmp-"):
                primary_owners[h].append(item["id"])
        if any(c.get("is_table") for c in item.get("citations") or []):
            n_table += 1
        if errs:
            failures += 1
            print(f"FAIL {item['id']}")
            for e in errs:
                print(f"  - {e}")
        else:
            print(f"PASS {item['id']}")

    print()
    reused = {h: ids for h, ids in primary_owners.items() if len(ids) > 1}
    if reused:
        print(f"shared non-cmp primaries: {len(reused)}")
        for h, ids in reused.items():
            print(f"  {h[:16]} {ids}")
        if args.strict_unique:
            failures += 1

    n = len(questions.eval_set)
    print()
    print(f"items={n} answerable={n-n_abs} abstain={n_abs} table_gold={n_table}")
    print(f"store_chunks={len(store)} failed_items={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
