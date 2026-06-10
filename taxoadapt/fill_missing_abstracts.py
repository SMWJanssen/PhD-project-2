"""
fill_missing_abstracts.py

Finds papers in slr_corpus.json with missing abstracts,
searches Scopus by title, and fills in the gaps automatically.
"""

import json
import re
import time
from pathlib import Path

import pybliometrics
pybliometrics.init()

from pybliometrics.scopus import ScopusSearch

CORPUS_PATH = Path("datasets/slr_corpus.json")

# Patterns that indicate a line is journal metadata, NOT a title
JUNK_PATTERNS = [
    r"available online",
    r"contents lists available",
    r"sciencedirect",
    r"www\.",
    r"http",
    r"vol\.?\s*\d+",
    r"©",
    r"open access",
    r"all rights reserved",
    r"\d{4}[-/]\d+[-/]\d+",
    r"^\s*\d+\s*$",
    r"elsevier",
    r"springer",
    r"wiley",
    r"issn",
    r"doi",
]

def clean_title(title: str) -> str:
    """Remove junk metadata lines, keep only the real title."""
    if not title:
        return ""
    lines = [l.strip() for l in title.splitlines() if l.strip()]
    good_lines = []
    for line in lines:
        low = line.lower()
        if any(re.search(p, low) for p in JUNK_PATTERNS):
            continue
        good_lines.append(line)
    result = " ".join(good_lines[:3]).strip()
    result = re.sub(r"\s+\d{1,4}\s*\(\d{4}\).*$", "", result).strip()
    return result


def search_scopus_by_title(title: str) -> dict | None:
    """Search Scopus for a paper by title. Returns title+abstract or None."""
    try:
        clean = title.replace('"', '').replace("'", "")[:120]
        query = f'TITLE("{clean}")'
        search = ScopusSearch(query, download=True)

        if not search.results:
            words = clean.split()[:7]
            query = "TITLE(" + " AND ".join(words) + ")"
            search = ScopusSearch(query, download=True)

        if not search.results:
            return None

        result = search.results[0]
        return {
            "title": result.title or "",
            "abstract": result.description or "",
        }

    except Exception as e:
        print(f"  Scopus error: {e}")
        return None


def main():
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    papers = data["papers"]

    missing = [p for p in papers if p["extraction_status"] == "missing_abstract"]
    print(f"Found {len(missing)} papers with missing abstracts\n")

    filled = 0
    still_missing = 0

    for paper in missing:
        raw_title = paper.get("title", "").strip()
        title = clean_title(raw_title)
        print(f"Processing: {paper['source_file']}")
        print(f"  Raw title:   {raw_title[:70]}")
        print(f"  Clean title: {title[:70]}")

        if not title or len(title.split()) < 3:
            print("  Title too short or empty after cleaning — skipping")
            still_missing += 1
            continue

        result = search_scopus_by_title(title)

        if result and result["abstract"]:
            paper["abstract"] = result["abstract"]
            paper["title"] = result["title"]
            paper["extraction_status"] = "ok"
            print(f"  Filled ({len(result['abstract'])} chars)")
            filled += 1
        else:
            print(f"  Not found on Scopus")
            still_missing += 1

        time.sleep(1)

    CORPUS_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nDone!")
    print(f"  Filled:        {filled}")
    print(f"  Still missing: {still_missing}")
    print(f"  Saved to:      {CORPUS_PATH}")

    if still_missing > 0:
        print(f"\nStill missing abstracts:")
        for p in papers:
            if p["extraction_status"] == "missing_abstract":
                print(f"  - {p['source_file']}: {p['title'][:60]}")


if __name__ == "__main__":
    main()
