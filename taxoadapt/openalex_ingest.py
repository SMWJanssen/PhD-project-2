"""
openalex_ingest.py

Harvests peer-reviewed papers from OpenAlex (free, no API key required).
Used for Corpus 2 — healthcare AI / machine learning methodology papers.

No account or signup needed — the 'mailto' parameter is just a courtesy
header sent with each request, not a registration.

Usage:
    python openalex_ingest.py --start_year 2024 --end_year 2025 --max_papers 1000
"""

import json
import time
import argparse
import requests
from pathlib import Path

from study_filter import is_primary_study

BASE_URL = "https://api.openalex.org/works"
OUTPUT_PATH = Path("datasets/openalex_corpus.json")
API_KEY = "VeDhoqcRuhhklYfkxiSpyh"  
# Search query: AI/ML methods applied in clinical/healthcare settings
SEARCH_TERMS = '(artificial intelligence OR "machine learning" OR "deep learning") AND (clinical OR medical OR patient OR healthcare OR diagnosis OR treatment) NOT ("systematic review" OR "literature review" OR "narrative review" OR "scoping review" OR "meta-analysis" OR "overview" OR "guideline" OR "consensus" OR "position paper")'


def reconstruct_abstract(inverted_index: dict) -> str:
    """OpenAlex stores abstracts as a word -> [positions] inverted index.
    This rebuilds the plain text from that structure."""
    if not inverted_index:
        return ""
    position_word = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_word[pos] = word
    max_pos = max(position_word.keys()) if position_word else -1
    words = [position_word.get(i, "") for i in range(max_pos + 1)]
    return " ".join(words)


def fetch_papers(start_year: int, end_year: int, max_papers: int) -> list[dict]:
    """Page through OpenAlex search results, primary peer-reviewed studies only."""
    papers = []
    cursor = "*"
    per_page = 200  # OpenAlex max per page
    skipped_non_primary = 0

    print(f"Search terms: {SEARCH_TERMS}")
    print(f"Date range: {start_year}-{end_year}")
    print(f"Filter: peer-reviewed PRIMARY studies only")
    print(f"  (excludes reviews, meta-analyses, editorials, retractions, preprints)")
    print(f"Fetching up to {max_papers} papers...\n")

    while len(papers) < max_papers:
        params = {
            "search": SEARCH_TERMS,
            "filter": f"type:article|proceedings-article,is_retracted:false,from_publication_date:{start_year}-01-01,to_publication_date:{end_year}-12-31",
            "per_page": min(per_page, max_papers - len(papers)),
            "cursor": cursor,
            "api_key":API_KEY,
        }

        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  Request error: {e}")
            time.sleep(2)
            continue

        data = r.json()
        results = data.get("results", [])

        if not results:
            print("No more results.")
            break

        for work in results:
            title = work.get("title", "") or ""
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

            if not title or not abstract:
                continue  # skip papers without usable title/abstract

            type_crossref = (work.get("type_crossref") or "").lower()
            primary_location = work.get("primary_location") or {}
            source = primary_location.get("source") or {}
            source_type = source.get("type", "")

            if not is_primary_study(
                title=title,
                abstract=abstract,
                is_retracted=work.get("is_retracted", False),
                type_crossref=type_crossref,
                source_type=source_type,
            ):
                skipped_non_primary += 1
                continue  # skip reviews, meta-analyses, editorials, retractions, preprints

            papers.append({
                "openalex_id": work.get("id", ""),
                "title": title,
                "abstract": abstract,
                "year": str(work.get("publication_year", "")),
                "journal": (work.get("primary_location", {}) or {}).get("source", {}).get("display_name", "") if work.get("primary_location") else "",
                "doi": work.get("doi", ""),
                "type": work.get("type", ""),
            })

        print(f"  Fetched {len(papers)} / {max_papers} papers (with abstracts). Skipped {skipped_non_primary} non-primary (reviews/editorials/etc)...", flush=True)

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

        time.sleep(0.2)  # be polite to the API

    print(f"\nTotal primary-study papers with abstracts collected: {len(papers)}")
    print(f"Total non-primary papers skipped: {skipped_non_primary}")
    return papers[:max_papers]


def save_corpus(papers: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    formatted_papers = []
    for idx, p in enumerate(papers, start=1):
        formatted_papers.append({
            "paper_id": f"openalex_{idx:04d}",
            "source_file": p.get("openalex_id", ""),
            "title": p.get("title", ""),
            "abstract": p.get("abstract", ""),
            "year": p.get("year", ""),
            "journal": p.get("journal", ""),
            "doi": p.get("doi", ""),
            "extraction_status": "ok",
        })

    corpus = {
        "corpus_name": "openalex_healthcare_ai",
        "topic_hint": "artificial intelligence machine learning healthcare",
        "papers": formatted_papers,
    }
    OUTPUT_PATH.write_text(
        json.dumps(corpus, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=2022)
    parser.add_argument("--end_year",   type=int, default=2025)
    parser.add_argument("--max_papers", type=int, default=1000)
    args = parser.parse_args()

    papers = fetch_papers(args.start_year, args.end_year, args.max_papers)

    if not papers:
        print("No papers found. Check your query or date range.")
        return

    save_corpus(papers)

    print(f"\nDone!")
    print(f"  Total papers: {len(papers)}")
    print(f"  Saved:        {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
