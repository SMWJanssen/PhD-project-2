"""
scopus_ingest.py

Harvests papers from Scopus using the Search + Abstract Retrieval APIs.
Outputs a unified corpus JSON compatible with the TaxoAdapt pipeline.

Usage:
    python scopus_ingest.py --start_year 2024 --end_year 2025 --max_papers 3200
"""

import json
import re
import time
import argparse
import requests
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
CFG_PATH = Path.home() / ".config" / "pybliometrics.cfg"
SEARCH_URL = "https://api.elsevier.com/content/search/scopus"
ABSTRACT_URL = "https://api.elsevier.com/content/abstract/doi/{doi}"
OUTPUT_PATH = Path("datasets/scopus_corpus.json")
QUERY = 'TITLE-ABS-KEY(diabetes AND ("artificial intelligence" OR "machine learning"))'

def load_api_key() -> str:
    cfg = CFG_PATH.read_text()
    match = re.search(r"APIKey\s*=\s*(\S+)", cfg)
    if not match:
        raise ValueError(f"No APIKey found in {CFG_PATH}")
    return match.group(1)


# ── Step 1: Search for DOIs ──────────────────────────────────────────────────
def fetch_dois(api_key: str, start_year: int, end_year: int, max_papers: int) -> list[dict]:
    """Page through Scopus search results and collect title + DOI."""
    query = f"{QUERY} AND PUBYEAR > {start_year - 1} AND PUBYEAR < {end_year + 1}"
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    fields = "title,doi,publicationName,coverDate,citedby-count"

    results = []
    start = 0
    page_size = 25  # Scopus max per request on free tier

    print(f"Query: {query}")
    print(f"Fetching up to {max_papers} papers...\n")

    while len(results) < max_papers:
        params = {
            "query": query,
            "count": page_size,
            "start": start,
            "field": fields,
        }
        try:
            r = requests.get(SEARCH_URL, params=params, headers=headers, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  Search error at offset {start}: {e}")
            time.sleep(5)
            continue

        data = r.json().get("search-results", {})
        entries = data.get("entry", [])

        if not entries:
            print("No more results.")
            break

        for e in entries:
            doi = e.get("prism:doi", "").strip()
            title = e.get("dc:title", "").strip()
            if doi and title:
                results.append({
                    "doi": doi,
                    "title": title,
                    "journal": e.get("prism:publicationName", ""),
                    "year": e.get("prism:coverDate", "")[:4],
                })

        total_available = int(data.get("opensearch:totalResults", 0))
        print(f"  Fetched {len(results)} / {min(max_papers, total_available)} papers...", flush=True)

        start += page_size
        if start >= total_available:
            break

        time.sleep(0.5)  # be polite to the API

    print(f"\nTotal DOIs collected: {len(results)}")
    return results[:max_papers]


# ── Step 2: Fetch abstracts ──────────────────────────────────────────────────
def fetch_abstract(doi: str, api_key: str) -> str | None:
    """Fetch abstract for a single paper by DOI."""
    url = ABSTRACT_URL.format(doi=doi)
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            coredata = r.json().get("abstracts-retrieval-response", {}).get("coredata", {})
            return coredata.get("dc:description", None)
        elif r.status_code == 404:
            return None
        else:
            print(f"  HTTP {r.status_code} for DOI {doi}")
            return None
    except requests.RequestException as e:
        print(f"  Error fetching abstract for {doi}: {e}")
        return None


# ── Step 3: Build corpus ─────────────────────────────────────────────────────
def build_corpus(doi_list: list[dict], api_key: str) -> list[dict]:
    """Fetch abstracts for all papers and build corpus records."""
    papers = []
    total = len(doi_list)

    for idx, item in enumerate(doi_list, start=1):
        doi = item["doi"]
        print(f"[{idx}/{total}] {item['title'][:60]}", flush=True)

        abstract = fetch_abstract(doi, api_key)

        papers.append({
            "paper_id": f"scopus_{idx:04d}",
            "source_file": doi,
            "title": item["title"],
            "abstract": abstract or "",
            "year": item["year"],
            "journal": item["journal"],
            "extraction_status": "ok" if abstract else "missing_abstract",
        })

        # Save progress every 50 papers
        if idx % 50 == 0:
            save_corpus(papers, partial=True)
            print(f"  Progress saved ({idx} papers so far)")

        time.sleep(0.3)  # stay within API rate limits

    return papers


# ── Save ─────────────────────────────────────────────────────────────────────
def save_corpus(papers: list[dict], partial: bool = False) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    corpus = {
        "corpus_name": "scopus_harvest",
        "topic_hint": "diabetes mellitus artificial intelligence",
        "partial": partial,
        "papers": papers,
    }
    OUTPUT_PATH.write_text(
        json.dumps(corpus, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=2024)
    parser.add_argument("--end_year",   type=int, default=2025)
    parser.add_argument("--max_papers", type=int, default=3200)
    args = parser.parse_args()

    api_key = load_api_key()
    print(f"API key loaded: {api_key[:8]}...\n")

    # Step 1: collect DOIs
    doi_list = fetch_dois(api_key, args.start_year, args.end_year, args.max_papers)

    if not doi_list:
        print("No papers found. Check your query or date range.")
        return

    # Step 2 & 3: fetch abstracts and build corpus
    print(f"\nFetching abstracts for {len(doi_list)} papers...")
    print("This will take a while. Progress is saved every 50 papers.\n")
    papers = build_corpus(doi_list, api_key)

    # Final save
    save_corpus(papers, partial=False)

    # Summary
    ok = sum(1 for p in papers if p["extraction_status"] == "ok")
    missing = sum(1 for p in papers if p["extraction_status"] == "missing_abstract")
    print(f"\nDone!")
    print(f"  Total:   {len(papers)}")
    print(f"  OK:      {ok}")
    print(f"  Missing: {missing}")
    print(f"  Saved:   {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
