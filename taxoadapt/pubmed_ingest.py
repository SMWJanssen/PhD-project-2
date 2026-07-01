"""
pubmed_ingest.py

Harvests papers from PubMed using NCBI E-utilities (free, no API key required).
Outputs a unified corpus JSON compatible with the TaxoAdapt pipeline.

Usage:
    python pubmed_ingest.py --start_year 2024 --end_year 2025 --max_papers 1000
"""

import json
import time
import argparse
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
OUTPUT_PATH = Path("datasets/pubmed_corpus.json")

QUERY = '(diabetes) AND ("artificial intelligence" OR "machine learning")'


def fetch_pmids(start_year: int, end_year: int, max_papers: int) -> list[str]:
    """Search PubMed and collect PMIDs (PubMed IDs)."""
    date_filter = f'AND ("{start_year}/01/01"[Date - Publication] : "{end_year}/12/31"[Date - Publication])'
    full_query = f"{QUERY} {date_filter}"

    print(f"Query: {full_query}")
    print(f"Fetching up to {max_papers} PMIDs...\n")

    pmids = []
    retstart = 0
    batch_size = 200

    while len(pmids) < max_papers:
        params = {
            "db": "pubmed",
            "term": full_query,
            "retstart": retstart,
            "retmax": min(batch_size, max_papers - len(pmids)),
            "retmode": "json",
        }
        try:
            r = requests.get(ESEARCH_URL, params=params, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  Search error at offset {retstart}: {e}")
            time.sleep(2)
            continue

        result = r.json().get("esearchresult", {})
        ids = result.get("idlist", [])
        total = int(result.get("count", 0))

        if not ids:
            break

        pmids.extend(ids)
        print(f"  Fetched {len(pmids)} / {min(max_papers, total)} PMIDs...", flush=True)

        retstart += batch_size
        if retstart >= total:
            break

        time.sleep(0.4)  # NCBI rate limit: max 3 req/s without API key

    print(f"\nTotal PMIDs collected: {len(pmids)}")
    return pmids[:max_papers]


def fetch_details_batch(pmids: list[str]) -> list[dict]:
    """Fetch title + abstract for a batch of PMIDs via efetch (XML)."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    try:
        r = requests.get(EFETCH_URL, params=params, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  Efetch error: {e}")
        return []

    papers = []
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError as e:
        print(f"  XML parse error: {e}")
        return []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else None

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        abstract_parts = article.findall(".//Abstract/AbstractText")
        abstract = " ".join("".join(p.itertext()).strip() for p in abstract_parts) if abstract_parts else ""

        year_el = article.find(".//PubDate/Year")
        year = year_el.text if year_el is not None else ""

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else ""

        papers.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "year": year,
            "journal": journal,
            "extraction_status": "ok" if (title and abstract) else "missing_abstract",
        })

    return papers


def build_corpus(pmids: list[str]) -> list[dict]:
    """Fetch details for all PMIDs in batches of 50."""
    all_papers = []
    batch_size = 50
    total = len(pmids)

    for i in range(0, total, batch_size):
        batch = pmids[i:i + batch_size]
        print(f"[{i+1}-{min(i+batch_size, total)}/{total}] Fetching batch...", flush=True)

        papers = fetch_details_batch(batch)
        all_papers.extend(papers)

        # Save progress every batch
        save_corpus(all_papers, partial=True)

        time.sleep(0.4)  # rate limit

    return all_papers


def save_corpus(papers: list[dict], partial: bool = False) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Renumber paper_ids consistently
    formatted_papers = []
    for idx, p in enumerate(papers, start=1):
        formatted_papers.append({
            "paper_id": f"pubmed_{idx:04d}",
            "source_file": p.get("pmid", ""),
            "title": p.get("title", ""),
            "abstract": p.get("abstract", ""),
            "year": p.get("year", ""),
            "journal": p.get("journal", ""),
            "extraction_status": p.get("extraction_status", "missing_abstract"),
        })

    corpus = {
        "corpus_name": "pubmed_harvest",
        "topic_hint": "diabetes mellitus artificial intelligence",
        "partial": partial,
        "papers": formatted_papers,
    }
    OUTPUT_PATH.write_text(
        json.dumps(corpus, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=2024)
    parser.add_argument("--end_year",   type=int, default=2025)
    parser.add_argument("--max_papers", type=int, default=1000)
    args = parser.parse_args()

    pmids = fetch_pmids(args.start_year, args.end_year, args.max_papers)

    if not pmids:
        print("No papers found. Check your query or date range.")
        return

    print(f"\nFetching details for {len(pmids)} papers...")
    papers = build_corpus(pmids)

    save_corpus(papers, partial=False)

    ok = sum(1 for p in papers if p["extraction_status"] == "ok")
    missing = sum(1 for p in papers if p["extraction_status"] == "missing_abstract")
    print(f"\nDone!")
    print(f"  Total:   {len(papers)}")
    print(f"  OK:      {ok}")
    print(f"  Missing: {missing}")
    print(f"  Saved:   {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
