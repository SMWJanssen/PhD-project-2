"""
study_filter.py

Shared logic for filtering out non-primary-study papers (reviews, surveys,
meta-analyses, editorials, retractions, perspectives) from a harvested
corpus. Used by both pubmed_ingest.py and openalex_ingest.py so the
filtering rules stay consistent across data sources.
"""

import re

# Title phrases that strongly indicate non-primary-study content.
TITLE_PATTERNS = [
    r"\bsystematic review\b",
    r"\bliterature review\b",
    r"\bnarrative review\b",
    r"\bscoping review\b",
    r"\bumbrella review\b",
    r"\ba review\b",
    r"\breview of\b",
    r"\breview on\b",
    r"\bmeta-analysis\b",
    r"\bmeta analysis\b",
    r"\bsurvey of\b",
    r"\ba survey\b",
    r"\bcomprehensive survey\b",
    r"\bexhaustive survey\b",
    r"\beditorial\b",
    r"\bcommentary\b",
    r"\bletter to the editor\b",
    r"\berratum\b",
    r"\bcorrigendum\b",
    r"\bretraction\b",
    r"\bretracted\b",
    r"^perspective:",
    r"^opinion:",
    r"^viewpoint:",
    r"\bguideline[s]?\b",
    r"\bconsensus statement\b",
    r"\bconsensus paper\b",
    r"\breporting guideline\b",
    r"\brecommendation[s]?\b",
    r"\bposition paper\b",
    r"\bwhite paper\b",
    r"\bstudy protocol\b",
    r"\bprotocol for\b",
    r"\bresponse to\b",
    r"\breply to\b",
    r"\bclinical practice guideline\b",
    r"\bstatement\b",
    r"\bchecklist\b",
]

# Abstract opening phrases that strongly indicate review/survey content,
# even when the title doesn't mention it explicitly. Checked only against
# the FIRST ~250 characters of the abstract (where authors typically
# state the paper's purpose).
ABSTRACT_OPENING_PATTERNS = [
    r"\bthis review\b",
    r"\bthis paper reviews\b",
    r"\bthis article reviews\b",
    r"\bthis study reviews\b",
    r"\bwe review\b",
    r"\bthis survey\b",
    r"\bthis paper surveys\b",
    r"\bthis article surveys\b",
    r"\boffered an? (exhaustive|comprehensive) survey\b",
    r"\bwe (comprehensively )?summarize\b",
    r"\bwe summarized\b",
    r"\bthis (paper|article|study) (elaborates|discusses|highlights|summarizes)\b",
    r"\bin this review\b",
    r"\bprovides? an overview\b",
    r"\bwe provide an overview\b",
    r"\bthis narrative review\b",
    r"\bthis (guideline|statement|consensus|checklist)\b",
    r"\bwe (developed|present|propose) (a |an |the )?(guideline|statement|checklist|framework|consensus)\b",
    r"\bthis (article|paper) (describes|presents) (the development|an update|a framework)\b",
]

_TITLE_RE = re.compile("|".join(TITLE_PATTERNS), re.IGNORECASE)
_ABSTRACT_RE = re.compile("|".join(ABSTRACT_OPENING_PATTERNS), re.IGNORECASE)


def is_primary_study(title: str, abstract: str, is_retracted: bool = False,
                      type_crossref: str = "", source_type: str = "") -> bool:
    """
    Returns True if the paper looks like a primary study (original research),
    False if it looks like a review, survey, meta-analysis, editorial,
    retraction, or preprint.

    Args:
        title: paper title
        abstract: paper abstract text
        is_retracted: explicit retraction flag from the source API, if available
        type_crossref: Crossref-style type string, if available
                        (e.g. "journal-article", "posted-content", "review")
        source_type: source type string, if available
                      (e.g. "repository" for preprint servers)
    """
    if is_retracted:
        return False

    if title and _TITLE_RE.search(title):
        return False

    # Check only the opening of the abstract — authors state purpose early
    abstract_opening = (abstract or "")[:300]
    if abstract_opening and _ABSTRACT_RE.search(abstract_opening):
        return False

    if type_crossref:
        allowed = {"journal-article", "proceedings-article"}
        if type_crossref.lower() not in allowed:
            return False

    if source_type == "repository":
        return False  # preprint servers, institutional repositories

    return True


def filter_corpus(papers: list[dict]) -> tuple[list[dict], int]:
    """
    Filters a list of paper dicts (each with at least 'title' and 'abstract'
    keys) down to primary studies only.

    Returns (filtered_papers, num_skipped).
    """
    kept = []
    skipped = 0

    for paper in papers:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        is_retracted = paper.get("is_retracted", False)
        type_crossref = paper.get("type_crossref", "")
        source_type = paper.get("source_type", "")

        if is_primary_study(title, abstract, is_retracted, type_crossref, source_type):
            kept.append(paper)
        else:
            skipped += 1

    return kept, skipped
