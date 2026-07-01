import requests
url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
params = {"db": "pubmed", "term": "diabetes mellitus AND (prediction OR diagnosis OR screening OR treatment OR complication OR monitoring)", "retmax": 0, "retmode": "json"}
r = requests.get(url, params=params, timeout=30)
print("Corpus 1 (DM clinical):", r.json()["esearchresult"]["count"])

params2 = {"db": "pubmed", "term": "(machine learning OR deep learning OR artificial intelligence) AND (clinical OR medical OR patient OR disease)", "retmax": 0, "retmode": "json"}
r2 = requests.get(url, params=params2, timeout=30)
print("Corpus 2 (healthcare AI):", r2.json()["esearchresult"]["count"])
