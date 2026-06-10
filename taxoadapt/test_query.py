import requests, re

key = open(r'C:\Users\janss266\.config\pybliometrics.cfg').read()
api_key = re.search(r'APIKey\s*=\s*(\S+)', key).group(1)

url = 'https://api.elsevier.com/content/search/scopus'
query = 'TITLE-ABS-KEY("diabetes mellitus" AND (prediction OR classification OR detection OR monitoring OR forecasting OR risk OR outcome)) AND PUBYEAR > 2023 AND PUBYEAR < 2026 AND DOCTYPE(ar OR re)'

params = {'query': query, 'count': 1, 'field': 'title'}
headers = {'X-ELS-APIKey': api_key, 'Accept': 'application/json'}
r = requests.get(url, params=params, headers=headers, timeout=30)
total = r.json().get('search-results', {}).get('opensearch:totalResults', 'error')
print(f'Total papers found: {total}')