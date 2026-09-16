#!/usr/bin/env python3

import requests

URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"

query = """
SELECT TOP 5
    areasymbol,
    musym,
    muname,
    mukey
FROM mapunit
INNER JOIN legend
    ON mapunit.lkey = legend.lkey
WHERE areasymbol LIKE 'CA%'
"""

payload = {
    "SERVICE": "query",
    "REQUEST": "query",
    "QUERY": query,
    "FORMAT": "JSON+COLUMNNAME"
}

print("Contacting USDA Soil Data Access...")

r = requests.post(
    URL,
    data=payload,
    timeout=60
)

print("HTTP status:", r.status_code)

r.raise_for_status()

data = r.json()

for row in data["Table"]:
    print(row)
