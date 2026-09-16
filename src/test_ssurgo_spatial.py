#!/usr/bin/env python3

import requests

url = "https://sdmdataaccess.sc.egov.usda.gov/Spatial/SDMNAD83Geographic.wfs"

params = {
    "SERVICE": "WFS",
    "VERSION": "1.1.0",
    "REQUEST": "GetCapabilities",
}

print("Contacting USDA SSURGO spatial service...")

r = requests.get(
    url,
    params=params,
    timeout=60,
)

print("HTTP status:", r.status_code)
print("Content type:", r.headers.get("content-type"))

if r.status_code != 200:
    print()
    print("Server response:")
    print(r.text[:2000])

r.raise_for_status()

print("Response length:", len(r.text))
print("Spatial service works.")
