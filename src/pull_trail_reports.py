import os
import requests
import pandas as pd

SUPABASE_URL = "https://ldyeqmgdtaiaeciakmvi.supabase.co"

SUPABASE_KEY = os.environ.get(
    "HERODIRT_SUPABASE_SECRET"
)

if not SUPABASE_KEY:
    raise RuntimeError(
        "HERODIRT_SUPABASE_SECRET is not set"
    )

OUTDIR = "output/validation"
OUTFILE = os.path.join(
    OUTDIR,
    "trail_reports.csv"
)

os.makedirs(
    OUTDIR,
    exist_ok=True
)

url = (
    f"{SUPABASE_URL}"
    "/rest/v1/trail_reports"
)

headers = {
    "apikey": SUPABASE_KEY,
}

params = {
    "select": "*",
    "order": "submitted_utc.asc",
}

r = requests.get(
    url,
    headers=headers,
    params=params,
    timeout=30,
)

if not r.ok:
    print(r.status_code)
    print(r.text)

r.raise_for_status()

rows = r.json()

df = pd.DataFrame(rows)

if len(df) == 0:
    print("No trail reports found.")

else:
    df.to_csv(
        OUTFILE,
        index=False,
    )

    print(
        f"Downloaded {len(df)} "
        "trail reports"
    )
    print(
        f"Saved: {OUTFILE}"
    )
    print()
    print(df.tail())
