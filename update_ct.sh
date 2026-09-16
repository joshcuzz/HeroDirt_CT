#!/bin/bash

set -euo pipefail

# ============================================================
# HERO DIRT CONNECTICUT DAILY UPDATE
# ============================================================

ROOT="$HOME/HeroDirt_CT"
PYTHON="$HOME/miniconda3/envs/herodirt/bin/python"

cd "$ROOT"

echo
echo "============================================================"
echo " HERO DIRT CT DAILY UPDATE"
echo "============================================================"
echo "Started: $(date)"
echo


# ============================================================
# 1. UPDATE OBSERVATIONS / FORECASTS
# ============================================================

echo
echo "------------------------------------------------------------"
echo "1/6  Updating MRMS..."
echo "------------------------------------------------------------"

"$PYTHON" src/get_mrms.py


echo
echo "------------------------------------------------------------"
echo "2/6  Updating SMAP..."
echo "------------------------------------------------------------"

"$PYTHON" src/get_smap.py


echo
echo "------------------------------------------------------------"
echo "3/6  Updating NWS forecast..."
echo "------------------------------------------------------------"

"$PYTHON" src/get_nws.py


# ============================================================
# 2. RUN V2 MODEL
# ============================================================

echo
echo "------------------------------------------------------------"
echo "4/6  Running Hero Dirt v2 beta..."
echo "------------------------------------------------------------"

"$PYTHON" src/run_soil_model_v2.py


# ============================================================
# 3. BUILD FORECAST MAP
# ============================================================

echo
echo "------------------------------------------------------------"
echo "5/6  Building Connecticut trail forecast..."
echo "------------------------------------------------------------"

"$PYTHON" src/build_trail_map.py


# ============================================================
# 4. UPDATE GITHUB PAGES FILES
# ============================================================

echo
echo "------------------------------------------------------------"
echo "6/6  Updating GitHub Pages..."
echo "------------------------------------------------------------"

mkdir -p docs/data

cp web/HeroDirt_trails.html docs/index.html
cp web/data/*.geojson docs/data/


# ============================================================
# 5. COMMIT + PUSH
# ============================================================

git add docs/index.html
git add -f docs/data/*.geojson

if git diff --cached --quiet; then

    echo
    echo "No website changes to commit."

else

    git commit \
        -m "Update CT forecast $(date '+%Y-%m-%d %H:%M')"

    git push

fi


echo
echo "============================================================"
echo " HERO DIRT CT UPDATE COMPLETE"
echo "============================================================"
echo "Finished: $(date)"
echo
echo "Live site:"
echo "https://joshcuzz.github.io/HeroDirt_CT/"
echo
