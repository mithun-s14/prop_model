#!/bin/bash

echo "===================================="
echo "Updating ALL NBA Data Locally"
echo "===================================="

cd backend

echo ""
echo "1. Running NBA API scraper..."
python nba_data_scraper.py

echo ""
echo "2. Running usage rate scraper..."
python nba_usage_rate_scraper.py

echo ""
echo "3. Running position scraper..."
# Package-style module (imports `backend.historical_gamelog_scraper`), so it
# must run from the repo root -- from inside backend/ it fails with
# "No module named 'backend'".
(cd .. && python -m backend.nba_position_scraper --season 2026)
