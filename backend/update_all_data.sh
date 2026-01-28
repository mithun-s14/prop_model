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