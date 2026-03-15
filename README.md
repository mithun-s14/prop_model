---
title: Prop Model
emoji: 📚
colorFrom: purple
colorTo: gray
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
license: mit
---

# NBA Predictive Analytics Platform

An NBA player stat prediction tool that uses an ensemble of six machine learning models to forecast Points, Assists, and Rebounds for the day's games based on given game context and historical performance data.

## Features

- **Ensemble Predictions** — Averages predictions from Bayesian Ridge, Gradient Boost, LightGBM, Linear Regression, Random Forest, and XGBoost models
- **Game Context Inputs** — Factors in opponent, home/away status, spread, and total
- **Automated Data Scraping** — Fetches player stats, defensive ratings, and usage rates from NBA sources (NBA.com, Basketball Reference)
- **Dark-Themed UI** — Clean Gradio interface for quick lookups

## Tech Stack

- **ML / Data**: scikit-learn, XGBoost, LightGBM, pandas, NumPy
- **Frontend**: Gradio, Tailwind CSS
- **Data Collection**: Custom scrapers using Scrapling

## Project Structure

```
prop_model/
├── app.py                  # Gradio web application ()
├── backend/
│   ├── model.py            # ML ensemble pipeline
│   ├── api.py              # API endpoints
│   ├── nba_data_scraper.py # Player & game data scraper
│   ├── nba_defense_scraper.py
│   ├── nba_usage_rate_scraper.py
│   ├── update_all_data.sh  # Refresh all cached data
│   ├── tests/              # Pytest test suite
│   └── cached_*.csv/json   # Scraped data cache
├── frontend/               # UI styling (Tailwind CSS), was used when I used Render + Vercel to deploy this app, then found out Render didn't have enough RAM.
├── requirements.txt
└── packages.txt
```

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

The app launches at `http://localhost:7860`.

### Usage

1. Enter a player name (e.g., "LeBron James")
2. Select a stat — Points, Assists, or Rebounds
3. Set the game spread and total
4. Click **Generate Prediction** to see ensemble and individual model results

## Running Tests

```bash
pytest
```

### To-Do List
- [ ] Autocomplete for player names
- [ ] Allow users to enter opponent information so users can analyze players in future games
