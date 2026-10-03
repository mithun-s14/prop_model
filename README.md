---
title: NBA Predictive Analytics Platform
emoji: 📊
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.2.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: NBA player prop predictions
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

## Project Structure

```
app.py                  Gradio entry point
backend/
  data/                 All cached data files (CSV / JSON / Excel)
  *.py                  Scrapers, feature engineering, models
  tests/                Pytest suite
frontend/               React UI
```

All scrapers write into `backend/data/`, and every loader reads from it.

## Running Tests

```bash
pytest
```
