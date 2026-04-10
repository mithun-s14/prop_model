"""
NBA Odds Scraper - Fetches today's spreads and totals via OddsHarvester
Runs daily via GitHub Actions
"""

import os
import json
import subprocess
import sys
import pandas as pd
from datetime import datetime


OUTPUT_JSON = "nba_odds_today.json"
OUTPUT_CSV = "nba_odds_today.csv"


def get_today_str() -> str:
    """Return today's date as YYYYMMDD."""
    return datetime.now().strftime("%Y%m%d")


def run_oddsharvester(date_str: str, output_path: str) -> bool:
    """
    Invoke OddsHarvester CLI to scrape NBA spreads and totals for the given date.
    Returns True on success, False on failure.
    """
    cmd = [
        sys.executable, "-m", "oddsharvester", "upcoming",
        "-s", "basketball",
        "-d", date_str,
        "-m", "spreads,totals",
        "--headless",
        "--storage-format", "json",
        "--file-path", output_path,
    ]

    print(f"Running OddsHarvester for date {date_str}...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"OddsHarvester exited with code {result.returncode}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print("OddsHarvester timed out after 300 seconds")
        return False
    except FileNotFoundError:
        print("OddsHarvester not found. Ensure it is installed: pip install oddsharvester")
        return False


def parse_odds_output(raw_path: str) -> list[dict]:
    """
    Parse OddsHarvester JSON output into a flat list of game odds records.

    OddsHarvester returns a list of match objects. Each match may have nested
    odds under 'markets' with keys 'spreads' and 'totals'. We flatten each
    match into one record per game with the best available spread and total.
    """
    if not os.path.exists(raw_path):
        print(f"Raw odds file not found: {raw_path}")
        return []

    with open(raw_path) as f:
        data = json.load(f)

    if not data:
        print("No odds data returned by OddsHarvester")
        return []

    # Handle both a bare list and a dict wrapper {"data": [...]}
    if isinstance(data, dict):
        matches = data.get("data", data.get("matches", data.get("events", [])))
    else:
        matches = data

    records = []
    for match in matches:
        record = _flatten_match(match)
        if record:
            records.append(record)

    return records


def _flatten_match(match: dict) -> dict | None:
    """Extract a single normalised odds record from a raw OddsHarvester match."""
    home = match.get("home_team") or match.get("home") or match.get("homeTeam", "")
    away = match.get("away_team") or match.get("away") or match.get("awayTeam", "")
    if not home or not away:
        return None

    record: dict = {
        "home_team": home,
        "away_team": away,
        "date": match.get("date") or match.get("start_time") or match.get("datetime", ""),
        "sport": "NBA",
        # spread fields
        "home_spread": None,
        "home_spread_odds": None,
        "away_spread": None,
        "away_spread_odds": None,
        # total fields
        "total_line": None,
        "over_odds": None,
        "under_odds": None,
    }

    markets = match.get("markets") or match.get("odds") or {}

    # --- spreads ---
    spreads = markets.get("spreads") or markets.get("spread") or []
    if spreads:
        best = spreads[0] if isinstance(spreads, list) else spreads
        record["home_spread"] = best.get("home_handicap") or best.get("home_line") or best.get("handicap")
        record["home_spread_odds"] = best.get("home_odds") or best.get("odds_home")
        record["away_spread"] = best.get("away_handicap") or best.get("away_line")
        record["away_spread_odds"] = best.get("away_odds") or best.get("odds_away")

    # --- totals ---
    totals = markets.get("totals") or markets.get("total") or []
    if totals:
        best = totals[0] if isinstance(totals, list) else totals
        record["total_line"] = best.get("total") or best.get("line") or best.get("total_line")
        record["over_odds"] = best.get("over_odds") or best.get("over")
        record["under_odds"] = best.get("under_odds") or best.get("under")

    return record


def save_odds(records: list[dict], current_dir: str) -> None:
    """Persist odds records to JSON and CSV."""
    json_path = os.path.join(current_dir, OUTPUT_JSON)
    csv_path = os.path.join(current_dir, OUTPUT_CSV)

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Saved {len(records)} game odds to {json_path}")

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Saved odds CSV to {csv_path}")


def main() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    date_str = get_today_str()
    raw_path = os.path.join(current_dir, f"odds_raw_{date_str}.json")

    print("=" * 60)
    print(f"NBA ODDS SCRAPER  (OddsHarvester) — {date_str}")
    print("=" * 60)

    success = run_oddsharvester(date_str, raw_path)
    if not success:
        print("OddsHarvester run failed; saving empty files")
        save_odds([], current_dir)
        return

    records = parse_odds_output(raw_path)
    save_odds(records, current_dir)

    # Clean up the raw temp file
    if os.path.exists(raw_path):
        os.remove(raw_path)

    print(f"Done — {len(records)} NBA games scraped for {date_str}")


if __name__ == "__main__":
    main()
