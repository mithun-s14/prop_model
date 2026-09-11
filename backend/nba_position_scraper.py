"""
Scrape player positions from Basketball Reference into players_positions.csv.

Why this exists: `players_positions.csv` was read by model.py and written by
nothing in the repo -- an orphaned file, most likely left behind by the deleted
odds scraper. Its rows never gain a new rookie or signing, so coverage decays
every season while appearing fine. This module regenerates it.

Output carries `player_id` alongside the display name, so accented names
(Luka Doncic, Nikola Jokic) stop depending on exact string matching.

Usage:
    python -m backend.nba_position_scraper --season 2026
"""
import argparse
import os
import re

import pandas as pd
from bs4 import BeautifulSoup

from backend.historical_gamelog_scraper import DEFAULT_DELAY, _fetch

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(BACKEND_DIR, 'data', 'players_positions.csv')
BASE_URL = 'https://www.basketball-reference.com'

VALID_POSITIONS = ['PG', 'SG', 'SF', 'PF', 'C']

# Basketball Reference writes combination and single-letter positions; map them
# onto the five the rest of the codebase uses.
POSITION_MAP = {'G': 'SG', 'F': 'SF', 'GF': 'SG', 'FG': 'SF', 'FC': 'PF', 'CF': 'C'}


def normalize_position(raw):
    """'SG-SF' -> 'SG'; 'G' -> 'SG'; whitespace stripped; unknown -> 'UNK'."""
    text = str(raw or '').strip()
    if not text:
        return 'UNK'
    primary = text.split('-')[0].strip().upper()
    primary = POSITION_MAP.get(primary, primary)
    return primary if primary in VALID_POSITIONS else 'UNK'


def scrape_positions(season, delay=DEFAULT_DELAY):
    """All players in a season with their positions, from the totals page (one request)."""
    html = _fetch(f'{BASE_URL}/leagues/NBA_{season}_totals.html', delay=delay)
    if not html:
        return pd.DataFrame(columns=['player_id', 'Player', 'Position'])

    soup = BeautifulSoup(html, 'lxml')
    table = soup.find('table', id='totals_stats')
    if table is None or table.find('tbody') is None:
        return pd.DataFrame(columns=['player_id', 'Player', 'Position'])

    rows = {}
    for tr in table.find('tbody').find_all('tr'):
        link = tr.find('a', href=re.compile(r'/players/./\w+\.html'))
        if link is None:
            continue
        player_id = re.search(r'/players/./(\w+)\.html', link['href']).group(1)
        cell = tr.find('td', {'data-stat': 'pos'}) or tr.find('td', {'data-stat': 'position'})
        position = normalize_position(cell.get_text(strip=True) if cell else '')
        # A player traded mid-season appears more than once; the first row is
        # their combined-season line.
        rows.setdefault(player_id, {'player_id': player_id,
                                    'Player': link.get_text(strip=True),
                                    'Position': position})
    return pd.DataFrame(list(rows.values()))


def update_positions_file(season, path=POSITIONS_PATH, delay=DEFAULT_DELAY):
    """
    Regenerate players_positions.csv for a season, preserving prior players.

    A failed scrape leaves the existing file untouched rather than truncating it
    to a header: a stale file is far better than an empty one, since every
    downstream position lookup would otherwise fall back to 'UNK' at once.
    """
    scraped = scrape_positions(season, delay=delay)
    if scraped.empty:
        print(f"  [nba_position_scraper] scrape returned no rows for {season}; "
              f"leaving {os.path.basename(path)} unchanged")
        return None

    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing.columns = [c.strip() for c in existing.columns]
        for column in ('Player', 'Position'):
            if column in existing.columns:
                existing[column] = existing[column].astype(str).str.strip()
        if 'Position' in existing.columns:
            existing['Position'] = existing['Position'].map(normalize_position)
        if 'player_id' not in existing.columns:
            existing['player_id'] = None
        # Scraped rows win; players from earlier seasons are retained.
        merged = pd.concat([scraped, existing[['player_id', 'Player', 'Position']]],
                           ignore_index=True).drop_duplicates('Player', keep='first')
    else:
        merged = scraped

    merged = merged[['player_id', 'Player', 'Position']].sort_values('Player')
    merged.to_csv(path, index=False)
    print(f"  [nba_position_scraper] wrote {len(merged)} players to {path}")
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season', type=int, default=2026,
                        help='season end year, e.g. 2026 = the 2025-26 season')
    parser.add_argument('--out', default=POSITIONS_PATH)
    parser.add_argument('--delay', type=float, default=DEFAULT_DELAY)
    args = parser.parse_args()
    update_positions_file(args.season, path=args.out, delay=args.delay)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
