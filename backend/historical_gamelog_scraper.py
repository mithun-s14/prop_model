"""
Backfill historical NBA player game logs from Basketball Reference.

Motivation: the evaluation chain (evaluate.py -> backtest_model.py ->
train_real_model.py -> train_engineered_model.py) bottomed out on data volume,
not modelling. cached_player_gamelogs.csv covers a single month (~4.9k rows,
~10 games per player), which left the engineered model's 2% MAE gain over the
naive baseline statistically indistinguishable from noise (p=0.062). This
module fetches whole seasons so those comparisons can actually resolve.

Strategy: one request per player-season against the player game log page,
rather than one request per box score. A season has ~1230 games but only ~580
players, so per-player logs cover the same games in roughly half the requests.
A season costs ~580 requests (~35 min at the default delay).

Politeness: Basketball Reference allows about 20 requests/minute. DEFAULT_DELAY
is 3.5s to stay under that, matching nba_data_scraper.py. Do not lower it.

Resumability: progress is checkpointed after every player and rows are appended
to the output CSV as they arrive, so an interrupted run resumes where it left
off instead of re-fetching. Re-running a completed season is a no-op.

Usage:
    python -m backend.historical_gamelog_scraper --seasons 2026 2025 2024
    python -m backend.historical_gamelog_scraper --seasons 2026 --limit 5   # smoke test
    python -m backend.historical_gamelog_scraper --merge                    # fold into the cache
"""
import argparse
import json
import os
import re
import time
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BACKEND_DIR, 'cached_player_gamelogs.csv')
BACKFILL_PATH = os.path.join(BACKEND_DIR, 'historical_player_gamelogs.csv')
CHECKPOINT_PATH = os.path.join(BACKEND_DIR, '.historical_scrape_checkpoint.json')

BASE_URL = 'https://www.basketball-reference.com'
DEFAULT_DELAY = 3.5  # seconds; Basketball Reference allows ~20 req/min

# Column order of cached_player_gamelogs.csv, which downstream code expects.
SCHEMA = [
    'PLAYER_NAME', 'Player_ID', 'GAME_DATE', 'team', 'game_id', 'MIN',
    'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
    'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
]

# Basketball Reference data-stat -> our column name
_STAT_MAP = {
    'fg': 'FGM', 'fga': 'FGA', 'fg_pct': 'FG_PCT',
    'fg3': 'FG3M', 'fg3a': 'FG3A', 'fg3_pct': 'FG3_PCT',
    'ft': 'FTM', 'fta': 'FTA', 'ft_pct': 'FT_PCT',
    'orb': 'OREB', 'drb': 'DREB', 'trb': 'REB',
    'ast': 'AST', 'stl': 'STL', 'blk': 'BLK', 'tov': 'TOV', 'pf': 'PF', 'pts': 'PTS',
}
_INT_COLUMNS = ['FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
                'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']


def _fetch(url, delay=DEFAULT_DELAY, max_retries=3):
    """GET a Basketball Reference page as HTML, rate-limited and retried."""
    from scrapling.fetchers import Fetcher

    for attempt in range(max_retries):
        try:
            response = Fetcher.get(url, stealthy_headers=True, timeout=30)
            if response.status == 404:
                time.sleep(delay)
                return None  # player didn't play that season
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status} for {url}")
            html = response.body if isinstance(getattr(response, 'body', None), str) else str(response)
            time.sleep(delay)
            return html
        except Exception as exc:  # noqa: BLE001 - retry any transport/parse failure
            print(f"    attempt {attempt + 1} failed: {exc}")
            if attempt == max_retries - 1:
                raise
            time.sleep(delay * (attempt + 2))
    return None


def parse_minutes(value):
    """'34:12' -> 34.2 decimal minutes. Returns None if unparseable."""
    if not value:
        return None
    match = re.fullmatch(r'(\d+):(\d{1,2})', str(value).strip())
    if not match:
        return None
    minutes, seconds = int(match.group(1)), int(match.group(2))
    return round(minutes + seconds / 60.0, 1)


def format_game_date(iso_date):
    """'2025-10-22' -> 'Wed, Oct 22, 2025', matching the existing cache format."""
    parsed = datetime.strptime(str(iso_date).strip(), '%Y-%m-%d')
    return f"{parsed:%a, %b} {parsed.day}, {parsed.year}"


def build_game_id(iso_date, team, is_home, opponent):
    """
    Basketball Reference box score IDs are YYYYMMDD + '0' + HOME team abbr.
    Downstream code (features.py) reads the home team back off this suffix,
    so it must be the home side, not the player's team.
    """
    home_team = team if is_home else opponent
    compact = str(iso_date).strip().replace('-', '')
    return f"{compact}0{home_team}"


def parse_gamelog_html(html, player_id, player_name, table_id='player_game_log_reg'):
    """
    Parse one player's season game log into rows matching SCHEMA.

    Skips header separator rows and DNP/inactive games (no minutes played),
    which carry no performance to learn from and would otherwise pollute
    rolling averages with spurious zeros.
    """
    soup = BeautifulSoup(html, 'lxml')
    table = soup.find('table', id=table_id)
    if table is None or table.find('tbody') is None:
        return []

    rows = []
    for tr in table.find('tbody').find_all('tr'):
        if 'thead' in (tr.get('class') or []):
            continue

        cells = {c.get('data-stat'): c.get_text(strip=True) for c in tr.find_all(['th', 'td'])}

        minutes = parse_minutes(cells.get('mp'))
        if minutes is None:
            continue  # DNP / inactive / suspended

        iso_date = cells.get('date', '')
        team = cells.get('team_name_abbr', '')
        opponent = cells.get('opp_name_abbr', '')
        if not (iso_date and team and opponent):
            continue

        is_home = cells.get('game_location', '') != '@'

        row = {
            'PLAYER_NAME': player_name,
            'Player_ID': player_id,
            'GAME_DATE': format_game_date(iso_date),
            'team': team,
            'game_id': build_game_id(iso_date, team, is_home, opponent),
            'MIN': minutes,
        }
        for stat, column in _STAT_MAP.items():
            raw = cells.get(stat, '')
            try:
                row[column] = float(raw) if raw not in ('', None) else 0.0
            except ValueError:
                row[column] = 0.0
        for column in _INT_COLUMNS:
            row[column] = int(row[column])

        rows.append(row)

    return rows


def get_season_players(season, delay=DEFAULT_DELAY):
    """
    All players who appeared in a season, from its totals page (one request).
    Returns a sorted list of (player_id, player_name).
    """
    html = _fetch(f"{BASE_URL}/leagues/NBA_{season}_totals.html", delay=delay)
    if not html:
        return []

    soup = BeautifulSoup(html, 'lxml')
    table = soup.find('table', id='totals_stats')
    if table is None or table.find('tbody') is None:
        return []

    players = {}
    for link in table.find('tbody').find_all('a', href=re.compile(r'/players/./\w+\.html')):
        player_id = re.search(r'/players/./(\w+)\.html', link['href']).group(1)
        players[player_id] = link.get_text(strip=True)
    return sorted(players.items())


def scrape_player_season(player_id, player_name, season, delay=DEFAULT_DELAY):
    """Fetch and parse one player's game log for one season."""
    url = f"{BASE_URL}/players/{player_id[0]}/{player_id}/gamelog/{season}"
    html = _fetch(url, delay=delay)
    if not html:
        return []
    return parse_gamelog_html(html, player_id, player_name)


def _load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as handle:
            return set(json.load(handle).get('completed', []))
    return set()


def _save_checkpoint(path, completed):
    with open(path, 'w') as handle:
        json.dump({'completed': sorted(completed), 'updated': datetime.now().isoformat()}, handle)


def _append_rows(path, rows):
    """Append rows to the backfill CSV, writing a header only on first write."""
    frame = pd.DataFrame(rows, columns=SCHEMA)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    frame.to_csv(path, mode='a', header=write_header, index=False)


def backfill(seasons, out_path=BACKFILL_PATH, checkpoint_path=CHECKPOINT_PATH,
             limit=None, delay=DEFAULT_DELAY):
    """
    Scrape whole seasons of player game logs, resuming from any prior run.

    `limit` caps players per season (for smoke tests). Progress is checkpointed
    per player, so interrupting and re-running continues rather than restarting.
    """
    completed = _load_checkpoint(checkpoint_path)
    total_rows = 0

    for season in seasons:
        print(f"\n=== season {season} ===")
        players = get_season_players(season, delay=delay)
        if limit:
            players = players[:limit]
        print(f"  {len(players)} players to consider")

        pending = [(pid, name) for pid, name in players if f"{season}:{pid}" not in completed]
        print(f"  {len(pending)} not yet scraped "
              f"({len(players) - len(pending)} already done)")
        if pending:
            estimate = len(pending) * delay / 60
            print(f"  estimated time: {estimate:.0f} min at {delay}s/request")

        for index, (player_id, player_name) in enumerate(pending, start=1):
            try:
                rows = scrape_player_season(player_id, player_name, season, delay=delay)
            except Exception as exc:  # noqa: BLE001 - keep going past one bad player
                print(f"  [{index}/{len(pending)}] {player_name}: FAILED ({exc})")
                continue

            if rows:
                _append_rows(out_path, rows)
                total_rows += len(rows)
            completed.add(f"{season}:{player_id}")
            _save_checkpoint(checkpoint_path, completed)

            if index % 25 == 0 or index == len(pending):
                print(f"  [{index}/{len(pending)}] {player_name}: "
                      f"{len(rows)} games (total rows this run: {total_rows})")

    print(f"\nbackfill wrote {total_rows} rows to {out_path}")
    return total_rows


def merge_into_cache(backfill_path=BACKFILL_PATH, cache_path=CACHE_PATH, output_path=None):
    """
    Merge backfilled rows into the cache, de-duplicating on (Player_ID, game_id).

    Existing cached rows win over backfilled ones for the same player-game, so
    a re-scrape never silently rewrites data already in use.
    """
    output_path = output_path or cache_path
    if not os.path.exists(backfill_path):
        raise FileNotFoundError(f"no backfill file at {backfill_path}; run the scrape first")

    backfilled = pd.read_csv(backfill_path)
    existing = pd.read_csv(cache_path) if os.path.exists(cache_path) else pd.DataFrame(columns=SCHEMA)

    before = len(existing)
    # Concat only non-empty frames: an all-NA frame would coerce dtypes.
    parts = [frame for frame in (existing, backfilled) if not frame.empty]
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SCHEMA)
    combined = combined.drop_duplicates(subset=['Player_ID', 'game_id'], keep='first')

    combined['_sort_date'] = pd.to_datetime(combined['GAME_DATE'], format='mixed', errors='coerce')
    combined = (combined.sort_values(['_sort_date', 'game_id', 'PLAYER_NAME'])
                        .drop(columns='_sort_date'))

    combined[SCHEMA].to_csv(output_path, index=False)
    print(f"merged: {before} cached + {len(backfilled)} backfilled "
          f"-> {len(combined)} unique rows -> {output_path}")
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seasons', type=int, nargs='+', default=[2026],
                        help='season end years, e.g. 2026 = the 2025-26 season')
    parser.add_argument('--limit', type=int, default=None,
                        help='cap players per season (smoke test)')
    parser.add_argument('--delay', type=float, default=DEFAULT_DELAY,
                        help=f'seconds between requests (default {DEFAULT_DELAY}; do not lower)')
    parser.add_argument('--out', default=BACKFILL_PATH, help='backfill CSV path')
    parser.add_argument('--merge', action='store_true',
                        help='merge the backfill into cached_player_gamelogs.csv and exit')
    args = parser.parse_args()

    if args.merge:
        merge_into_cache(backfill_path=args.out)
        return

    backfill(args.seasons, out_path=args.out, limit=args.limit, delay=args.delay)


if __name__ == '__main__':
    main()
