import pandas as pd
from io import StringIO
import os
from datetime import datetime, timedelta
try:
    from scrapling.fetchers import StealthyFetcher
    _SCRAPING_AVAILABLE = True
except ImportError:
    _SCRAPING_AVAILABLE = False

_FANTASYPROS_URL = "https://www.fantasypros.com/nba/stats/avg-overall.php?days=15"
_CACHE_FILE = "fantasypros_last15_averages.csv"
_CACHE_MAX_AGE_HOURS = 24

_STATUS_TAGS = {'DTD', 'OUT', 'G-League', 'TWO-WAY', 'FA'}


def _clean_player_name(name):
    """
    Normalise a raw FantasyPros player name:
      - Remove team/position suffix:  'Luka Doncic (LAL - PG,SG)' → 'Luka Doncic'
      - Strip surrounding quotes:     '"Jayson Tatum"'             → 'Jayson Tatum'
      - Remove status tags:           'Jaylen Brown DTD'           → 'Jaylen Brown'
    """
    if not isinstance(name, str):
        return name
    # Remove (TEAM - POS) or (TEAM - POS,POS) suffix
    name = name.split('(')[0]
    # Strip surrounding quotes
    name = name.strip('"\'')
    # Remove trailing status tags
    tokens = name.split()
    tokens = [t for t in tokens if t not in _STATUS_TAGS]
    return ' '.join(tokens).strip()


def scrape_fantasypros_last15():
    """
    Scrape last-15-day averages for all NBA players from FantasyPros.
    Uses StealthyFetcher to bypass bot protection.
    Returns a cleaned DataFrame.
    """
    if not _SCRAPING_AVAILABLE:
        print("scrapling not available — cannot scrape. Load from cache instead.")
        return pd.DataFrame()
    print("Fetching FantasyPros last-15 averages...")
    page = StealthyFetcher.fetch(_FANTASYPROS_URL, headless=True, network_idle=True)

    try:
        tables = pd.read_html(StringIO(page.html_content))
    except ValueError:
        print("No table found on FantasyPros page.")
        return pd.DataFrame()

    if not tables:
        print("No table found on FantasyPros page.")
        return pd.DataFrame()

    df = tables[0]

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    print(f"Columns found: {list(df.columns)}")

    # Drop repeated header rows injected mid-table
    if 'Player' in df.columns:
        df = df[df['Player'] != 'Player'].reset_index(drop=True)
        df['Player'] = df['Player'].apply(_clean_player_name)

    # Drop rank column if present
    df = df.drop(columns=['RK', '#'], errors='ignore')

    # Coerce all stat columns to numeric
    for col in df.columns:
        if col not in ('Player', 'Team', 'POS'):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.drop_duplicates(subset='Player', keep='first').reset_index(drop=True)
    print(f"Total players scraped: {len(df)}")
    return df


def load_fantasypros_last15(force_refresh=False):
    """
    Load last-15 averages from cache, re-scraping only when fresh data is
    needed AND the scraping environment is available (browser binaries present).
    Falls back to stale cache rather than crashing when scraping is unavailable.
    """
    cache_exists = os.path.exists(_CACHE_FILE)

    if not force_refresh and cache_exists:
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(_CACHE_FILE))
        if file_age < timedelta(hours=_CACHE_MAX_AGE_HOURS):
            df = pd.read_csv(_CACHE_FILE)
            print(f"Loaded FantasyPros last-15 averages from cache ({len(df)} players)")
            return df
        print("FantasyPros cache is stale, attempting re-scrape...")

    try:
        df = scrape_fantasypros_last15()
        if not df.empty:
            df.to_csv(_CACHE_FILE, index=False)
            print(f"Saved FantasyPros last-15 averages to {_CACHE_FILE}")
            return df
    except Exception as e:
        print(f"Scraping failed ({e}), falling back to cached data.")

    # Scraping failed or unavailable — use whatever cache exists
    if cache_exists:
        df = pd.read_csv(_CACHE_FILE)
        print(f"Using stale cache ({len(df)} players)")
        return df

    print("No cache and scraping unavailable — returning empty DataFrame.")
    return pd.DataFrame()


def get_player_last15_features(player_name):
    """
    Look up a player's last-15 averages and return a feature dict
    compatible with the model's expected input keys.

    Returns a dict with keys: pts_roll_avg, reb_roll_avg, ast_roll_avg,
    min_roll_avg, fga_roll_avg, fg3a_roll_avg, fta_roll_avg,
    stl_roll_avg, blk_roll_avg, tov_roll_avg
    """
    df = load_fantasypros_last15()
    if df.empty:
        return {}

    name_clean = player_name.strip()

    # Exact match
    match = df[df['Player'].str.strip() == name_clean]

    # Partial match
    if match.empty:
        match = df[df['Player'].str.contains(name_clean, case=False, na=False)]

    # Last name match
    if match.empty:
        last_name = name_clean.split()[-1]
        match = df[df['Player'].str.contains(last_name, case=False, na=False)]

    if match.empty:
        print(f"Warning: '{player_name}' not found in FantasyPros last-15 data.")
        return {}

    row = match.iloc[0]
    print(f"Found FantasyPros last-15 stats for: {row['Player']}")

    return {
        'pts_roll_avg':  float(row.get('PPG', 0) or 0),
        'reb_roll_avg':  float(row.get('RPG', 0) or 0),
        'ast_roll_avg':  float(row.get('APG', 0) or 0),
        'min_roll_avg':  float(row.get('MPG', 0) or 0),
        'fga_roll_avg':  float(row.get('FGA', 0) or 0),
        'fg3a_roll_avg': float(row.get('3PA', 0) or 0),
        'fta_roll_avg':  float(row.get('FTA', 0) or 0),
        'stl_roll_avg':  float(row.get('SPG', 0) or 0),
        'blk_roll_avg':  float(row.get('BPG', 0) or 0),
        'tov_roll_avg':  float(row.get('TOV', 0) or 0),
    }


if __name__ == "__main__":
    df = scrape_fantasypros_last15()
    if not df.empty:
        df.to_csv(_CACHE_FILE, index=False)
        print(df.head().to_string())
