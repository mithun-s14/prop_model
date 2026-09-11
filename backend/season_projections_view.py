"""
Loading and filtering helpers for the 2026-2027 season projections table.

Kept free of any Gradio imports so the data layer can be tested on its own;
app.py wraps these in the "Season Projections" tab.
"""
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PROJECTIONS_CSV = os.path.join(DATA_DIR, 'season_projections_2027.csv')
EXCLUDED_CSV = os.path.join(DATA_DIR, 'season_projections_2027_excluded.csv')

# (source column, display column, decimal places)
DISPLAY_COLUMNS = [
    ('player', 'Player', None),
    ('team_prev', 'Team', None),
    ('position', 'Pos', None),
    ('age', 'Age', 1),
    ('pred_gp', 'GP', 1),
    ('pred_min', 'MIN', 1),
    ('pred_pts', 'PTS', 1),
    ('pred_reb', 'REB', 1),
    ('pred_ast', 'AST', 1),
    ('pred_stl', 'STL', 1),
    ('pred_blk', 'BLK', 1),
    ('pred_fg3m', '3PM', 1),
    ('pred_tov', 'TOV', 1),
    ('pred_fg_pct', 'FG%', 3),
    ('pred_ft_pct', 'FT%', 3),
    ('value_total', 'Value', 2),
]

# Display name -> source column for the sort dropdown.
SORT_OPTIONS = {
    'Value': 'value_total',
    'Points': 'pred_pts',
    'Rebounds': 'pred_reb',
    'Assists': 'pred_ast',
    'Steals': 'pred_stl',
    'Blocks': 'pred_blk',
    '3PM': 'pred_fg3m',
    'Minutes': 'pred_min',
    'Games Played': 'pred_gp',
    'Age': 'age',
    'Player (A-Z)': 'player',
}
ALL_TEAMS = 'All Teams'
ALL_POSITIONS = 'All Positions'


def load_season_projections(path=PROJECTIONS_CSV):
    """Load the raw projections CSV. Returns an empty DataFrame if missing."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=[src for src, _, _ in DISPLAY_COLUMNS])
    return pd.read_csv(path)


def load_excluded_players(path=EXCLUDED_CSV):
    """Load the list of players excluded from the projections (retired, etc.)."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=['player', 'player_id', 'reason'])
    return pd.read_csv(path)


def get_team_choices(df):
    """Sorted team dropdown choices, with the 'all' sentinel first."""
    if df.empty or 'team_prev' not in df.columns:
        return [ALL_TEAMS]
    teams = sorted(df['team_prev'].dropna().astype(str).unique().tolist())
    return [ALL_TEAMS] + teams


def get_position_choices(df):
    """Sorted position dropdown choices, with the 'all' sentinel first."""
    if df.empty or 'position' not in df.columns:
        return [ALL_POSITIONS]
    positions = sorted(df['position'].dropna().astype(str).unique().tolist())
    return [ALL_POSITIONS] + positions


def filter_projections(df, search='', team=ALL_TEAMS, position=ALL_POSITIONS,
                       sort_by='Value', limit=100):
    """
    Filter/sort the projections and return a display-ready DataFrame with
    friendly column names and rounded values.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[label for _, label, _ in DISPLAY_COLUMNS])

    out = df.copy()

    if search and str(search).strip():
        q = str(search).strip().lower()
        out = out[out['player'].astype(str).str.lower().str.contains(q, na=False)]

    if team and team != ALL_TEAMS:
        out = out[out['team_prev'].astype(str) == str(team)]

    if position and position != ALL_POSITIONS:
        out = out[out['position'].astype(str) == str(position)]

    sort_col = SORT_OPTIONS.get(sort_by, 'value_total')
    if sort_col in out.columns:
        ascending = sort_col == 'player'
        out = out.sort_values(sort_col, ascending=ascending, na_position='last')

    if limit is not None and int(limit) > 0:
        out = out.head(int(limit))

    display = pd.DataFrame()
    for src, label, decimals in DISPLAY_COLUMNS:
        if src not in out.columns:
            display[label] = pd.Series([None] * len(out), index=out.index)
            continue
        col = out[src]
        if decimals is not None:
            col = pd.to_numeric(col, errors='coerce').round(decimals)
        display[label] = col

    return display.reset_index(drop=True)


def projections_summary(df, excluded_df=None):
    """One-line summary of the projection set, used above the table."""
    if df is None or df.empty:
        return "No projections available."
    n_players = len(df)
    n_teams = df['team_prev'].nunique() if 'team_prev' in df.columns else 0
    n_excluded = 0 if excluded_df is None else len(excluded_df)
    return (f"{n_players} players projected across {n_teams} teams "
            f"({n_excluded} excluded).")
