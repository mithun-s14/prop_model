"""
Shared pytest fixtures for NBA Projection Model tests.
Provides reusable mock data (DataFrames, dicts) that mirror production CSV/Excel schemas.
"""
import os
import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import patch
from datetime import datetime, timedelta


# Player info fixtures (mirrors cached_player_info.csv)

@pytest.fixture
def sample_player_info_df():
    """DataFrame matching cached_player_info.csv schema."""
    return pd.DataFrame({
        'player_id': ['jamesle01', 'curryst01', 'duranke01'],
        'player_name': ['LeBron James', 'Stephen Curry', 'Kevin Durant'],
        'team_id': [1610612747, 1610612744, 1610612745],
        'team_abbreviation': ['LAL', 'GSW', 'HOU'],
        'team_name': ['Los Angeles Lakers', 'Golden State Warriors', 'Houston Rockets'],
        'position': ['SF', 'PG', 'SF'],
    })


# Usage rate fixtures (mirrors nba_usage_rates_latest.csv)
@pytest.fixture
def sample_usage_df():
    """DataFrame matching nba_usage_rates_latest.csv schema."""
    return pd.DataFrame({
        'RANK': [1, 2, 3],
        'PLAYER': ['LEBRON JAMES', 'STEPHEN CURRY', 'KEVIN DURANT'],
        'TEAM': ['LAL', 'GSW', 'HOU'],
        'USGPCT': [27.1, 31.0, 26.1],
    })


# Game log fixtures (mirrors cached_player_gamelogs.csv)
@pytest.fixture
def sample_gamelog_df():
    """DataFrame matching cached_player_gamelogs.csv schema."""
    dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'Player_ID': ['jamesle01'] * 10,
        'PLAYER_NAME': ['LeBron James'] * 10,
        'GAME_DATE': [d.strftime('%b %d, %Y') for d in dates],
        'PTS': rng.integers(18, 35, size=10).tolist(),
        'REB': rng.integers(5, 12, size=10).tolist(),
        'AST': rng.integers(4, 12, size=10).tolist(),
        'MIN': rng.integers(28, 38, size=10).tolist(),
        'FGA': rng.integers(12, 22, size=10).tolist(),
        'FG3A': rng.integers(2, 8, size=10).tolist(),
        'FTA': rng.integers(3, 10, size=10).tolist(),
        'STL': rng.integers(0, 3, size=10).tolist(),
        'BLK': rng.integers(0, 3, size=10).tolist(),
        'TOV': rng.integers(1, 5, size=10).tolist(),
    })


# ---------------------------------------------------------------------------
# Schedule / teams fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_todays_games_df():
    """DataFrame matching cached_todays_games.csv schema."""
    return pd.DataFrame({
        'HOME_TEAM_ID': [1610612747],
        'VISITOR_TEAM_ID': [1610612738],
        'GAME_ID': ['0022500999'],
        'home_team': ['Los Angeles Lakers'],
        'visitor_team': ['Boston Celtics'],
    })


@pytest.fixture
def sample_teams_df():
    """DataFrame matching cached_all_teams.csv schema."""
    return pd.DataFrame({
        'id': [1610612747, 1610612738, 1610612744, 1610612756],
        'full_name': ['Los Angeles Lakers', 'Boston Celtics', 'Golden State Warriors', 'Phoenix Suns'],
        'abbreviation': ['LAL', 'BOS', 'GSW', 'PHX'],
    })


# Game context & defense stats fixtures
@pytest.fixture
def sample_game_context():
    """Dict matching the shape returned by get_tonights_game_context."""
    return {
        'opponent': 'BOS',
        'is_home': 1,
        'spread': -3.5,
        'total': 225.5,
    }


@pytest.fixture
def sample_defense_stats():
    """Dict matching the shape returned by get_opponent_defense_stats."""
    return {
        'opp_pts_allowed': 24.5,
        'opp_reb_allowed': 7.8,
        'opp_ast_allowed': 5.9,
        'opp_fd_allowed': 42.0,
    }


# FantasyPros last-15 averages fixture — mirrors actual CSV column names
@pytest.fixture
def sample_fantasypros_df():
    """DataFrame matching the real schema from FantasyPros avg-overall (last 15 days)."""
    return pd.DataFrame({
        'Player': ['LeBron James', 'Stephen Curry', 'Kevin Durant', 'Jayson Tatum'],
        'PTS':    [27.3, 29.8, 26.1, 28.4],
        'REB':    [ 8.1,  4.5,  7.0,  8.1],
        'AST':    [ 7.4,  6.3,  3.8,  4.2],
        'BLK':    [ 0.9,  0.2,  1.2,  0.8],
        'STL':    [ 1.3,  1.5,  0.9,  1.1],
        'FG%':    [0.529, 0.452, 0.544, 0.452],
        'FT%':    [0.756, 0.921, 0.882, 0.845],
        '3PM':    [ 1.8,  5.2,  2.1,  3.4],
        'TO':     [ 3.1,  2.8,  2.5,  2.3],
        'GP':     [10, 10, 10, 10],
        'MIN':    [34.2, 32.1, 35.0, 36.5],
        'FTM':    [ 5.1,  3.8,  6.2,  4.9],
        '2PM':    [ 8.4,  5.3,  7.7,  6.7],
        'A/TO':   [ 2.4,  2.3,  1.5,  1.8],
        'PF':     [ 1.8,  2.1,  2.4,  2.0],
    })


# RealGM last-10 averages fixture (mirrors realgm_last10_averages.csv)
@pytest.fixture
def sample_realgm_df():
    """DataFrame matching the schema returned by scrape_realgm_last10."""
    return pd.DataFrame({
        'Player': ['LeBron James', 'Stephen Curry', 'Kevin Durant', 'Jayson Tatum'],
        'Team':   ['LAL', 'GSW', 'HOU', 'BOS'],
        'GP':     [10, 10, 10, 10],
        'MPG':    [34.2, 32.1, 35.0, 36.5],
        'PPG':    [27.3, 29.8, 26.1, 28.4],
        'FGM':    [10.2, 10.5,  9.8, 10.1],
        'FGA':    [18.5, 20.2, 18.0, 19.3],
        'FG_PCT': [55.1, 52.0, 54.4, 52.3],
        '3PM':    [ 1.8,  5.2,  2.1,  3.4],
        '3PA':    [ 4.8, 12.3,  5.0,  8.9],
        '3P_PCT': [37.5, 42.3, 42.0, 38.2],
        'FTM':    [ 5.1,  3.8,  6.2,  4.9],
        'FTA':    [ 6.2,  4.4,  7.5,  5.8],
        'FT_PCT': [82.3, 86.4, 82.7, 84.5],
        'ORB':    [ 1.2,  0.4,  0.8,  1.0],
        'DRB':    [ 6.9,  4.1,  6.2,  7.1],
        'RPG':    [ 8.1,  4.5,  7.0,  8.1],
        'APG':    [ 7.4,  6.3,  3.8,  4.2],
        'SPG':    [ 1.3,  1.5,  0.9,  1.1],
        'BPG':    [ 0.9,  0.2,  1.2,  0.8],
        'TOV':    [ 3.1,  2.8,  2.5,  2.3],
        'PF':     [ 1.8,  2.1,  2.4,  2.0],
    })


# Player features fixture
@pytest.fixture
def sample_player_features():
    """Dict matching the shape returned by calculate_rolling_features."""
    return {
        'pts_roll_avg': 27.3,
        'reb_roll_avg': 8.1,
        'ast_roll_avg': 7.4,
        'min_roll_avg': 34.2,
        'fga_roll_avg': 18.5,
        'fg3a_roll_avg': 4.8,
        'fta_roll_avg': 6.2,
        'stl_roll_avg': 1.3,
        'blk_roll_avg': 0.9,
        'tov_roll_avg': 3.1,
    }


# ---------------------------------------------------------------------------
# Season-projection fixtures (season_features.py / train_season_model.py)
# ---------------------------------------------------------------------------

def make_player_season(name, season, n_games, pts=20, reb=5, ast=4, stl=1, blk=1,
                       fg3m=2, tov=2, minutes=30, fgm=8, fga=17, ftm=4, fta=5,
                       fg3a=6, team='LAL', player_id=None, start_month=11):
    """
    One player's game log for one season, with constant per-game values.

    Constant values make hand-computed expectations trivial: the season average
    of every stat is just the value passed in. `season` is the NBA end-year, so
    season=2025 means the 2024-25 season and games are dated in Nov 2024.
    """
    year = season - 1 if start_month >= 10 else season
    dates = pd.date_range(f'{year}-{start_month:02d}-01', periods=n_games, freq='2D')
    return pd.DataFrame({
        'PLAYER_NAME': [name] * n_games,
        'Player_ID': [player_id or name.lower().replace(' ', '')[:8]] * n_games,
        'GAME_DATE': [d.strftime('%a, %b %d, %Y') for d in dates],
        'team': [team] * n_games,
        'game_id': [f'{d:%Y%m%d}0{team}' for d in dates],
        'MIN': [minutes] * n_games,
        'FGM': [fgm] * n_games, 'FGA': [fga] * n_games,
        'FG3M': [fg3m] * n_games, 'FG3A': [fg3a] * n_games,
        'FTM': [ftm] * n_games, 'FTA': [fta] * n_games,
        'REB': [reb] * n_games, 'AST': [ast] * n_games,
        'STL': [stl] * n_games, 'BLK': [blk] * n_games,
        'TOV': [tov] * n_games, 'PTS': [pts] * n_games,
    })


@pytest.fixture
def multi_season_gamelog():
    """
    Three seasons (2024, 2025, 2026) for four players with distinct profiles:
      Steady Sam    -- same team, same production all three seasons
      Rising Rick   -- improves each season
      Moving Mike   -- changes teams between 2025 and 2026
      Short Steve   -- only 18 games in 2026 (below the feature threshold)
    """
    frames = [
        make_player_season('Steady Sam', 2024, 70, pts=20, team='LAL'),
        make_player_season('Steady Sam', 2025, 70, pts=20, team='LAL'),
        make_player_season('Steady Sam', 2026, 70, pts=20, team='LAL'),
        make_player_season('Rising Rick', 2024, 60, pts=10, minutes=20, team='BOS'),
        make_player_season('Rising Rick', 2025, 65, pts=15, minutes=28, team='BOS'),
        make_player_season('Rising Rick', 2026, 68, pts=22, minutes=34, team='BOS'),
        make_player_season('Moving Mike', 2025, 55, pts=18, team='MIA'),
        make_player_season('Moving Mike', 2026, 58, pts=16, team='DEN'),
        make_player_season('Short Steve', 2025, 40, pts=12, team='NYK'),
        make_player_season('Short Steve', 2026, 18, pts=9, team='NYK'),
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def season_player_info():
    """Player info for the multi_season_gamelog players, one deliberately absent."""
    return pd.DataFrame({
        'player_id': ['steadysa', 'risingri', 'movingmi'],   # Short Steve missing on purpose
        'player_name': ['Steady Sam', 'Rising Rick', 'Moving Mike'],
        'birth_date': ['June 1, 1995', 'March 15, 2000', 'January 20, 1992'],
        'experience': ['8', 'R', '12'],
    })


@pytest.fixture
def season_positions():
    """Position map including whitespace-corrupted values, as the real CSV has."""
    return {'Steady Sam': 'PG', 'Rising Rick': ' SG', 'Moving Mike': 'C'}
