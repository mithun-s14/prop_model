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


# FantasyPros last-15 averages fixture
@pytest.fixture
def sample_fantasypros_df():
    """DataFrame matching the schema returned by scrape_fantasypros_last15."""
    return pd.DataFrame({
        'Player': ['LeBron James', 'Stephen Curry', 'Kevin Durant', 'Jayson Tatum'],
        'Team':   ['LAL', 'GSW', 'HOU', 'BOS'],
        'POS':    ['SF', 'PG', 'SF', 'SF'],
        'GP':     [10, 10, 10, 10],
        'MPG':    [34.2, 32.1, 35.0, 36.5],
        'PPG':    [27.3, 29.8, 26.1, 28.4],
        'FGA':    [18.5, 20.2, 18.0, 19.3],
        '3PA':    [ 4.8, 12.3,  5.0,  8.9],
        'FTA':    [ 6.2,  4.4,  7.5,  5.8],
        'RPG':    [ 8.1,  4.5,  7.0,  8.1],
        'APG':    [ 7.4,  6.3,  3.8,  4.2],
        'SPG':    [ 1.3,  1.5,  0.9,  1.1],
        'BPG':    [ 0.9,  0.2,  1.2,  0.8],
        'TOV':    [ 3.1,  2.8,  2.5,  2.3],
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
