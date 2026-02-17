"""
Unit tests for data loading / lookup functions in model.py.
Uses unittest.mock to patch file I/O so tests run without real CSV files.
"""
import sys
import os
import pytest
import pandas as pd
import json
from unittest.mock import patch, mock_open
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import (
    get_player_info,
    get_all_players_cached,
    get_player_position,
    calculate_usage_rate,
    load_latest_usage_data,
    fetch_recent_gamelog,
    get_player_team_id,
    get_team_schedule,
    get_tonights_game_context,
    get_opponent_defense_stats,
)


# get_player_info tests various name matching scenarios and error handling when loading player info from CSV.
class TestGetPlayerInfo:
    @patch('model.pd.read_csv')
    def test_exact_match(self, mock_read_csv, sample_player_info_df):
        mock_read_csv.return_value = sample_player_info_df
        result = get_player_info('LeBron James')

        assert result is not None
        assert result['id'] == 'jamesle01'
        assert result['team'] == 'LAL'
        assert result['position'] == 'SF'

    @patch('model.pd.read_csv')
    def test_case_insensitive_match(self, mock_read_csv, sample_player_info_df):
        mock_read_csv.return_value = sample_player_info_df
        result = get_player_info('lebron james')

        assert result is not None
        assert result['id'] == 'jamesle01'

    @patch('model.pd.read_csv')
    def test_partial_match(self, mock_read_csv, sample_player_info_df):
        mock_read_csv.return_value = sample_player_info_df
        result = get_player_info('LeBron')

        assert result is not None
        assert result['team'] == 'LAL'

    @patch('model.pd.read_csv')
    def test_last_name_match(self, mock_read_csv, sample_player_info_df):
        mock_read_csv.return_value = sample_player_info_df
        result = get_player_info('Curry')

        assert result is not None
        assert result['team'] == 'GSW'

    @patch('model.pd.read_csv')
    def test_player_not_found(self, mock_read_csv, sample_player_info_df):
        mock_read_csv.return_value = sample_player_info_df
        result = get_player_info('Fake Player')

        assert result is None

    @patch('model.pd.read_csv', side_effect=Exception("file not found"))
    def test_handles_file_error(self, mock_read_csv):
        result = get_player_info('LeBron James')
        assert result is None


# get_all_players_cached

class TestGetAllPlayersCached:
    @patch('builtins.open', mock_open(read_data='[{"id": "jamesle01", "full_name": "LeBron James"}]'))
    def test_loads_from_json(self):
        result = get_all_players_cached()
        assert len(result) == 1
        assert result[0]['full_name'] == 'LeBron James'


# get_player_position

class TestGetPlayerPosition:
    @patch('builtins.open', mock_open(read_data='Player,Position\nLeBron James,SF\nStephen Curry,PG\n'))
    def test_exact_match_from_csv(self):
        result = get_player_position('LeBron James')
        assert result == 'SF'

    @patch('builtins.open', mock_open(read_data='Player,Position\nStephen Curry,PG\n'))
    @patch('model.pd.read_csv')
    def test_fallback_to_cached_player_info(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            'player_name': ['LeBron James'],
            'position': ['SF'],
        })
        result = get_player_position('LeBron James')
        assert result == 'SF'

    @patch('builtins.open', side_effect=FileNotFoundError)
    @patch('model.pd.read_csv', side_effect=Exception("no file"))
    def test_defaults_to_sg_on_failure(self, mock_csv, mock_file):
        result = get_player_position('Unknown Player')
        assert result == 'SG'


# calculate_usage_rate
class TestCalculateUsageRate:
    @patch('model.load_latest_usage_data')
    def test_exact_match(self, mock_load, sample_usage_df):
        mock_load.return_value = sample_usage_df
        result = calculate_usage_rate('LeBron James')
        assert result == 27.1

    @patch('model.load_latest_usage_data')
    def test_partial_match(self, mock_load, sample_usage_df):
        mock_load.return_value = sample_usage_df
        result = calculate_usage_rate('LeBron')
        assert result == 27.1

    @patch('model.load_latest_usage_data')
    def test_last_name_match(self, mock_load, sample_usage_df):
        mock_load.return_value = sample_usage_df
        result = calculate_usage_rate('Curry')
        assert result == 31.0

    @patch('model.load_latest_usage_data')
    def test_not_found_returns_default(self, mock_load, sample_usage_df):
        mock_load.return_value = sample_usage_df
        result = calculate_usage_rate('Fake Player')
        assert result == 20.0

    @patch('model.load_latest_usage_data')
    def test_no_data_returns_default(self, mock_load):
        mock_load.return_value = None
        result = calculate_usage_rate('LeBron James')
        assert result == 25.0


# load_latest_usage_data
class TestLoadLatestUsageData:
    @patch('model.os.path.exists', return_value=False)
    def test_missing_file_returns_none(self, mock_exists):
        result = load_latest_usage_data()
        assert result is None

    @patch('model.pd.read_csv')
    @patch('model.os.path.getmtime')
    @patch('model.os.path.exists', return_value=True)
    def test_fresh_file_loads(self, mock_exists, mock_mtime, mock_csv, sample_usage_df):
        mock_mtime.return_value = datetime.now().timestamp()
        mock_csv.return_value = sample_usage_df
        result = load_latest_usage_data()
        assert len(result) == 3

    @patch('model.pd.read_csv')
    @patch('model.os.path.getmtime')
    @patch('model.os.path.exists', return_value=True)
    def test_stale_file_still_loads_with_warning(self, mock_exists, mock_mtime, mock_csv, sample_usage_df):
        mock_mtime.return_value = (datetime.now() - timedelta(hours=48)).timestamp()
        mock_csv.return_value = sample_usage_df
        result = load_latest_usage_data()
        assert result is not None


# fetch_recent_gamelog
class TestFetchRecentGamelog:
    @patch('model.pd.read_csv')
    def test_loads_by_player_id(self, mock_read_csv, sample_gamelog_df):
        mock_read_csv.return_value = sample_gamelog_df
        result = fetch_recent_gamelog('jamesle01', 'LAL')
        assert not result.empty
        assert len(result) == 10

    @patch('model.get_player_season_averages')
    @patch('model.pd.read_csv')
    def test_no_matching_id_falls_back_to_name(self, mock_read_csv, mock_averages, sample_gamelog_df, sample_player_info_df):
        # First call: gamelogs, second call: player_info
        mock_read_csv.side_effect = [sample_gamelog_df, sample_player_info_df]
        mock_averages.return_value = pd.DataFrame()

        result = fetch_recent_gamelog('jamesle01', 'LAL')
        assert not result.empty


# get_player_team_id
class TestGetPlayerTeamId:
    @patch('model.pd.read_csv')
    def test_returns_team_id(self, mock_csv, sample_player_info_df):
        mock_csv.return_value = sample_player_info_df
        result = get_player_team_id('jamesle01')
        assert result == 1610612747

    @patch('model.pd.read_csv')
    def test_player_not_found(self, mock_csv, sample_player_info_df):
        mock_csv.return_value = sample_player_info_df
        result = get_player_team_id('fakeplayer99')
        assert result is None


# get_team_schedule
class TestGetTeamSchedule:
    @patch('model.pd.read_csv')
    def test_finds_home_game(self, mock_csv, sample_todays_games_df):
        mock_csv.return_value = sample_todays_games_df
        result = get_team_schedule(1610612747)  # Lakers
        assert result is not None
        assert result['HOME_TEAM_ID'] == 1610612747

    @patch('model.pd.read_csv')
    def test_finds_away_game(self, mock_csv, sample_todays_games_df):
        mock_csv.return_value = sample_todays_games_df
        result = get_team_schedule(1610612738)  # Celtics (visitor)
        assert result is not None
        assert result['VISITOR_TEAM_ID'] == 1610612738

    @patch('model.pd.read_csv')
    def test_no_game_returns_none(self, mock_csv, sample_todays_games_df):
        mock_csv.return_value = sample_todays_games_df
        result = get_team_schedule(9999999)  # Fake team
        assert result is None

    @patch('model.pd.read_csv', side_effect=FileNotFoundError)
    def test_missing_file_returns_none(self, mock_csv):
        result = get_team_schedule(1610612747)
        assert result is None
