"""
Unit tests for pure utility functions in model.py.
These have no external dependencies and are the easiest to test.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import (
    safe_float,
    get_default_defense_stats,
    get_team_full_name,
    calculate_rolling_features,
)


# safe_float
class TestSafeFloat:
    def test_int_input(self):
        assert safe_float(42) == 42.0

    def test_float_input(self):
        assert safe_float(3.14) == 3.14

    def test_string_number(self):
        assert safe_float("12.5") == 12.5

    def test_invalid_string_returns_zero(self):
        assert safe_float("not_a_number") == 0.0

    def test_none_returns_zero(self):
        assert safe_float(None) == 0.0

    def test_empty_string_returns_zero(self):
        assert safe_float("") == 0.0


# get_default_defense_stats
class TestGetDefaultDefenseStats:
    def test_returns_dict(self):
        result = get_default_defense_stats()
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = get_default_defense_stats()
        required = ['opp_pts_allowed', 'opp_reb_allowed', 'opp_ast_allowed', 'opp_fd_allowed']
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_values_are_positive_floats(self):
        for value in get_default_defense_stats().values():
            assert isinstance(value, float)
            assert value > 0


# get_team_full_name
class TestGetTeamFullName:
    @pytest.mark.parametrize("abbr,expected", [
        ('LAL', 'Los Angeles Lakers'),
        ('BOS', 'Boston Celtics'),
        ('GSW', 'Golden State Warriors'),
        ('BKN', 'Brooklyn Nets'),
        ('NYK', 'New York Knicks'),
    ])
    def test_known_teams(self, abbr, expected):
        assert get_team_full_name(abbr) == expected

    def test_unknown_abbreviation_returns_input(self):
        assert get_team_full_name('XYZ') == 'XYZ'


# calculate_rolling_features
class TestCalculateRollingFeatures:
    def test_empty_dataframe_returns_empty_dict(self):
        assert calculate_rolling_features(pd.DataFrame()) == {}

    def test_returns_rolling_averages(self, sample_gamelog_df):
        features = calculate_rolling_features(sample_gamelog_df, window=5)
        assert 'pts_roll_avg' in features
        assert 'reb_roll_avg' in features
        assert 'ast_roll_avg' in features

    def test_rolling_avg_values_are_reasonable(self, sample_gamelog_df):
        features = calculate_rolling_features(sample_gamelog_df, window=5)
        # Rolling avg of PTS should be between min and max of the data
        assert sample_gamelog_df['PTS'].min() <= features['pts_roll_avg'] <= sample_gamelog_df['PTS'].max()

    def test_window_of_one_equals_last_value(self):
        df = pd.DataFrame({
            'GAME_DATE': ['2025-01-01', '2025-01-02', '2025-01-03'],
            'PTS': [10, 20, 30],
        })
        features = calculate_rolling_features(df, window=1)
        assert features['pts_roll_avg'] == 30.0

    def test_missing_columns_handled(self):
        df = pd.DataFrame({'GAME_DATE': ['2025-01-01'], 'RANDOM_COL': [5]})
        result = calculate_rolling_features(df)
        assert result == {}