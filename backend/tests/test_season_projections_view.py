"""
Tests for backend/season_projections_view.py: loading, filtering, and
display formatting of the 2026-27 season projections shown in the
Gradio "2026-27 Season Projections" tab.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from season_projections_view import (  # noqa: E402
    ALL_POSITIONS,
    ALL_TEAMS,
    DISPLAY_COLUMNS,
    EXCLUDED_CSV,
    PROJECTIONS_CSV,
    SORT_OPTIONS,
    filter_projections,
    get_position_choices,
    get_team_choices,
    load_excluded_players,
    load_season_projections,
    projections_summary,
)

DISPLAY_LABELS = [label for _, label, _ in DISPLAY_COLUMNS]


@pytest.fixture
def sample_projections_df():
    """Small DataFrame mirroring season_projections_2027.csv's schema."""
    return pd.DataFrame({
        'player': ['Nikola Jokic', 'Victor Wembanyama', 'Stephen Curry', 'Jalen Suggs'],
        'player_id': ['jokicni01', 'wembavi01', 'curryst01', 'suggsja01'],
        'team_prev': ['DEN', 'SAS', 'GSW', 'ORL'],
        'position': ['C', 'C', 'PG', 'PG'],
        'age': [31.95071, 23.07734, 38.4, 25.2],
        'gp_prev': [65.0, 64.0, 70.0, 35.0],
        'pred_gp': [64.52115, 62.33609, 66.1, 55.4],
        'pred_min': [34.32674, 29.79427, 30.5, 28.1],
        'pred_pts': [26.77827, 25.00792, 22.4, 14.9],
        'pred_reb': [12.40441, 11.46666, 4.3, 3.6],
        'pred_ast': [10.15612, 3.20892, 6.1, 4.2],
        'pred_stl': [1.39085, 1.04491, 1.0, 1.4],
        'pred_blk': [0.80592, 2.85075, 0.3, 0.5],
        'pred_fg3m': [1.63991, 1.86072, 4.6, 2.1],
        'pred_tov': [3.68200, 2.49322, 2.7, 2.0],
        'pred_fg_pct': [0.5648305, 0.5102128, 0.451234, 0.431111],
        'pred_ft_pct': [0.8288042, 0.8243338, 0.919876, 0.782345],
        'value_total': [12.55735, 11.97506, 8.4, 2.1],
    })


@pytest.fixture
def sample_excluded_df():
    return pd.DataFrame({
        'player': ['Russell Westbrook'],
        'player_id': ['westbru01'],
        'reason': ['RETIRED'],
    })


# Loading

class TestLoading:
    def test_real_projections_csv_exists_and_loads(self):
        assert os.path.exists(PROJECTIONS_CSV)
        df = load_season_projections()
        assert not df.empty
        for src, _, _ in DISPLAY_COLUMNS:
            assert src in df.columns

    def test_real_excluded_csv_loads(self):
        assert os.path.exists(EXCLUDED_CSV)
        df = load_excluded_players()
        assert list(df.columns) == ['player', 'player_id', 'reason']

    def test_missing_projections_file_returns_empty_frame(self, tmp_path):
        df = load_season_projections(str(tmp_path / 'nope.csv'))
        assert df.empty

    def test_missing_excluded_file_returns_empty_frame(self, tmp_path):
        df = load_excluded_players(str(tmp_path / 'nope.csv'))
        assert df.empty
        assert list(df.columns) == ['player', 'player_id', 'reason']

    def test_loads_from_explicit_path(self, tmp_path, sample_projections_df):
        path = tmp_path / 'proj.csv'
        sample_projections_df.to_csv(path, index=False)
        df = load_season_projections(str(path))
        assert len(df) == 4


# Data validation on the real projections file

class TestRealDataValidation:
    def test_no_duplicate_player_ids(self):
        df = load_season_projections()
        assert df['player_id'].duplicated().sum() == 0

    def test_projected_stats_are_non_negative(self):
        df = load_season_projections()
        for col in ['pred_pts', 'pred_reb', 'pred_ast', 'pred_stl',
                    'pred_blk', 'pred_fg3m', 'pred_tov', 'pred_min', 'pred_gp']:
            assert (df[col].dropna() >= 0).all(), f"{col} has negative values"

    def test_percentages_are_fractions(self):
        df = load_season_projections()
        for col in ['pred_fg_pct', 'pred_ft_pct']:
            vals = df[col].dropna()
            assert vals.between(0, 1).all(), f"{col} outside 0-1"

    def test_projection_intervals_bracket_the_point_estimate(self):
        df = load_season_projections()
        for stat in ['pts', 'reb', 'ast', 'min', 'gp']:
            lo, mid, hi = f'pred_{stat}_low', f'pred_{stat}', f'pred_{stat}_high'
            sub = df[[lo, mid, hi]].dropna()
            assert (sub[lo] <= sub[mid]).all(), f"{lo} above point estimate"
            assert (sub[mid] <= sub[hi]).all(), f"{hi} below point estimate"

    def test_ages_are_plausible(self):
        df = load_season_projections()
        ages = df['age'].dropna()
        assert ages.between(17, 50).all()

    def test_minutes_and_games_within_season_bounds(self):
        df = load_season_projections()
        assert (df['pred_min'].dropna() <= 48).all()
        assert (df['pred_gp'].dropna() <= 82).all()


# Dropdown choices

class TestChoices:
    def test_team_choices_sorted_with_sentinel_first(self, sample_projections_df):
        choices = get_team_choices(sample_projections_df)
        assert choices[0] == ALL_TEAMS
        assert choices[1:] == ['DEN', 'GSW', 'ORL', 'SAS']

    def test_position_choices_sorted_with_sentinel_first(self, sample_projections_df):
        choices = get_position_choices(sample_projections_df)
        assert choices == [ALL_POSITIONS, 'C', 'PG']

    def test_choices_on_empty_frame(self):
        assert get_team_choices(pd.DataFrame()) == [ALL_TEAMS]
        assert get_position_choices(pd.DataFrame()) == [ALL_POSITIONS]

    def test_choices_skip_missing_values(self, sample_projections_df):
        sample_projections_df.loc[0, 'team_prev'] = None
        choices = get_team_choices(sample_projections_df)
        assert 'DEN' not in choices
        assert choices[0] == ALL_TEAMS


# Filtering / sorting / display shaping

class TestFilterProjections:
    def test_default_returns_display_columns(self, sample_projections_df):
        out = filter_projections(sample_projections_df)
        assert list(out.columns) == DISPLAY_LABELS
        assert len(out) == 4

    def test_default_sorts_by_value_descending(self, sample_projections_df):
        out = filter_projections(sample_projections_df)
        assert out['Player'].tolist()[0] == 'Nikola Jokic'
        assert out['Value'].is_monotonic_decreasing

    def test_sort_by_points(self, sample_projections_df):
        out = filter_projections(sample_projections_df, sort_by='Points')
        assert out['PTS'].is_monotonic_decreasing

    def test_sort_by_player_is_ascending(self, sample_projections_df):
        out = filter_projections(sample_projections_df, sort_by='Player (A-Z)')
        assert out['Player'].tolist() == sorted(out['Player'].tolist())

    def test_unknown_sort_key_falls_back_to_value(self, sample_projections_df):
        out = filter_projections(sample_projections_df, sort_by='Not A Column')
        assert out['Value'].is_monotonic_decreasing

    def test_every_sort_option_is_usable(self, sample_projections_df):
        for label in SORT_OPTIONS:
            out = filter_projections(sample_projections_df, sort_by=label)
            assert len(out) == 4

    def test_search_is_case_insensitive_substring(self, sample_projections_df):
        out = filter_projections(sample_projections_df, search='jOkI')
        assert out['Player'].tolist() == ['Nikola Jokic']

    def test_search_with_no_match_returns_empty_with_headers(self, sample_projections_df):
        out = filter_projections(sample_projections_df, search='zzzz')
        assert len(out) == 0
        assert list(out.columns) == DISPLAY_LABELS

    def test_team_filter(self, sample_projections_df):
        out = filter_projections(sample_projections_df, team='GSW')
        assert out['Player'].tolist() == ['Stephen Curry']

    def test_position_filter(self, sample_projections_df):
        out = filter_projections(sample_projections_df, position='PG')
        assert set(out['Pos']) == {'PG'}
        assert len(out) == 2

    def test_filters_combine(self, sample_projections_df):
        out = filter_projections(
            sample_projections_df, search='a', team='SAS', position='C'
        )
        assert out['Player'].tolist() == ['Victor Wembanyama']

    def test_limit_truncates(self, sample_projections_df):
        out = filter_projections(sample_projections_df, limit=2)
        assert len(out) == 2

    def test_limit_none_or_zero_returns_all(self, sample_projections_df):
        assert len(filter_projections(sample_projections_df, limit=None)) == 4
        assert len(filter_projections(sample_projections_df, limit=0)) == 4

    def test_values_are_rounded_for_display(self, sample_projections_df):
        out = filter_projections(sample_projections_df, search='Jokic')
        row = out.iloc[0]
        assert row['PTS'] == 26.8
        assert row['Age'] == 32.0
        assert row['FG%'] == 0.565
        assert row['Value'] == 12.56

    def test_index_is_reset(self, sample_projections_df):
        out = filter_projections(sample_projections_df, sort_by='Points')
        assert out.index.tolist() == list(range(len(out)))

    def test_empty_or_none_input(self):
        for df in (pd.DataFrame(), None):
            out = filter_projections(df)
            assert out.empty
            assert list(out.columns) == DISPLAY_LABELS

    def test_missing_source_column_is_filled_with_nulls(self, sample_projections_df):
        out = filter_projections(sample_projections_df.drop(columns=['pred_blk']))
        assert 'BLK' in out.columns
        assert out['BLK'].isna().all()

    def test_real_data_filters_end_to_end(self):
        df = load_season_projections()
        out = filter_projections(df, sort_by='Points', limit=25)
        assert len(out) == 25
        assert list(out.columns) == DISPLAY_LABELS
        assert out['PTS'].is_monotonic_decreasing


# Summary line

class TestSummary:
    def test_summary_counts(self, sample_projections_df, sample_excluded_df):
        text = projections_summary(sample_projections_df, sample_excluded_df)
        assert '4 players' in text
        assert '4 teams' in text
        assert '1 excluded' in text

    def test_summary_without_excluded_frame(self, sample_projections_df):
        assert '0 excluded' in projections_summary(sample_projections_df)

    def test_summary_on_empty_frame(self):
        assert projections_summary(pd.DataFrame()) == 'No projections available.'
