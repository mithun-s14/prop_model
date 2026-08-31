"""
Tests for backend/train_real_model.py: training NBAProjectionModel's
ensemble architecture on real historical outcomes instead of synthetic
formula-derived targets.
"""
import sys
import os
import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from train_real_model import build_training_dataset, train_and_evaluate, FEATURE_COLUMNS


DEFENSE_STATS = {
    'opp_pts_allowed': 24.0, 'opp_reb_allowed': 7.5, 'opp_ast_allowed': 5.5, 'opp_fd_allowed': 24.0,
}


def make_matchup_gamelog(home_team='CHO', away_team='HOU', n_games=8):
    """Two players who play each other every game; game_id encodes home team, mirroring
    Basketball-Reference IDs (e.g. '202602190CHO')."""
    dates = pd.date_range(start='2026-01-01', periods=n_games, freq='D')
    rows = []
    for i, date in enumerate(dates):
        game_id = f"{date.strftime('%Y%m%d')}0{home_team}"
        rows.append({
            'PLAYER_NAME': 'Home Player', 'team': home_team, 'game_id': game_id,
            'GAME_DATE': date.strftime('%Y-%m-%d'), 'PTS': 20 + i, 'REB': 5, 'AST': 5,
            'MIN': 30, 'FGA': 15, 'FG3A': 4, 'FTA': 5, 'STL': 1, 'BLK': 1, 'TOV': 2,
        })
        rows.append({
            'PLAYER_NAME': 'Away Player', 'team': away_team, 'game_id': game_id,
            'GAME_DATE': date.strftime('%Y-%m-%d'), 'PTS': 10 + i, 'REB': 4, 'AST': 3,
            'MIN': 28, 'FGA': 12, 'FG3A': 3, 'FTA': 4, 'STL': 1, 'BLK': 0, 'TOV': 2,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def mocked_lookups():
    with patch('train_real_model.calculate_usage_rate', return_value=22.0), \
         patch('train_real_model.get_player_position', return_value='SG'), \
         patch('train_real_model.get_opponent_defense_stats', return_value=DEFENSE_STATS):
        yield


# Happy path

class TestBuildTrainingDataset:
    def test_produces_one_row_per_game_after_min_prior(self, mocked_lookups):
        df = make_matchup_gamelog(n_games=8)
        dataset = build_training_dataset(df, min_prior_games=5)
        # Each player: games at index 5,6,7 qualify (3 rows) => 6 rows total
        assert len(dataset) == 6
        assert set(dataset['player']) == {'Home Player', 'Away Player'}

    def test_only_last_game_per_player_is_test(self, mocked_lookups):
        df = make_matchup_gamelog(n_games=8)
        dataset = build_training_dataset(df, min_prior_games=5)
        home_rows = dataset[dataset['player'] == 'Home Player'].sort_values('game_date')
        assert home_rows['is_test'].tolist() == [False, False, True]
        # Held-out game (index 7) has actual PTS = 27
        assert home_rows.iloc[-1]['actual_pts'] == 27

    def test_feature_columns_present(self, mocked_lookups):
        df = make_matchup_gamelog(n_games=8)
        dataset = build_training_dataset(df, min_prior_games=5)
        for col in FEATURE_COLUMNS:
            assert col in dataset.columns

    def test_baseline_pred_is_mean_of_strictly_prior_games(self, mocked_lookups):
        df = make_matchup_gamelog(n_games=8)
        dataset = build_training_dataset(df, min_prior_games=5)
        home_rows = dataset[dataset['player'] == 'Home Player'].sort_values('game_date').reset_index(drop=True)
        # First qualifying row predicts game index 5 (PTS=25) using games 0..4 (PTS 20..24) -> mean=22
        assert home_rows.iloc[0]['baseline_pred'] == pytest.approx(22.0)


class TestTrainAndEvaluate:
    def test_returns_summary_and_predictions(self, mocked_lookups):
        df = make_matchup_gamelog(n_games=10)
        summary, results_df = train_and_evaluate(df, min_prior_games=5)
        assert summary['n_test'] == 2  # one held-out game per player
        assert summary['n_train'] > 0
        assert summary['baseline_mae'] >= 0
        assert summary['model_mae'] >= 0
        assert set(results_df.columns) == {'player', 'actual', 'baseline_pred', 'model_pred'}
        assert len(results_df) == summary['n_test']


# Edge cases

class TestEdgeCases:
    def test_empty_gamelog_returns_zero_counts(self, mocked_lookups):
        df = pd.DataFrame(columns=['PLAYER_NAME', 'GAME_DATE', 'game_id', 'team', 'PTS'])
        summary, results_df = train_and_evaluate(df, min_prior_games=5)
        assert summary == {'n_train': 0, 'n_test': 0, 'baseline_mae': None, 'model_mae': None}
        assert results_df.empty

    def test_no_player_meets_min_prior_games(self, mocked_lookups):
        df = make_matchup_gamelog(n_games=3)  # below default min_prior_games=5
        dataset = build_training_dataset(df, min_prior_games=5)
        assert dataset.empty

    def test_all_rows_test_yields_empty_train_and_none_mae(self, mocked_lookups):
        """A player with exactly min_prior_games + 1 games contributes only
        a single row, which is always the held-out test row -> empty train set."""
        df = make_matchup_gamelog(n_games=6)  # index 5 is the only qualifying row, and it's last
        summary, results_df = train_and_evaluate(df, min_prior_games=5)
        assert summary['n_train'] == 0
        assert summary['n_test'] == 2
        assert summary['baseline_mae'] is None
        assert summary['model_mae'] is None


# Failure cases

class TestFailureCases:
    def test_missing_required_column_raises(self):
        df = pd.DataFrame({'PLAYER_NAME': ['A'], 'GAME_DATE': ['2026-01-01']})
        with pytest.raises(ValueError):
            build_training_dataset(df)
