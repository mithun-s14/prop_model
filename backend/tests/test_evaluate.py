"""
Tests for backend/evaluate.py: walk-forward points evaluation against a
naive season-average baseline.
"""
import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluate import compute_predictions, evaluate_points_mae


def make_gamelog(player_points, player_name='LeBron James', start_date='2026-01-01'):
    """Build a minimal gamelog DataFrame for one player with given PTS sequence, in order."""
    dates = pd.date_range(start=start_date, periods=len(player_points), freq='D')
    return pd.DataFrame({
        'PLAYER_NAME': [player_name] * len(player_points),
        'GAME_DATE': dates.strftime('%Y-%m-%d'),
        'PTS': player_points,
    })


# Happy path

class TestComputePredictions:
    def test_skips_first_game_per_player(self):
        df = make_gamelog([10, 20, 30])
        result = compute_predictions(df)
        # First game has no prior history, so only 2 predictions expected
        assert len(result) == 2

    def test_baseline_is_running_average(self):
        df = make_gamelog([10, 20, 30])
        result = compute_predictions(df).sort_values('actual').reset_index(drop=True)
        # Predicting game 2 (actual=20): baseline = avg([10]) = 10
        row_for_20 = result[result['actual'] == 20].iloc[0]
        assert row_for_20['baseline_pred'] == 10
        # Predicting game 3 (actual=30): baseline = avg([10, 20]) = 15
        row_for_30 = result[result['actual'] == 30].iloc[0]
        assert row_for_30['baseline_pred'] == 15

    def test_model_pred_uses_rolling_window(self):
        df = make_gamelog([10, 20, 30, 40, 50, 60, 70])
        result = compute_predictions(df, rolling_window=3)
        # Predicting the last game (actual=70): model_pred = avg(last 3 prior) = avg([30,40,50,60][-3:])
        last_row = result[result['actual'] == 70].iloc[0]
        assert last_row['model_pred'] == pytest.approx((40 + 50 + 60) / 3)

    def test_multiple_players_handled_independently(self):
        df1 = make_gamelog([10, 20], player_name='Player A')
        df2 = make_gamelog([5, 15], player_name='Player B')
        combined = pd.concat([df1, df2], ignore_index=True)
        result = compute_predictions(combined)
        assert set(result['player']) == {'Player A', 'Player B'}
        assert len(result) == 2  # one prediction per player

    def test_predictions_ordered_chronologically(self):
        # Provide rows out of order; function should sort by GAME_DATE internally
        df = pd.DataFrame({
            'PLAYER_NAME': ['LeBron James'] * 3,
            'GAME_DATE': ['2026-01-03', '2026-01-01', '2026-01-02'],
            'PTS': [30, 10, 20],
        })
        result = compute_predictions(df).sort_values('actual').reset_index(drop=True)
        row_for_20 = result[result['actual'] == 20].iloc[0]
        assert row_for_20['baseline_pred'] == 10


class TestEvaluatePointsMae:
    def test_returns_mae_for_baseline_and_model(self):
        df = make_gamelog([10, 20, 30, 40, 50])
        results = evaluate_points_mae(df, rolling_window=5)
        assert results['n_predictions'] == 4
        assert results['baseline_mae'] >= 0
        assert results['model_mae'] >= 0

    def test_perfect_constant_scoring_has_zero_mae(self):
        df = make_gamelog([20, 20, 20, 20])
        results = evaluate_points_mae(df)
        assert results['baseline_mae'] == pytest.approx(0)
        assert results['model_mae'] == pytest.approx(0)


# Edge cases

class TestEdgeCases:
    def test_single_game_player_produces_no_predictions(self):
        df = make_gamelog([25])
        results = evaluate_points_mae(df)
        assert results['n_predictions'] == 0
        assert results['baseline_mae'] is None
        assert results['model_mae'] is None

    def test_empty_dataframe_produces_no_predictions(self):
        df = pd.DataFrame(columns=['PLAYER_NAME', 'GAME_DATE', 'PTS'])
        results = evaluate_points_mae(df)
        assert results['n_predictions'] == 0
        assert results['baseline_mae'] is None
        assert results['model_mae'] is None

    def test_rolling_window_larger_than_history_uses_all_prior_games(self):
        df = make_gamelog([10, 20, 30])
        result = compute_predictions(df, rolling_window=100)
        # With only 2 prior games max, rolling avg should equal baseline avg
        assert (result['model_pred'] == result['baseline_pred']).all()


# Failure cases

class TestFailureCases:
    def test_missing_required_column_raises(self):
        df = pd.DataFrame({'PLAYER_NAME': ['LeBron James'], 'GAME_DATE': ['2026-01-01']})
        with pytest.raises(ValueError):
            compute_predictions(df)

    def test_missing_player_column_raises(self):
        df = pd.DataFrame({'GAME_DATE': ['2026-01-01', '2026-01-02'], 'PTS': [10, 20]})
        with pytest.raises(ValueError):
            compute_predictions(df)
