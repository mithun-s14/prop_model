"""
Tests for backend/train_engineered_model.py: splitting, feature selection,
fitting, and the paired baseline comparison.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from features import build_feature_dataset
from train_engineered_model import (
    FEATURE_BLOCKS,
    columns_for_blocks,
    compare_to_baseline,
    evaluate_split,
    fit_ridge,
    noise_floor_analysis,
    select_feature_blocks,
    split_by_season,
    split_holdout,
    split_temporal,
)
from tests.test_features import make_matchup_gamelog, POSITIONS


@pytest.fixture
def dataset():
    rng = np.random.default_rng(7)
    n = 30
    pts = (20 + rng.normal(0, 5, n)).round().clip(0).astype(int).tolist()
    gamelog = make_matchup_gamelog(n_games=n, home_pts=pts)
    return build_feature_dataset(gamelog, positions=POSITIONS, min_prior_games=5)


# Splitting

class TestSplits:
    def test_holdout_takes_last_game_per_player(self, dataset):
        train_df, test_df = split_holdout(dataset)
        assert len(test_df) == dataset['player'].nunique()
        for player, group in dataset.groupby('player'):
            assert test_df[test_df['player'] == player]['game_date'].iloc[0] == group['game_date'].max()

    def test_holdout_train_and_test_are_disjoint(self, dataset):
        train_df, test_df = split_holdout(dataset)
        assert len(train_df) + len(test_df) == len(dataset)

    def test_temporal_test_is_strictly_after_train(self, dataset):
        train_df, test_df = split_temporal(dataset, test_fraction=0.2)
        assert train_df['game_date'].max() < test_df['game_date'].min()

    def test_temporal_split_covers_all_rows(self, dataset):
        train_df, test_df = split_temporal(dataset, test_fraction=0.2)
        assert len(train_df) + len(test_df) == len(dataset)

    def test_by_season_split_holds_out_newest_season(self):
        frame = pd.DataFrame({
            'player': ['A'] * 4,
            'season': [2024, 2024, 2026, 2026],
            'game_date': pd.to_datetime(['2024-01-01', '2024-01-02', '2026-01-01', '2026-01-02']),
            'actual_pts': [10.0, 12.0, 14.0, 16.0],
            'baseline_pred': [11.0, 11.0, 15.0, 15.0],
        })
        train_df, test_df = split_by_season(frame)
        assert set(train_df['season']) == {2024}
        assert set(test_df['season']) == {2026}

    def test_by_season_split_empty_for_single_season(self, dataset):
        train_df, test_df = split_by_season(dataset)
        assert train_df.empty and test_df.empty

    def test_by_season_split_empty_without_season_column(self):
        frame = pd.DataFrame({'player': ['A'], 'actual_pts': [1.0], 'baseline_pred': [1.0]})
        train_df, test_df = split_by_season(frame)
        assert train_df.empty and test_df.empty

    def test_temporal_split_on_single_date_returns_empty(self):
        single = pd.DataFrame({
            'player': ['A'], 'game_date': [pd.Timestamp('2026-01-01')],
            'actual_pts': [10.0], 'baseline_pred': [11.0],
        })
        train_df, test_df = split_temporal(single)
        assert train_df.empty and test_df.empty


# Block helpers

class TestColumnsForBlocks:
    def test_flattens_blocks_in_order(self):
        cols = columns_for_blocks(['PTS', 'SHOT'])
        assert cols == FEATURE_BLOCKS['PTS'] + FEATURE_BLOCKS['SHOT']

    def test_unknown_block_raises(self):
        with pytest.raises(KeyError):
            columns_for_blocks(['NOT_A_BLOCK'])


# Paired baseline comparison

class TestCompareToBaseline:
    def test_perfect_model_beats_baseline(self):
        actual = [10.0, 20.0, 30.0, 40.0]
        summary = compare_to_baseline(actual, model_pred=actual, baseline_pred=[12.0, 18.0, 33.0, 37.0])
        assert summary['model_mae'] == pytest.approx(0.0)
        assert summary['improvement'] > 0
        assert summary['win_rate'] == 1.0

    def test_identical_predictions_show_no_improvement(self):
        actual = [10.0, 20.0, 30.0, 40.0]
        preds = [11.0, 19.0, 31.0, 39.0]
        summary = compare_to_baseline(actual, model_pred=preds, baseline_pred=preds)
        assert summary['improvement'] == pytest.approx(0.0)
        assert summary['win_rate'] == 0.0
        assert summary['baseline_mae'] == pytest.approx(summary['model_mae'])

    def test_worse_model_reports_negative_improvement(self):
        actual = [10.0, 20.0, 30.0]
        summary = compare_to_baseline(actual, model_pred=[0.0, 0.0, 0.0], baseline_pred=actual)
        assert summary['improvement'] < 0

    def test_confidence_interval_brackets_improvement(self):
        rng = np.random.default_rng(0)
        actual = rng.normal(20, 5, 200)
        summary = compare_to_baseline(actual, actual + rng.normal(0, 1, 200), actual + rng.normal(0, 3, 200))
        assert summary['ci_low'] <= summary['improvement'] <= summary['ci_high']
        assert summary['n_test'] == 200


# Fitting and evaluation

class TestFitAndEvaluate:
    def test_fit_ridge_returns_one_prediction_per_test_row(self, dataset):
        train_df, test_df = split_temporal(dataset, test_fraction=0.3)
        preds = fit_ridge(train_df, test_df, columns_for_blocks(['PTS']), alpha=30)
        assert len(preds) == len(test_df)
        assert np.isfinite(preds).all()

    def test_evaluate_split_returns_summary_and_predictions(self, dataset):
        train_df, test_df = split_temporal(dataset, test_fraction=0.3)
        summary, results_df = evaluate_split(train_df, test_df, blocks=['PTS'], alpha=30)
        assert summary['n_train'] == len(train_df)
        assert summary['n_test'] == len(test_df)
        assert summary['model_mae'] >= 0
        assert set(results_df.columns) == {'player', 'game_date', 'actual', 'baseline_pred', 'model_pred'}

    def test_evaluate_split_handles_empty_train(self, dataset):
        empty = dataset.iloc[0:0]
        summary, results_df = evaluate_split(empty, dataset, blocks=['PTS'])
        assert summary['baseline_mae'] is None
        assert results_df.empty

    def test_select_feature_blocks_returns_valid_choice(self, dataset):
        train_df, val_df = split_temporal(dataset, test_fraction=0.3)
        blocks, alpha, val_mae = select_feature_blocks(train_df, val_df)
        assert 'PTS' in blocks  # the seed block is always retained
        assert all(b in FEATURE_BLOCKS for b in blocks)
        assert alpha in [3, 10, 30, 100, 300, 1000]
        assert val_mae >= 0

    def test_selection_never_worsens_seed_validation_mae(self, dataset):
        """Greedy selection only accepts a block if it improves validation MAE."""
        train_df, val_df = split_temporal(dataset, test_fraction=0.3)
        blocks, alpha, val_mae = select_feature_blocks(train_df, val_df)
        from sklearn.metrics import mean_absolute_error
        seed_best = min(
            mean_absolute_error(val_df['actual_pts'], fit_ridge(train_df, val_df, columns_for_blocks(['PTS']), a))
            for a in [3, 10, 30, 100, 300, 1000]
        )
        assert val_mae <= seed_best + 1e-6


# Noise floor

class TestNoiseFloorAnalysis:
    def test_oracles_beat_the_naive_baseline(self, dataset):
        gamelog = make_matchup_gamelog(n_games=30)
        _, test_df = split_temporal(dataset, test_fraction=0.3)
        floor = noise_floor_analysis(gamelog, test_df)
        # A leaky oracle that knows the test-period mean must beat the naive baseline
        assert floor['oracle_test_period_mae'] <= floor['baseline_mae']
        assert floor['mean_within_player_std'] > 0

    def test_oracle_is_season_aware(self):
        """
        A player who averages 40 one season and 10 the next must not be scored
        against a 25-point career mean. Averaging across seasons makes the
        'oracle' worse than the naive baseline and headroom go negative.
        """
        rows = []
        for season_start, pts in (('2024-11-01', 40), ('2025-11-01', 10)):
            for date in pd.date_range(start=season_start, periods=8, freq='D'):
                rows.append({'PLAYER_NAME': 'Swingy Player', 'PTS': pts,
                             'GAME_DATE': date.strftime('%Y-%m-%d')})
        gamelog = pd.DataFrame(rows)

        test_df = pd.DataFrame({
            'player': ['Swingy Player'] * 3,
            'season': [2026] * 3,
            'actual_pts': [10.0, 10.0, 10.0],
            'baseline_pred': [10.0, 10.0, 10.0],
        })

        floor = noise_floor_analysis(gamelog, test_df)
        # Season-aware oracle predicts 10 for the 2026 rows -> zero error.
        # A career-mean oracle would predict 25 and score 15.0 MAE.
        assert floor['oracle_full_season_mae'] == pytest.approx(0.0)
        assert floor['oracle_full_season_mae'] <= floor['baseline_mae']

    def test_gaussian_floor_scales_with_volatility(self, dataset):
        gamelog = make_matchup_gamelog(n_games=30)
        _, test_df = split_temporal(dataset, test_fraction=0.3)
        floor = noise_floor_analysis(gamelog, test_df)
        assert floor['gaussian_mae_floor'] == pytest.approx(0.7979 * floor['mean_within_player_std'], rel=1e-6)
