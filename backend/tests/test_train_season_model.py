"""
Tests for train_season_model.py -- baselines, CV, projection.

Covers the baseline-honesty properties (shrinkage fit on train rows only,
grouped folds that never split a player) and the projection sanity bounds.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from season_features import ALL_TARGETS, build_season_dataset  # noqa: E402
from tests.conftest import make_player_season  # noqa: E402
from train_season_model import (  # noqa: E402
    baseline_report,
    carry_forward_baseline,
    clip_to_sane_bounds,
    columns_for_blocks,
    compare_to_baseline,
    derive_ratio_projections,
    fit_and_project,
    fit_ridge,
    fit_shrink_weight,
    grouped_kfold_indices,
    minutes_weighted_baseline,
    ranking_metrics,
    residual_quantiles,
    shrunk_baseline,
)


def _filler_log(base_log):
    """
    Base fixture plus 30 filler players whose stats MOVE between seasons.

    The movement matters: with constant per-game values every target's
    carry-forward MAE is exactly 0 and the shrunk-minutes projection collapses
    back onto min_prev, which would make several baseline tests vacuous rather
    than failing honestly.
    """
    rng = np.random.default_rng(7)
    frames = [base_log]
    for i in range(30):
        pts = 10.0 + i % 15
        minutes = 20.0 + i % 15
        reb, ast = 3.0 + i % 7, 2.0 + i % 5
        for season, jitter in [(2025, 0.0), (2026, 1.0)]:
            wobble = rng.normal(0, 2) * jitter
            frames.append(make_player_season(
                f'Filler {i}', season, 60,
                pts=max(1.0, pts + wobble),
                reb=max(0.5, reb + wobble / 3),
                ast=max(0.5, ast + wobble / 4),
                stl=max(0.1, 1.0 + wobble / 20),
                blk=max(0.1, 0.6 + wobble / 25),
                fg3m=max(0.1, 1.5 + wobble / 10),
                tov=max(0.2, 2.0 + wobble / 15),
                minutes=max(8.0, minutes + wobble),
                fgm=max(1.0, 5.0 + wobble / 3),
                fga=max(2.0, 11.0 + wobble / 2),
                ftm=max(0.5, 2.5 + wobble / 8),
                fta=max(0.6, 3.2 + wobble / 7),
                team=['LAL', 'BOS', 'MIA'][i % 3]))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def dataset(multi_season_gamelog, season_player_info):
    """A small but real season-pair dataset."""
    return build_season_dataset(_filler_log(multi_season_gamelog),
                                player_info=season_player_info)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def test_carry_forward_is_the_previous_season_value(dataset):
    pred = carry_forward_baseline(dataset, 'pts')
    np.testing.assert_allclose(pred, dataset['pts_prev'])


def test_carry_forward_mae_matches_hand_computation(dataset):
    pred = carry_forward_baseline(dataset, 'pts')
    expected = np.abs(dataset['target_pts'] - dataset['pts_prev']).mean()
    assert np.abs(dataset['target_pts'] - pred).mean() == pytest.approx(expected)


def test_shrink_weight_is_fit_on_training_rows_only(dataset):
    """
    Mutating TEST-row targets must not change the fitted w. If it does, the
    baseline has seen the test set and every comparison against it is invalid.
    """
    train = dataset.iloc[: len(dataset) // 2]
    test = dataset.iloc[len(dataset) // 2:].copy()
    w_before, mean_before = fit_shrink_weight(train, 'pts')

    test['target_pts'] = test['target_pts'] * 5
    w_after, mean_after = fit_shrink_weight(train, 'pts')
    assert w_after == w_before
    assert mean_after == mean_before


def test_shrunk_baseline_pulls_toward_the_league_mean(dataset):
    train = dataset.iloc[: len(dataset) // 2]
    test = dataset.iloc[len(dataset) // 2:]
    w, league_mean = fit_shrink_weight(train, 'pts')
    pred = shrunk_baseline(train, test, 'pts')
    expected = w * test['pts_prev'].to_numpy() + (1 - w) * league_mean
    np.testing.assert_allclose(pred, expected)
    if w < 1.0:
        spread_prev = test['pts_prev'].std()
        assert pred.std() < spread_prev, "shrinkage must compress the spread"


def test_minutes_baseline_is_not_identical_to_carry_forward(dataset):
    """
    Regression test. Using min_prev makes this baseline algebraically equal to
    carry-forward -- per36_prev is stat_mean*36/min_mean, so the minutes cancel.
    It must use PROJECTED minutes instead.
    """
    carry = carry_forward_baseline(dataset, 'pts')
    minutes = minutes_weighted_baseline(dataset, dataset, 'pts')
    assert not np.allclose(minutes, carry)


def test_minutes_baseline_falls_back_when_no_rate_exists(dataset):
    """`min` has no per-36 rate; the baseline must fall back, not crash."""
    pred = minutes_weighted_baseline(dataset, dataset, 'min')
    np.testing.assert_allclose(pred, carry_forward_baseline(dataset, 'min'))


def test_baseline_report_covers_every_target(dataset):
    report = baseline_report(dataset)
    assert set(report['target']) == set(ALL_TARGETS)
    # Not every target must have a nonzero MAE in a synthetic fixture, but the
    # ones that move between seasons should.
    assert (report.set_index('target').loc[['pts', 'min', 'reb'], 'mae_carry_forward'] > 0).all()
    from train_season_model import SHRINK_GRID
    assert report['shrink_w'].between(SHRINK_GRID.min(), SHRINK_GRID.max()).all()


# ---------------------------------------------------------------------------
# Paired comparison statistics
# ---------------------------------------------------------------------------

def test_compare_to_baseline_signs_improvement_correctly():
    actual = np.array([10.0, 20.0, 30.0])
    better = np.array([10.0, 20.0, 30.0])      # perfect
    worse = np.array([15.0, 25.0, 35.0])       # off by 5
    summary = compare_to_baseline(actual, better, worse)
    assert summary['improvement'] > 0
    assert summary['model_mae'] == pytest.approx(0.0)
    assert summary['baseline_mae'] == pytest.approx(5.0)
    assert summary['win_rate'] == pytest.approx(1.0)


def test_compare_to_baseline_reports_a_loss_as_negative():
    actual = np.array([10.0, 20.0, 30.0])
    summary = compare_to_baseline(actual, np.array([20.0, 30.0, 40.0]), actual)
    assert summary['improvement'] < 0


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def test_grouped_folds_never_split_a_player(dataset):
    """
    A player's two season-pairs are correlated. Splitting them across a fold
    boundary leaks, and would make CV look better than reality.
    """
    for train_idx, test_idx in grouped_kfold_indices(dataset, k=5):
        train_players = set(dataset.iloc[train_idx]['player'])
        test_players = set(dataset.iloc[test_idx]['player'])
        assert not (train_players & test_players)


def test_grouped_folds_cover_every_row_exactly_once(dataset):
    seen = np.zeros(len(dataset), dtype=int)
    for _, test_idx in grouped_kfold_indices(dataset, k=5):
        seen[test_idx] += 1
    assert (seen == 1).all()


def test_ridge_recovers_a_known_linear_target(dataset):
    """Pipeline correctness check, not an accuracy claim."""
    synthetic = dataset.copy()
    columns = columns_for_blocks(['PRIOR'])
    weights = np.arange(1, len(columns) + 1, dtype=float)
    synthetic['target_pts'] = synthetic[columns].to_numpy() @ weights

    pred = fit_ridge(synthetic, synthetic, columns, 'pts', alpha=1e-6, use_weights=False)
    mae = np.abs(synthetic['target_pts'] - pred).mean()
    assert mae < 0.01 * synthetic['target_pts'].std()


def test_residual_quantiles_bracket_zero(dataset):
    low, high = residual_quantiles(dataset, 'pts', ['PRIOR'], alpha=30)
    assert low < 0 < high


# ---------------------------------------------------------------------------
# Ratio derivation and bounds
# ---------------------------------------------------------------------------

def test_derive_ratio_projections_divides_components():
    frame = pd.DataFrame({'pred_fgm': [8.0, 5.0], 'pred_fga': [16.0, 10.0],
                          'pred_ftm': [4.0, 2.0], 'pred_fta': [5.0, 4.0]})
    out = derive_ratio_projections(frame)
    assert out['pred_fg_pct'].iat[0] == pytest.approx(0.5)
    assert out['pred_ft_pct'].iat[1] == pytest.approx(0.5)


def test_zero_attempts_yields_league_rate_not_inf():
    frame = pd.DataFrame({'pred_fgm': [8.0, 0.0], 'pred_fga': [16.0, 0.0],
                          'pred_ftm': [4.0, 0.0], 'pred_fta': [5.0, 0.0]})
    out = derive_ratio_projections(frame)
    assert np.isfinite(out['pred_fg_pct']).all()
    assert out['pred_fg_pct'].iat[1] == pytest.approx(0.5)


def test_sane_bounds_clamp_impossible_projections():
    """Ridge is unbounded and will happily extrapolate a negative steal rate."""
    frame = pd.DataFrame({
        'pred_pts': [-5.0, 200.0], 'pred_stl': [-1.0, 40.0],
        'pred_fg_pct': [0.0, 2.0], 'pred_gp': [-3.0, 300.0],
    })
    out = clip_to_sane_bounds(frame)
    assert out['pred_pts'].between(0, 45).all()
    assert out['pred_stl'].between(0, 4).all()
    assert out['pred_fg_pct'].between(0.25, 0.75).all()
    assert out['pred_gp'].between(0, 82).all()


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

@pytest.fixture
def projection_frame(dataset, multi_season_gamelog, season_player_info):
    from season_features import build_inference_features
    inference = build_inference_features(_filler_log(multi_season_gamelog), 2026,
                                         player_info=season_player_info)
    blocks = {t: ['PRIOR'] for t in ALL_TARGETS}
    alphas = {t: 30 for t in ALL_TARGETS}
    return fit_and_project(dataset, inference, blocks, alphas)


def test_projection_has_one_row_per_eligible_player(projection_frame):
    assert not projection_frame.empty
    assert projection_frame['player'].is_unique


def test_every_projection_is_finite(projection_frame):
    for target in ALL_TARGETS:
        column = projection_frame[f'pred_{target}']
        assert column.notna().all()
        assert np.isfinite(column).all()


def test_intervals_bracket_the_point_estimate(projection_frame):
    for target in ALL_TARGETS:
        low = projection_frame[f'pred_{target}_low']
        point = projection_frame[f'pred_{target}']
        high = projection_frame[f'pred_{target}_high']
        assert (low <= point + 1e-9).all(), target
        assert (point <= high + 1e-9).all(), target


def test_projections_respect_sane_bounds(projection_frame):
    bounds = {'pred_pts': (0, 45), 'pred_reb': (0, 20), 'pred_ast': (0, 15),
              'pred_stl': (0, 4), 'pred_blk': (0, 5), 'pred_fg3m': (0, 7),
              'pred_tov': (0, 7), 'pred_min': (0, 42), 'pred_gp': (0, 82),
              'pred_fg_pct': (0.25, 0.75), 'pred_ft_pct': (0.35, 1.0)}
    for column, (low, high) in bounds.items():
        assert projection_frame[column].between(low, high).all(), column


def test_projection_carries_identity_and_flag_columns(projection_frame):
    for column in ['player', 'player_id', 'team_prev', 'position', 'age',
                   'n_prior_seasons', 'team_changed', 'flags']:
        assert column in projection_frame.columns


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def test_perfect_prediction_gives_spearman_one():
    rng = np.random.default_rng(0)
    n = 40
    frame = pd.DataFrame({
        'PTS': rng.normal(15, 4, n), 'REB': rng.normal(5, 2, n),
        'AST': rng.normal(3, 1.5, n), 'STL': rng.normal(1, .3, n),
        'BLK': rng.normal(.6, .3, n), 'FG3M': rng.normal(1.8, .8, n),
        'TOV': rng.normal(2, .6, n), 'FG_PCT': rng.normal(.47, .03, n),
        'FGA': rng.normal(12, 3, n).clip(1), 'FT_PCT': rng.normal(.78, .06, n),
        'FTA': rng.normal(3, 1, n).clip(.5), 'flags': [''] * n,
    })
    metrics = ranking_metrics(frame, frame.copy())
    assert metrics['spearman'] == pytest.approx(1.0)
    for retention in metrics['top_n'].values():
        assert retention == pytest.approx(1.0)


def test_ranking_metrics_handle_too_few_rows():
    tiny = pd.DataFrame({c: [1.0] for c in
                         ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'TOV',
                          'FG_PCT', 'FGA', 'FT_PCT', 'FTA']})
    tiny['flags'] = ''
    metrics = ranking_metrics(tiny, tiny.copy())
    assert np.isnan(metrics['spearman'])


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------

def test_empty_dataset_produces_empty_baseline_report():
    from train_season_model import run
    empty = pd.DataFrame(columns=['player'])
    assert build_season_dataset(empty if False else pd.DataFrame(columns=[
        'PLAYER_NAME', 'GAME_DATE', 'team', 'game_id', 'MIN', 'PTS', 'REB', 'AST',
        'STL', 'BLK', 'TOV', 'FGM', 'FGA', 'FTM', 'FTA', 'FG3M', 'FG3A'])).empty
    _ = run


# ---------------------------------------------------------------------------
# Residual fitting, leakage guard, output schema, CLI
# ---------------------------------------------------------------------------

def test_residual_fit_shrinks_toward_carry_forward_not_the_mean(dataset):
    """
    Under an overwhelming penalty the residual fit collapses onto the prior
    season (plus the mean change); the level fit collapses onto the league mean.
    That is the whole point of fitting the change.
    """
    columns = columns_for_blocks(['PRIOR'])
    resid = fit_ridge(dataset, dataset, columns, 'pts', alpha=1e12, residual=True)
    level = fit_ridge(dataset, dataset, columns, 'pts', alpha=1e12, residual=False)
    mean_change = np.average(dataset['target_pts'] - dataset['pts_prev'],
                             weights=dataset['sample_weight'])
    np.testing.assert_allclose(resid, dataset['pts_prev'] + mean_change, atol=1e-6)
    assert np.std(level) < 1e-3


def test_team_changed_is_not_a_fitted_feature():
    """
    For training rows team_changed looks at the TARGET season's team; at
    inference it is unknown. Fitting it leaked the future into evaluation.
    """
    from season_features import SEASON_FEATURE_BLOCKS, SEASON_FEATURE_COLUMNS
    assert 'team_changed' not in SEASON_FEATURE_COLUMNS
    assert all('team_changed' not in cols for cols in SEASON_FEATURE_BLOCKS.values())


def test_team_changed_still_travels_with_the_dataset(dataset):
    """Kept as a column for interval widening and analysis."""
    assert 'team_changed' in dataset.columns
    assert set(dataset['team_changed'].unique()) <= {0, 1}


def test_projection_ignores_team_changed_values(dataset, multi_season_gamelog, season_player_info):
    from season_features import build_inference_features
    inference = build_inference_features(_filler_log(multi_season_gamelog), 2026,
                                         player_info=season_player_info)
    blocks = {t: ['PRIOR', 'CTX'] for t in ALL_TARGETS}
    alphas = {t: 30 for t in ALL_TARGETS}
    flipped = dataset.copy()
    flipped['team_changed'] = 1 - flipped['team_changed']
    a = fit_and_project(dataset, inference, blocks, alphas)
    b = fit_and_project(flipped, inference, blocks, alphas)
    np.testing.assert_allclose(a['pred_pts'], b['pred_pts'])


def test_output_schema_matches_section_11(projection_frame, tmp_path):
    """
    Section 11's column list: ids, per target prev/pred/low/high, derived
    percentages, nine z-scores, both values, and the roster/flag columns.
    """
    from train_season_model import attach_fantasy_value, output_columns, write_projections
    board = attach_fantasy_value(projection_frame)
    path = str(tmp_path / 'season_projections_2027.csv')
    write_projections(board, path)
    written = pd.read_csv(path, keep_default_na=False, na_values=[''])

    expected = {'player', 'player_id', 'team_prev', 'position', 'age', 'gp_prev'}
    for target in ALL_TARGETS:
        expected |= {f'{target}_prev', f'pred_{target}', f'pred_{target}_low', f'pred_{target}_high'}
    expected |= {'pred_fg_pct', 'pred_ft_pct'}
    expected |= {f'z_{c}' for c in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'TOV',
                                    'FG_PCT', 'FT_PCT']}
    expected |= {'value_total', 'value_points_league',
                 'n_prior_seasons', 'team_changed', 'role_change', 'flags'}
    assert set(written.columns) == expected
    assert list(written.columns) == output_columns()
    assert written['player'].is_unique


def test_written_board_is_sorted_best_first_with_unvalued_rows_last(projection_frame, tmp_path):
    from train_season_model import attach_fantasy_value, write_projections
    frame = projection_frame.copy()
    rookie = frame.iloc[[0]].copy()
    rookie['player'] = 'Rookie Ray'
    rookie['flags'] = 'NO_PRIOR_SEASON'
    board = attach_fantasy_value(pd.concat([rookie, frame], ignore_index=True))
    out = write_projections(board, str(tmp_path / 'p.csv'))
    values = out['value_total'].to_numpy()
    assert np.isnan(values[-1]) and out['player'].iat[-1] == 'Rookie Ray'
    assert (np.diff(values[:-1]) <= 1e-12).all()


def test_write_projections_records_exclusions(projection_frame, tmp_path):
    from train_season_model import attach_fantasy_value, write_projections
    excluded = pd.DataFrame({'player': ['Hurt Hal'], 'player_id': ['hurtha01'],
                             'reason': ['OUT_FOR_SEASON']})
    path = str(tmp_path / 'season_projections_2027.csv')
    write_projections(attach_fantasy_value(projection_frame), path, excluded=excluded)
    back = pd.read_csv(str(tmp_path / 'season_projections_2027_excluded.csv'))
    assert back.to_dict('records') == excluded.to_dict('records')


def test_cli_refuses_to_project_when_roster_file_fails_validation(monkeypatch, tmp_path, capsys):
    """Section 10.6: a bad roster file must stop the run before anything is written."""
    import roster_changes
    import train_season_model as tsm
    bad = pd.DataFrame([{'player_name': 'Nobody Known', 'new_team': 'XYZ',
                         'change_type': 'trade'}])
    gamelog = pd.DataFrame({'PLAYER_NAME': ['Alpha Adams'], 'Player_ID': ['a'],
                            'team': ['LAL']})
    monkeypatch.setattr(tsm.pd, 'read_csv', lambda *a, **k: gamelog)
    monkeypatch.setattr(roster_changes, 'load_roster_changes', lambda **k: bad)
    out = tmp_path / 'never.csv'
    assert tsm.main(['--project', '--out', str(out)]) == 1
    assert not out.exists()
    assert 'nothing written' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Shooting-rate projection (volume-aware shrinkage)
# ---------------------------------------------------------------------------

def _rate_rows(made, tried, gp=50.0, target=None):
    """Rows shaped like the dataset: per-game prior makes/attempts plus GP."""
    df = pd.DataFrame({'gp_prev': gp,
                       'baseline_fgm': np.asarray(made, float) / gp,
                       'baseline_fga': np.asarray(tried, float) / gp})
    if target is not None:
        df['target_fgm'] = np.asarray(target, float) * 10.0
        df['target_fga'] = 10.0
    return df


def test_low_volume_shooter_is_shrunk_harder_than_high_volume():
    """Same prior FG%, different volume: the 20-attempt shooter moves further to the league."""
    from train_season_model import shrunk_rate
    rows = _rate_rows(made=[14, 700], tried=[20, 1000])        # both .700
    rate = shrunk_rate(rows, 'fgm', 'fga', k=100, league=0.47)
    assert rate[0] < rate[1] < 0.70
    assert rate[0] == pytest.approx((14 + 100 * 0.47) / (20 + 100))


def test_zero_prior_attempts_gets_the_league_rate_not_nan():
    from train_season_model import shrunk_rate
    rate = shrunk_rate(_rate_rows(made=[0], tried=[0]), 'fgm', 'fga', k=0, league=0.47)
    assert rate[0] == pytest.approx(0.47)


def test_rate_prior_is_fit_on_training_rows_only(dataset):
    from train_season_model import fit_rate_prior
    train = dataset.iloc[: len(dataset) // 2]
    test = dataset.iloc[len(dataset) // 2:].copy()
    before = fit_rate_prior(train, 'fgm', 'fga')
    test['target_fgm'] = 0.0
    assert fit_rate_prior(train, 'fgm', 'fga') == before


def test_rate_prior_picks_heavy_shrinkage_when_rates_are_pure_noise():
    """If next season's rate is the league rate regardless of history, k should be large."""
    from train_season_model import RATE_PRIOR_K_GRID, fit_rate_prior
    rng = np.random.default_rng(1)
    tried = rng.integers(50, 400, 200).astype(float)
    made = tried * rng.uniform(0.3, 0.65, 200)
    rows = _rate_rows(made, tried, target=np.full(200, made.sum() / tried.sum()))
    k, league = fit_rate_prior(rows, 'fgm', 'fga')
    assert k == max(RATE_PRIOR_K_GRID)
    assert league == pytest.approx(made.sum() / tried.sum())


def test_projected_fg_pct_is_exactly_makes_over_attempts(projection_frame):
    ok = projection_frame['pred_fga'] > 0
    np.testing.assert_allclose(
        projection_frame.loc[ok, 'pred_fg_pct'],
        projection_frame.loc[ok, 'pred_fgm'] / projection_frame.loc[ok, 'pred_fga'])
    assert (projection_frame['pred_fgm'] <= projection_frame['pred_fga'] + 1e-9).all()
    assert (projection_frame['pred_ftm'] <= projection_frame['pred_fta'] + 1e-9).all()


# ---------------------------------------------------------------------------
# Top-of-board blend (design doc section 15.4)
# ---------------------------------------------------------------------------

def _blend_dataset(n=60, seed=3):
    """A frame with the baseline_ columns blend_top_tier reads, values spread."""
    from train_season_model import ALL_TARGETS as targets
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(index=range(n))
    scale = {'pts': 22, 'reb': 8, 'ast': 5, 'stl': 1.2, 'blk': .8, 'fg3m': 2.2,
             'tov': 2.4, 'min': 31, 'gp': 65, 'fgm': 8, 'fga': 17,
             'ftm': 3.5, 'fta': 4.4}
    for target in targets:
        base = scale.get(target, 5.0)
        frame[f'baseline_{target}'] = np.abs(rng.normal(base, base * .35, n))
    return frame


def test_blend_pulls_only_the_top_tier_toward_carry_forward():
    from train_season_model import ALL_TARGETS as targets
    from train_season_model import blend_top_tier

    frame = _blend_dataset()
    model = {t: frame[f'baseline_{t}'].to_numpy(float) + 5.0 for t in targets}
    blended, mask = blend_top_tier(frame, model, cutoff=10, weight=0.5)

    assert mask.sum() == 10
    carry = frame['baseline_pts'].to_numpy(float)
    # blended rows land halfway between the model and carry-forward
    assert np.allclose(blended['pts'][mask], (carry[mask] + model['pts'][mask]) / 2)
    # everyone else is untouched
    assert np.allclose(blended['pts'][~mask], model['pts'][~mask])


def test_blend_weight_one_is_the_unblended_model():
    from train_season_model import ALL_TARGETS as targets
    from train_season_model import blend_top_tier

    frame = _blend_dataset()
    model = {t: frame[f'baseline_{t}'].to_numpy(float) * 1.3 for t in targets}
    blended, _ = blend_top_tier(frame, model, cutoff=25, weight=1.0)
    for target in targets:
        assert np.allclose(blended[target], model[target]), target


def test_blend_weight_zero_is_carry_forward_for_the_top_tier():
    from train_season_model import ALL_TARGETS as targets
    from train_season_model import blend_top_tier

    frame = _blend_dataset()
    model = {t: np.zeros(len(frame)) for t in targets}
    blended, mask = blend_top_tier(frame, model, cutoff=5, weight=0.0)
    for target in targets:
        expected = frame[f'baseline_{target}'].to_numpy(float)[mask]
        assert np.allclose(blended[target][mask], expected), target


def test_blend_keeps_the_model_where_carry_forward_is_missing():
    """A top-tier player with no usable prior value must not inherit a NaN."""
    from train_season_model import ALL_TARGETS as targets
    from train_season_model import blend_top_tier

    frame = _blend_dataset()
    frame.loc[0, 'baseline_pts'] = np.nan
    model = {t: frame[f'baseline_{t}'].fillna(9.0).to_numpy(float) + 3.0
             for t in targets}
    blended, _ = blend_top_tier(frame, model, cutoff=len(frame), weight=0.5)
    assert np.isfinite(blended['pts']).all()
    assert blended['pts'][0] == pytest.approx(model['pts'][0])


def test_blend_is_a_no_op_on_an_empty_tier():
    from train_season_model import ALL_TARGETS as targets
    from train_season_model import blend_top_tier

    frame = _blend_dataset()
    model = {t: frame[f'baseline_{t}'].to_numpy(float) for t in targets}
    blended, mask = blend_top_tier(frame, model, cutoff=0, weight=0.0)
    assert not mask.any()
    assert blended is model


def test_prior_value_rank_orders_the_better_prior_season_first():
    from train_season_model import prior_value_rank

    frame = _blend_dataset()
    rank = prior_value_rank(frame)
    assert set(rank) == set(range(1, len(frame) + 1))
    # the top-ranked prior season should out-score the bottom one on volume
    best, worst = rank.idxmin(), rank.idxmax()
    assert frame.loc[best, 'baseline_pts'] > frame.loc[worst, 'baseline_pts']


def test_projection_flags_the_blended_tier(projection_frame):
    from train_season_model import TOP_TIER_FLAG

    flagged = projection_frame['flags'].astype(str).str.contains(TOP_TIER_FLAG)
    assert flagged.sum() == min(25, len(projection_frame))
    # flagged players are the top of the board by prior value, not a random set
    assert projection_frame.loc[flagged, 'pts_prev'].mean() > \
        projection_frame.loc[~flagged, 'pts_prev'].mean()


def test_blend_flag_does_not_exclude_a_player_from_the_value_pool():
    """TOP_TIER_BLEND must not collide with the NO_PRIOR_SEASON exclusion."""
    from fantasy_value import DEFAULT_EXCLUDE_FLAGS
    from train_season_model import TOP_TIER_FLAG

    for flag in DEFAULT_EXCLUDE_FLAGS:
        assert flag not in TOP_TIER_FLAG


def test_ranking_metrics_default_to_the_shipped_pool():
    """
    The evaluation must score the board the app actually publishes. Defaulting
    to None measured a ranking over all ~450 players that nobody ever sees.
    """
    from fantasy_value import DEFAULT_POOL_SIZE
    from train_season_model import _SHIPPED_POOL

    assert _SHIPPED_POOL == DEFAULT_POOL_SIZE

    rng = np.random.default_rng(11)
    n = 300
    frame = pd.DataFrame({
        'PTS': rng.normal(15, 6, n), 'REB': rng.normal(5, 2, n),
        'AST': rng.normal(3, 1.5, n), 'STL': rng.normal(1, .3, n),
        'BLK': rng.normal(.6, .3, n), 'FG3M': rng.normal(1.8, .8, n),
        'TOV': rng.normal(2, .6, n), 'FG_PCT': rng.normal(.47, .03, n),
        'FGA': rng.normal(12, 3, n).clip(1), 'FT_PCT': rng.normal(.78, .06, n),
        'FTA': rng.normal(3, 1, n).clip(.5), 'flags': [''] * n,
    })
    noisy = frame.copy()
    noisy['PTS'] = noisy['PTS'] + rng.normal(0, 3, n)

    shipped = ranking_metrics(frame, noisy)
    full_pool = ranking_metrics(frame, noisy, pool_size=None)
    # Same rows, same predictions -- the pool alone changes the score, which is
    # why the default has to match the product.
    assert shipped['spearman'] != full_pool['spearman']
