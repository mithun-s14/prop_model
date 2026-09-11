"""
Tests for season_features.py -- preseason season-level projection features.

Two tests here carry most of the weight and are marked as such in the design
doc (markdowns/season_projection_model.md section 12):

  test_ratio_stats_use_sum_of_makes_over_attempts
      Guards the central modelling decision: FG%/FT% are sum(makes)/sum(attempts),
      never the mean of per-game percentages.

  test_features_are_leakage_free_against_target_season
      Guards point-in-time correctness: mutating a player's target-season games
      must leave their feature row byte-identical.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from season_features import (  # noqa: E402
    ALL_TARGETS,
    FULL_SEASON_GP,
    MIN_GP_FEATURE,
    MIN_GP_TARGET,
    SEASON_FEATURE_COLUMNS,
    aggregate_player_seasons,
    build_inference_features,
    build_season_dataset,
    load_positions,
)
from tests.conftest import make_player_season  # noqa: E402


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_aggregate_produces_one_row_per_player_season(multi_season_gamelog):
    agg = aggregate_player_seasons(multi_season_gamelog)
    assert len(agg) == 10
    assert set(agg['season']) == {2024, 2025, 2026}
    assert agg.groupby(['player', 'season']).size().max() == 1


def test_season_averages_match_hand_computed_values(multi_season_gamelog):
    agg = aggregate_player_seasons(multi_season_gamelog)
    sam = agg[(agg['player'] == 'Steady Sam') & (agg['season'] == 2025)].iloc[0]
    # The fixture uses constant per-game values, so the mean is the value itself
    assert sam['pts_mean'] == pytest.approx(20.0)
    assert sam['gp'] == 70
    assert sam['min_mean'] == pytest.approx(30.0)


def test_dataset_pairs_consecutive_seasons(multi_season_gamelog, season_player_info):
    ds = build_season_dataset(multi_season_gamelog, player_info=season_player_info)
    pairs = set(zip(ds['feature_season'], ds['target_season']))
    assert pairs == {(2024, 2025), (2025, 2026)}
    sam = ds[(ds['player'] == 'Steady Sam') & (ds['target_season'] == 2026)].iloc[0]
    assert sam['pts_prev'] == pytest.approx(20.0)
    assert sam['target_pts'] == pytest.approx(20.0)
    assert sam['baseline_pts'] == pytest.approx(sam['pts_prev'])


def test_per36_rates_scale_with_minutes(multi_season_gamelog):
    agg = aggregate_player_seasons(multi_season_gamelog)
    rick = agg[(agg['player'] == 'Rising Rick') & (agg['season'] == 2024)].iloc[0]
    # 10 pts in 20 minutes -> 18 per 36
    assert rick['pts_per36'] == pytest.approx(10 * 36 / 20)


# ---------------------------------------------------------------------------
# The ratio-stat decision (design doc section 4)
# ---------------------------------------------------------------------------

def test_ratio_stats_use_sum_of_makes_over_attempts():
    """
    An 0-for-2 game and a 12-for-20 game must give 12/22 == 0.545, NOT the mean
    of per-game percentages ((0.0 + 0.6)/2 == 0.30).

    This is the highest-value test in the suite: the naive form is what the
    `FG_PCT` column in the game log invites, and on real data it is wrong by
    0.0415 on average -- more than half the entire year-over-year spread of FG%.
    """
    log = pd.DataFrame({
        'PLAYER_NAME': ['Split Shooter'] * 2,
        'Player_ID': ['splitsh'] * 2,
        'GAME_DATE': ['Wed, Oct 23, 2024', 'Fri, Oct 25, 2024'],
        'team': ['LAL'] * 2, 'game_id': ['g1', 'g2'],
        'MIN': [20, 30], 'FGM': [0, 12], 'FGA': [2, 20],
        'FG3M': [0, 2], 'FG3A': [1, 4], 'FTM': [0, 6], 'FTA': [0, 8],
        'REB': [1, 5], 'AST': [1, 2], 'STL': [0, 1], 'BLK': [0, 1],
        'TOV': [1, 2], 'PTS': [0, 32],
    })
    agg = aggregate_player_seasons(log)
    assert agg['fg_pct'].iat[0] == pytest.approx(12 / 22)
    assert agg['fg_pct'].iat[0] != pytest.approx((0.0 + 0.6) / 2)
    assert agg['ft_pct'].iat[0] == pytest.approx(6 / 8)
    assert agg['fg3_pct'].iat[0] == pytest.approx(2 / 5)


def test_true_shooting_uses_standard_formula():
    log = make_player_season('TS Guy', 2025, 10, pts=25, fgm=9, fga=18, ftm=5, fta=6)
    agg = aggregate_player_seasons(log)
    expected = (25 * 10) / (2 * (18 * 10 + 0.44 * 6 * 10))
    assert agg['ts_pct'].iat[0] == pytest.approx(expected)


def test_zero_free_throw_attempts_is_imputed_not_zero(multi_season_gamelog,
                                                      season_player_info):
    """
    A season with no free throw attempts has an undefined FT%. It must not
    become 0.0 -- a 0% free-throw shooter would be catastrophically mis-valued
    in a fantasy percentage category. One such season exists in the real data.
    """
    no_ft = pd.concat([
        make_player_season('No Freebies', 2025, 40, ftm=0, fta=0),
        make_player_season('No Freebies', 2026, 40, ftm=0, fta=0),
    ], ignore_index=True)
    log = pd.concat([multi_season_gamelog, no_ft], ignore_index=True)

    agg = aggregate_player_seasons(no_ft)
    assert np.isnan(agg['ft_pct'].iat[0]), "raw aggregate should expose the undefined rate"

    ds = build_season_dataset(log, player_info=season_player_info)
    row = ds[ds['player'] == 'No Freebies'].iloc[0]
    assert row['ft_pct_prev_missing'] == 1
    assert row['ft_pct_prev'] > 0.0, "undefined FT% must be imputed, not zeroed"
    assert not np.isnan(row['ft_pct_prev'])


# ---------------------------------------------------------------------------
# Leakage (design doc section 12)
# ---------------------------------------------------------------------------

def test_features_are_leakage_free_against_target_season(multi_season_gamelog,
                                                         season_player_info):
    """
    Mutating every target-season game for a player must leave that player's
    feature row byte-identical while only the targets move.

    This is the structural guarantee the whole model rests on. It is also what
    would catch the current-season snapshot files (usage rates, defense Excel)
    being wired in as features.
    """
    base = build_season_dataset(multi_season_gamelog, player_info=season_player_info)

    mutated_log = multi_season_gamelog.copy()
    dates = pd.to_datetime(mutated_log['GAME_DATE'], format='mixed')
    is_2026 = (dates.dt.year + (dates.dt.month >= 10).astype(int)) == 2026
    mask = (mutated_log['PLAYER_NAME'] == 'Steady Sam') & is_2026
    for col in ['PTS', 'REB', 'AST', 'MIN', 'FGM', 'FGA', 'FTM', 'FTA']:
        mutated_log.loc[mask, col] = mutated_log.loc[mask, col] * 3
    mutated = build_season_dataset(mutated_log, player_info=season_player_info)

    def feature_row(ds):
        sel = (ds['player'] == 'Steady Sam') & (ds['target_season'] == 2026)
        return ds[sel][SEASON_FEATURE_COLUMNS].reset_index(drop=True)

    pd.testing.assert_frame_equal(feature_row(base), feature_row(mutated))

    def target(ds):
        sel = (ds['player'] == 'Steady Sam') & (ds['target_season'] == 2026)
        return ds[sel]['target_pts'].iat[0]

    assert target(mutated) == pytest.approx(target(base) * 3)


def test_prior_season_row_is_unaffected_by_later_seasons(multi_season_gamelog,
                                                         season_player_info):
    """The 2024->2025 row must not shift when 2026 data changes."""
    base = build_season_dataset(multi_season_gamelog, player_info=season_player_info)
    trimmed_log = multi_season_gamelog.copy()
    dates = pd.to_datetime(trimmed_log['GAME_DATE'], format='mixed')
    season = dates.dt.year + (dates.dt.month >= 10).astype(int)
    trimmed = build_season_dataset(trimmed_log[season <= 2025], player_info=season_player_info)

    cols = SEASON_FEATURE_COLUMNS + ['target_pts']
    sel = lambda ds: (  # noqa: E731
        ds[(ds['player'] == 'Steady Sam') & (ds['target_season'] == 2025)][cols]
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(sel(base), sel(trimmed))


# ---------------------------------------------------------------------------
# Eligibility, weighting, season boundaries
# ---------------------------------------------------------------------------

def test_asymmetric_thresholds_drop_short_feature_seasons(multi_season_gamelog,
                                                          season_player_info):
    """
    18 GP in the FEATURE season is dropped (too little history to profile).
    18 GP in the TARGET season is KEPT -- a higher target bar would induce
    survivorship bias by discarding players who got hurt.
    """
    ds = build_season_dataset(multi_season_gamelog, player_info=season_player_info)
    steve = ds[ds['player'] == 'Short Steve']
    # Steve has 40 GP in 2025 (feature ok) and 18 GP in 2026 (target ok at 15)
    assert len(steve) == 1
    assert steve.iloc[0]['target_season'] == 2026
    assert steve.iloc[0]['target_gp'] == 18

    # ...but 2026 (18 GP) is too short to be a FEATURE season, so no 2026->2027 row
    assert not ((ds['player'] == 'Short Steve') & (ds['feature_season'] == 2026)).any()


def test_sample_weight_matches_documented_formula(multi_season_gamelog,
                                                  season_player_info):
    ds = build_season_dataset(multi_season_gamelog, player_info=season_player_info)
    expected = ds['target_gp'].clip(upper=FULL_SEASON_GP) / FULL_SEASON_GP
    np.testing.assert_allclose(ds['sample_weight'], expected)
    assert ds['sample_weight'].max() <= 1.0
    steve = ds[ds['player'] == 'Short Steve'].iloc[0]
    assert steve['sample_weight'] == pytest.approx(18 / FULL_SEASON_GP)


def test_threshold_boundaries_are_inclusive():
    """Exactly MIN_GP_FEATURE is kept; one game fewer is dropped."""
    exact = pd.concat([
        make_player_season('Exact Eddie', 2025, MIN_GP_FEATURE),
        make_player_season('Exact Eddie', 2026, MIN_GP_TARGET),
        make_player_season('Under Uma', 2025, MIN_GP_FEATURE - 1),
        make_player_season('Under Uma', 2026, 60),
    ], ignore_index=True)
    ds = build_season_dataset(exact)
    assert 'Exact Eddie' in set(ds['player'])
    assert 'Under Uma' not in set(ds['player'])


def test_october_games_belong_to_the_following_season():
    """Sep 30 and Oct 1 fall in different NBA seasons."""
    log = pd.concat([
        make_player_season('Boundary Bob', 2025, 30, start_month=10),   # Oct 2024 -> 2025
        make_player_season('Boundary Bob', 2026, 30, start_month=10),   # Oct 2025 -> 2026
    ], ignore_index=True)
    agg = aggregate_player_seasons(log)
    assert set(agg['season']) == {2025, 2026}


def test_gap_season_produces_no_pair():
    """A player present in N and N+2 but not N+1 yields no training row."""
    log = pd.concat([
        make_player_season('Gap Gary', 2024, 60),
        make_player_season('Gap Gary', 2026, 60),
    ], ignore_index=True)
    ds = build_season_dataset(log)
    assert 'Gap Gary' not in set(ds['player'])


# ---------------------------------------------------------------------------
# Team changes (derived, not manual)
# ---------------------------------------------------------------------------

def test_team_change_is_derived_from_the_game_logs(multi_season_gamelog,
                                                   season_player_info):
    ds = build_season_dataset(multi_season_gamelog, player_info=season_player_info)
    mike = ds[(ds['player'] == 'Moving Mike') & (ds['target_season'] == 2026)].iloc[0]
    sam = ds[(ds['player'] == 'Steady Sam') & (ds['target_season'] == 2026)].iloc[0]
    assert mike['team_changed'] == 1
    assert mike['team_prev'] == 'MIA'
    assert sam['team_changed'] == 0


def test_midseason_trade_sets_n_teams_and_primary_team():
    """Primary team = the one the player played the most games for."""
    log = pd.concat([
        make_player_season('Traded Tom', 2025, 20, team='POR'),
        make_player_season('Traded Tom', 2025, 45, team='SAC', start_month=12),
        make_player_season('Traded Tom', 2026, 60, team='SAC'),
    ], ignore_index=True)
    ds = build_season_dataset(log)
    row = ds.iloc[0]
    assert row['n_teams_prev'] == 2
    assert row['team_prev'] == 'SAC', "primary team is the one with more games"
    assert row['team_changed'] == 0, "SAC -> SAC is not a change"


# ---------------------------------------------------------------------------
# Age, position, missing data
# ---------------------------------------------------------------------------

def test_missing_player_info_flags_and_imputes_age(multi_season_gamelog,
                                                   season_player_info):
    ds = build_season_dataset(multi_season_gamelog, player_info=season_player_info)
    steve = ds[ds['player'] == 'Short Steve'].iloc[0]     # absent from player_info
    sam = ds[ds['player'] == 'Steady Sam'].iloc[0]
    assert steve['age_missing'] == 1
    assert steve['age'] > 0, "missing age must be imputed, never zero-filled"
    assert not np.isnan(steve['age'])
    assert sam['age_missing'] == 0
    assert 25 < sam['age'] < 35


def test_rookie_experience_parses_to_zero(multi_season_gamelog, season_player_info):
    ds = build_season_dataset(multi_season_gamelog, player_info=season_player_info)
    rick = ds[ds['player'] == 'Rising Rick'].iloc[0]      # experience 'R'
    assert rick['experience_years'] == pytest.approx(0.0)


def test_positions_are_whitespace_stripped(multi_season_gamelog, season_player_info,
                                           season_positions):
    """The real players_positions.csv contains ' PF' and ' PG' with leading spaces."""
    ds = build_season_dataset(multi_season_gamelog, player_info=season_player_info,
                              positions=season_positions)
    rick = ds[ds['player'] == 'Rising Rick'].iloc[0]      # fixture position is ' SG'
    assert rick['position'] == 'SG'
    assert rick['pos_SG'] == 1
    assert rick['pos_PG'] == 0


def test_unknown_position_gets_no_onehot(multi_season_gamelog, season_player_info,
                                         season_positions):
    ds = build_season_dataset(multi_season_gamelog, player_info=season_player_info,
                              positions=season_positions)
    steve = ds[ds['player'] == 'Short Steve'].iloc[0]     # absent from the position map
    assert steve['position'] == 'UNK'
    assert sum(steve[f'pos_{p}'] for p in ['PG', 'SG', 'SF', 'PF', 'C']) == 0


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def test_no_nan_or_inf_in_feature_columns(multi_season_gamelog, season_player_info):
    ds = build_season_dataset(multi_season_gamelog, player_info=season_player_info)
    features = ds[SEASON_FEATURE_COLUMNS]
    assert not features.isna().any().any()
    assert np.isfinite(features.to_numpy(dtype=float)).all()


def test_all_targets_and_baselines_present(multi_season_gamelog, season_player_info):
    ds = build_season_dataset(multi_season_gamelog, player_info=season_player_info)
    for target in ALL_TARGETS:
        assert f'target_{target}' in ds.columns
        assert f'baseline_{target}' in ds.columns


# ---------------------------------------------------------------------------
# Inference path
# ---------------------------------------------------------------------------

def test_inference_features_match_training_feature_columns(multi_season_gamelog,
                                                           season_player_info):
    """
    Training and inference must produce identical feature columns -- a train/serve
    skew at this sample size would be undetectable and would poison every
    projection silently.
    """
    ds = build_season_dataset(multi_season_gamelog, player_info=season_player_info)
    inf = build_inference_features(multi_season_gamelog, feature_season=2026,
                                   player_info=season_player_info)
    assert list(inf[SEASON_FEATURE_COLUMNS].columns) == list(ds[SEASON_FEATURE_COLUMNS].columns)
    assert not inf.empty
    assert (inf['target_season'] == 2027).all()


def test_inference_feature_values_equal_training_values(multi_season_gamelog,
                                                        season_player_info):
    """
    The 2025 feature row built for inference must equal the 2025 feature row
    built for training (same player, same season, same numbers).
    """
    ds = build_season_dataset(multi_season_gamelog, player_info=season_player_info)
    inf = build_inference_features(multi_season_gamelog, feature_season=2025,
                                   player_info=season_player_info)
    shared = ['pts_prev', 'min_prev', 'fg_pct_prev', 'pts_per36_prev', 'fga_share_prev']
    train_row = ds[(ds['player'] == 'Steady Sam') & (ds['feature_season'] == 2025)].iloc[0]
    inf_row = inf[inf['player'] == 'Steady Sam'].iloc[0]
    for col in shared:
        assert inf_row[col] == pytest.approx(train_row[col]), col


def test_inference_on_empty_season_returns_empty_frame(multi_season_gamelog):
    inf = build_inference_features(multi_season_gamelog, feature_season=2099)
    assert inf.empty
    assert list(inf.columns)[:2] == ['player', 'player_id']


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

def test_missing_required_column_raises_naming_it(multi_season_gamelog):
    broken = multi_season_gamelog.drop(columns=['FGA'])
    with pytest.raises(ValueError, match='FGA'):
        aggregate_player_seasons(broken)


def test_single_season_returns_empty_frame_not_an_exception():
    log = make_player_season('Lonely Lou', 2026, 60)
    ds = build_season_dataset(log)
    assert ds.empty
    assert 'player' in ds.columns
    for target in ALL_TARGETS:
        assert f'target_{target}' in ds.columns


def test_empty_gamelog_returns_empty_frame():
    empty = pd.DataFrame(columns=[
        'PLAYER_NAME', 'GAME_DATE', 'team', 'game_id', 'MIN', 'PTS', 'REB', 'AST',
        'STL', 'BLK', 'TOV', 'FGM', 'FGA', 'FTM', 'FTA', 'FG3M', 'FG3A',
    ])
    assert build_season_dataset(empty).empty


def test_unparseable_dates_are_dropped_not_fatal(multi_season_gamelog):
    log = multi_season_gamelog.copy()
    log.loc[0, 'GAME_DATE'] = 'not a date'
    agg = aggregate_player_seasons(log)
    assert not agg.empty


def test_missing_positions_file_yields_unknown(tmp_path):
    assert load_positions(tmp_path / 'nope.csv') == {}


# ---------------------------------------------------------------------------
# Partial-scrape detection
# ---------------------------------------------------------------------------

def test_partially_scraped_season_is_excluded(multi_season_gamelog):
    """
    A season holding far fewer players than its neighbours is a partial scrape.

    It must be dropped rather than used: team-level quantities like fga_share
    are summed over whoever is present, so a half-scraped season silently
    inflates every remaining player's share instead of leaving a visible gap.
    """
    partial = make_player_season('Lonely Straggler', 2027, 60, team='LAL')
    log = pd.concat([multi_season_gamelog, partial], ignore_index=True)
    ds = build_season_dataset(log)
    assert 2027 not in set(ds['target_season']), "partial season must not become a target"
    assert 2027 not in set(ds['feature_season'])


def test_completeness_guard_can_be_disabled(multi_season_gamelog):
    """With the guard off, the partial season's pair is retained."""
    log = pd.concat([
        multi_season_gamelog,
        make_player_season('Lonely Straggler', 2026, 60),   # pairs into 2027
        make_player_season('Lonely Straggler', 2027, 60),   # the partial season
    ], ignore_index=True)
    guarded = build_season_dataset(log)
    assert 2027 not in set(guarded['target_season'])

    unguarded = build_season_dataset(log, require_complete_seasons=False)
    assert 2027 in set(unguarded['target_season'])


def test_equal_sized_seasons_are_all_kept(multi_season_gamelog):
    """The guard is relative, so a small-but-consistent dataset is untouched."""
    ds = build_season_dataset(multi_season_gamelog)
    assert set(ds['target_season']) == {2025, 2026}
