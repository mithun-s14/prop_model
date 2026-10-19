"""
Tests for fantasy_value.py -- z-score aggregation for a draft board.

Two tests here guard decisions that are silent when wrong (design doc sections
7 and 10.8): percentage categories must be volume-weighted, and unprojected
rows must not enter the z-score pool.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fantasy_value import (  # noqa: E402
    DEFAULT_POINTS_FORMULA,
    NINE_CAT,
    build_value_frame,
    category_zscores,
    points_league_value,
    total_value,
)


def make_pool(n=60, seed=0):
    """A realistic projected player pool."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'player': [f'Player {i}' for i in range(n)],
        'PTS': rng.normal(15, 4, n).clip(0),
        'REB': rng.normal(5, 2, n).clip(0),
        'AST': rng.normal(3, 1.5, n).clip(0),
        'STL': rng.normal(1.0, 0.3, n).clip(0),
        'BLK': rng.normal(0.6, 0.3, n).clip(0),
        'FG3M': rng.normal(1.8, 0.8, n).clip(0),
        'TOV': rng.normal(2.0, 0.6, n).clip(0),
        'FG_PCT': rng.normal(0.47, 0.03, n),
        'FGA': rng.normal(12, 3, n).clip(1),
        'FT_PCT': rng.normal(0.78, 0.06, n),
        'FTA': rng.normal(3, 1, n).clip(0.5),
        'flags': [''] * n,
    })


# ---------------------------------------------------------------------------
# Z-score mechanics
# ---------------------------------------------------------------------------

def test_zscores_are_standardized_over_the_pool():
    z = category_zscores(make_pool(), pool_size=None)
    for cat in NINE_CAT:
        assert z[f'z_{cat}'].mean() == pytest.approx(0.0, abs=1e-9)
        assert z[f'z_{cat}'].std() == pytest.approx(1.0, abs=1e-6)


def test_turnovers_are_negated():
    """
    More turnovers must mean a LOWER z. A sign error here is silent and ruins
    a draft board by promoting the most turnover-prone players.
    """
    pool = make_pool()
    z = category_zscores(pool, pool_size=None)
    worst = pool['TOV'].idxmax()
    best = pool['TOV'].idxmin()
    assert z['z_TOV'][worst] == z['z_TOV'].min()
    assert z['z_TOV'][best] == z['z_TOV'].max()


def test_percentage_categories_are_volume_weighted():
    """
    The design doc's example: .620 on 4 FGA vs .580 on 18 FGA, in a pool whose
    attempt-weighted league rate is ~.470. The high-volume shooter must rank
    higher -- a rate-only implementation ranks the low-volume one higher and is
    the most common flaw in naive fantasy value calculations.
    """
    pool = make_pool()
    pool.loc[0, ['FG_PCT', 'FGA']] = [0.620, 4.0]
    pool.loc[1, ['FG_PCT', 'FGA']] = [0.580, 18.0]
    z = category_zscores(pool, pool_size=None)
    assert z['z_FG_PCT'][1] > z['z_FG_PCT'][0]
    # And both beat a below-average high-volume shooter
    pool.loc[2, ['FG_PCT', 'FGA']] = [0.400, 18.0]
    z = category_zscores(pool, pool_size=None)
    assert z['z_FG_PCT'][2] < z['z_FG_PCT'][1]


def test_low_volume_high_percentage_does_not_dominate():
    """A .900 shooter on 1 attempt must not outrank a .550 shooter on 20."""
    pool = make_pool()
    pool.loc[0, ['FG_PCT', 'FGA']] = [0.900, 1.0]
    pool.loc[1, ['FG_PCT', 'FGA']] = [0.550, 20.0]
    z = category_zscores(pool, pool_size=None)
    assert z['z_FG_PCT'][1] > z['z_FG_PCT'][0]


def test_pool_size_changes_zscores():
    """
    Z over all projected players differs from z over the replacement-level pool,
    and the replacement-level pool is the correct one for draft valuation.
    """
    pool = make_pool(n=100)
    narrow = category_zscores(pool, pool_size=20)
    wide = category_zscores(pool, pool_size=None)
    assert not np.allclose(narrow['z_PTS'], wide['z_PTS'])


def test_zero_attempts_does_not_produce_nan():
    pool = make_pool()
    pool.loc[0, ['FTA', 'FT_PCT']] = [0.0, 0.0]
    z = category_zscores(pool, pool_size=None)
    assert not np.isnan(z['z_FT_PCT'][0])


# ---------------------------------------------------------------------------
# Totals and punting
# ---------------------------------------------------------------------------

def test_total_value_sums_all_nine_categories():
    pool = make_pool()
    z = category_zscores(pool, pool_size=None)
    total = total_value(z)
    expected = z[[f'z_{c}' for c in NINE_CAT]].sum(axis=1)
    np.testing.assert_allclose(total, expected)


def test_punting_drops_categories_and_reorders_the_board():
    """
    A punt is just a re-sum over a subset -- which is why the output ships the
    nine individual z-scores, not only the total.
    """
    pool = make_pool()
    z = category_zscores(pool, pool_size=None)
    full = total_value(z)
    punted = total_value(z, punt=['FT_PCT', 'TOV'])
    expected = z[[f'z_{c}' for c in NINE_CAT if c not in {'FT_PCT', 'TOV'}]].sum(axis=1)
    np.testing.assert_allclose(punted, expected)
    assert not full.rank().equals(punted.rank()), "punting should reorder the board"


def test_unknown_punt_category_is_ignored_not_fatal():
    z = category_zscores(make_pool(), pool_size=None)
    assert total_value(z, punt=['NOT_A_CATEGORY']).notna().all()


# ---------------------------------------------------------------------------
# Unprojected rows (design doc section 10.8)
# ---------------------------------------------------------------------------

def test_flagged_rows_do_not_shift_other_players_zscores():
    """
    THE rookie safeguard. Adding 60 zero-stat rows flagged NO_PRIOR_SEASON must
    leave every real player's z-scores identical. Without the exclusion this
    shifts values by up to 1.23 and inflates pool SD by 7.3%.
    """
    pool = make_pool(n=60)
    baseline = category_zscores(pool, pool_size=None)

    rookies = pd.DataFrame({col: [0.0] * 60 for col in
                            ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'TOV',
                             'FG_PCT', 'FGA', 'FT_PCT', 'FTA']})
    rookies['player'] = [f'Rookie {i}' for i in range(60)]
    rookies['flags'] = 'NO_PRIOR_SEASON'
    with_rookies = category_zscores(pd.concat([pool, rookies], ignore_index=True),
                                    pool_size=None)

    pd.testing.assert_frame_equal(baseline, with_rookies.iloc[:len(pool)],
                                  check_exact=False, rtol=1e-12)


def test_flagged_rows_get_nan_value_not_a_computed_number():
    """
    A rookie ranked last on fabricated zeros is a STRONGER claim than no
    projection at all, and a drafter reading it as real would pass on a
    lottery pick.
    """
    pool = make_pool(n=20)
    rookie = pool.iloc[[0]].copy()
    rookie['player'] = 'Rookie Ronnie'
    rookie['flags'] = 'NO_PRIOR_SEASON'
    for col in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'TOV', 'FG_PCT', 'FT_PCT']:
        rookie[col] = 0.0
    frame = build_value_frame(pd.concat([pool, rookie], ignore_index=True), pool_size=None)

    row = frame[frame['player'] == 'Rookie Ronnie'].iloc[0]
    assert np.isnan(row['value_total'])
    assert np.isnan(row['value_points_league'])
    assert frame[frame['player'] != 'Rookie Ronnie']['value_total'].notna().all()
    assert 'Rookie Ronnie' in set(frame['player']), "rookie stays in the OUTPUT"


# ---------------------------------------------------------------------------
# Points leagues
# ---------------------------------------------------------------------------

def test_points_league_matches_hand_computed_formula():
    pool = pd.DataFrame({
        'PTS': [20.0], 'REB': [10.0], 'AST': [5.0],
        'STL': [1.0], 'BLK': [2.0], 'TOV': [3.0], 'flags': [''],
    })
    expected = 20 + 1.2 * 10 + 1.5 * 5 + 3 * 1 + 3 * 2 - 1 * 3
    assert points_league_value(pool).iat[0] == pytest.approx(expected)


def test_points_league_accepts_a_custom_formula():
    pool = pd.DataFrame({'PTS': [10.0], 'REB': [10.0], 'AST': [10.0],
                         'STL': [0.0], 'BLK': [0.0], 'TOV': [0.0], 'flags': ['']})
    custom = dict(DEFAULT_POINTS_FORMULA, PTS=2.0)
    assert points_league_value(pool, custom).iat[0] == pytest.approx(
        2 * 10 + 1.2 * 10 + 1.5 * 10)


def test_points_league_missing_column_raises():
    with pytest.raises(ValueError, match='PTS'):
        points_league_value(pd.DataFrame({'REB': [1.0]}))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_pool_returns_nan_columns():
    empty = pd.DataFrame({c: [] for c in
                          ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'TOV',
                           'FG_PCT', 'FGA', 'FT_PCT', 'FTA', 'flags']})
    z = category_zscores(empty)
    assert len(z) == 0 or z.isna().all().all()


def test_all_rows_flagged_returns_nan_values():
    pool = make_pool(n=5)
    pool['flags'] = 'NO_PRIOR_SEASON'
    z = category_zscores(pool, pool_size=None)
    assert z.isna().all().all()


def test_identical_players_produce_zero_not_nan():
    """Zero variance in a category must not divide by zero."""
    pool = make_pool(n=10)
    pool['BLK'] = 1.0
    z = category_zscores(pool, pool_size=None)
    assert z['z_BLK'].notna().all()
    assert (z['z_BLK'] == 0).all()


# ---------------------------------------------------------------------------
# Games-weighted (season-total) value
# ---------------------------------------------------------------------------

from fantasy_value import to_season_totals  # noqa: E402


def test_equal_games_ranks_like_per_game_value():
    pool = make_pool().assign(GP=70.0)
    frame = build_value_frame(pool, pool_size=40, games_col='GP')
    assert (frame['value_total'].rank() == frame['value_total_gp'].rank()).all()


def test_more_games_means_more_total_value():
    pool = make_pool()
    twin = pool.iloc[[0]].copy().assign(player='Iron Man')
    pool = pd.concat([pool, twin], ignore_index=True).assign(GP=60.0)
    pool.loc[pool['player'] == 'Iron Man', 'GP'] = 80.0
    frame = build_value_frame(pool, pool_size=None, games_col='GP').set_index('player')
    assert frame.loc['Iron Man', 'value_total'] == pytest.approx(frame.loc['Player 0', 'value_total'])
    assert frame.loc['Iron Man', 'value_total_gp'] > frame.loc['Player 0', 'value_total_gp']


def test_percentages_stay_rates_and_scale_through_attempts():
    pool = make_pool(n=5).assign(GP=[10.0, 20.0, 30.0, 40.0, 50.0])
    totals = to_season_totals(pool, 'GP')
    assert np.allclose(totals['FG_PCT'], pool['FG_PCT'])
    assert np.allclose(totals['FGA'], pool['FGA'] * pool['GP'])
    assert np.allclose(totals['FTA'], pool['FTA'] * pool['GP'])


def test_percentage_impact_doubles_with_games_not_rate():
    pool = make_pool(n=30).assign(GP=50.0)
    pool.loc[0, 'GP'] = 100.0                     # clipped to 82
    totals = to_season_totals(pool, 'GP')
    assert totals.loc[0, 'FGA'] == pytest.approx(pool.loc[0, 'FGA'] * 82)
    assert totals.loc[0, 'FG_PCT'] == pool.loc[0, 'FG_PCT']


def test_negative_games_clip_to_zero():
    pool = make_pool(n=3).assign(GP=[-5.0, 10.0, 20.0])
    assert to_season_totals(pool, 'GP').loc[0, 'PTS'] == 0.0


def test_missing_games_gives_nan_not_zero():
    pool = make_pool().assign(GP=65.0)
    pool.loc[3, 'GP'] = np.nan
    frame = build_value_frame(pool, pool_size=None, games_col='GP')
    assert np.isnan(frame.loc[3, 'value_total_gp'])
    assert frame.drop(index=3)['value_total_gp'].notna().all()


def test_excluded_flags_are_nan_in_games_weighted_value():
    pool = make_pool().assign(GP=65.0)
    pool.loc[0, 'flags'] = 'NO_PRIOR_SEASON'
    frame = build_value_frame(pool, pool_size=None, games_col='GP')
    assert np.isnan(frame.loc[0, 'value_total_gp'])


def test_games_col_leaves_existing_columns_unchanged():
    pool = make_pool().assign(GP=np.linspace(20, 82, 60))
    before = build_value_frame(pool, pool_size=40)
    after = build_value_frame(pool, pool_size=40, games_col='GP')
    pd.testing.assert_frame_equal(before, after.drop(columns='value_total_gp'))
    assert 'value_total_gp' not in before.columns
