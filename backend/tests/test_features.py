"""
Tests for backend/features.py: point-in-time feature engineering.

The central property under test is no-leakage: every feature for a game must
be computable from strictly-earlier games only.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from features import (
    build_feature_dataset,
    nba_season,
    prepare_gamelog,
    FEATURE_COLUMNS,
)


def make_matchup_gamelog(home_team='CHO', away_team='HOU', n_games=10, home_pts=None):
    """
    Two players facing each other every game. game_id encodes the home team as
    its last 3 characters, mirroring Basketball-Reference IDs ('202602190CHO').
    """
    dates = pd.date_range(start='2026-01-01', periods=n_games, freq='D')
    home_pts = home_pts if home_pts is not None else [20 + i for i in range(n_games)]
    rows = []
    for i, date in enumerate(dates):
        game_id = f"{date.strftime('%Y%m%d')}0{home_team}"
        rows.append({
            'PLAYER_NAME': 'Home Player', 'team': home_team, 'game_id': game_id,
            'GAME_DATE': date.strftime('%Y-%m-%d'), 'PTS': home_pts[i], 'REB': 5, 'AST': 5,
            'MIN': 30, 'FGA': 15, 'FG3A': 4, 'FTA': 5, 'STL': 1, 'BLK': 1, 'TOV': 2,
        })
        rows.append({
            'PLAYER_NAME': 'Away Player', 'team': away_team, 'game_id': game_id,
            'GAME_DATE': date.strftime('%Y-%m-%d'), 'PTS': 10 + i, 'REB': 4, 'AST': 3,
            'MIN': 25, 'FGA': 10, 'FG3A': 3, 'FTA': 4, 'STL': 1, 'BLK': 0, 'TOV': 2,
        })
    return pd.DataFrame(rows)


POSITIONS = {'Home Player': 'SF', 'Away Player': 'PG'}


# Gamelog preparation

class TestPrepareGamelog:
    def test_derives_is_home_from_game_id(self):
        df = prepare_gamelog(make_matchup_gamelog(), positions=POSITIONS)
        assert set(df[df['PLAYER_NAME'] == 'Home Player']['is_home']) == {1}
        assert set(df[df['PLAYER_NAME'] == 'Away Player']['is_home']) == {0}

    def test_derives_opponent(self):
        df = prepare_gamelog(make_matchup_gamelog(home_team='CHO', away_team='HOU'), positions=POSITIONS)
        assert set(df[df['PLAYER_NAME'] == 'Home Player']['opponent']) == {'HOU'}
        assert set(df[df['PLAYER_NAME'] == 'Away Player']['opponent']) == {'CHO'}

    def test_attaches_positions_and_defaults_unknown(self):
        df = prepare_gamelog(make_matchup_gamelog(), positions={'Home Player': 'SF'})
        assert set(df[df['PLAYER_NAME'] == 'Home Player']['position']) == {'SF'}
        assert set(df[df['PLAYER_NAME'] == 'Away Player']['position']) == {'UNK'}

    def test_drops_games_without_exactly_two_teams(self):
        df = make_matchup_gamelog(n_games=4)
        orphan = df.iloc[[0]].copy()
        orphan['game_id'] = 'ORPHAN_GAME'
        combined = pd.concat([df, orphan], ignore_index=True)
        prepared = prepare_gamelog(combined, positions=POSITIONS)
        assert 'ORPHAN_GAME' not in set(prepared['game_id'])

    def test_missing_required_column_raises(self):
        with pytest.raises(ValueError):
            prepare_gamelog(pd.DataFrame({'PLAYER_NAME': ['A'], 'GAME_DATE': ['2026-01-01']}))


# The core no-leakage guarantee

class TestNoLeakage:
    def test_pts_season_excludes_current_game(self):
        """pts_season must be the mean of strictly-prior games (the naive baseline)."""
        df = make_matchup_gamelog(n_games=10)
        ds = build_feature_dataset(df, positions=POSITIONS, min_prior_games=5)
        home = ds[ds['player'] == 'Home Player'].sort_values('game_date').reset_index(drop=True)
        # Home Player scores 20..29; first kept row is game index 5 (PTS=25),
        # so pts_season must be mean(20..24) = 22, not including 25.
        assert home.iloc[0]['actual_pts'] == 25
        assert home.iloc[0]['pts_season'] == pytest.approx(22.0)

    def test_pts_r3_uses_only_three_prior_games(self):
        df = make_matchup_gamelog(n_games=10)
        ds = build_feature_dataset(df, positions=POSITIONS, min_prior_games=5)
        home = ds[ds['player'] == 'Home Player'].sort_values('game_date').reset_index(drop=True)
        # Row for game index 5 (PTS=25): prior 3 games are 22,23,24 -> mean 23
        assert home.iloc[0]['pts_r3'] == pytest.approx(23.0)

    def test_changing_future_points_does_not_change_earlier_features(self):
        """The strongest leakage check: perturb a late game, early rows must be identical."""
        base = make_matchup_gamelog(n_games=10)
        perturbed = base.copy()
        mask = (perturbed['PLAYER_NAME'] == 'Home Player')
        last_idx = perturbed[mask].index[-1]
        perturbed.loc[last_idx, 'PTS'] = 999  # absurd future value

        ds_base = build_feature_dataset(base, positions=POSITIONS, min_prior_games=5)
        ds_pert = build_feature_dataset(perturbed, positions=POSITIONS, min_prior_games=5)

        home_base = ds_base[ds_base['player'] == 'Home Player'].sort_values('game_date')
        home_pert = ds_pert[ds_pert['player'] == 'Home Player'].sort_values('game_date')

        # All rows except the final one must be bit-identical across FEATURE_COLUMNS
        pd.testing.assert_frame_equal(
            home_base.iloc[:-1][FEATURE_COLUMNS].reset_index(drop=True),
            home_pert.iloc[:-1][FEATURE_COLUMNS].reset_index(drop=True),
        )

    def test_baseline_pred_equals_pts_season(self):
        df = make_matchup_gamelog(n_games=10)
        ds = build_feature_dataset(df, positions=POSITIONS, min_prior_games=5)
        assert (ds['baseline_pred'] == ds['pts_season']).all()


# Feature construction

class TestBuildFeatureDataset:
    def test_all_declared_feature_columns_present(self):
        ds = build_feature_dataset(make_matchup_gamelog(), positions=POSITIONS, min_prior_games=5)
        for col in FEATURE_COLUMNS:
            assert col in ds.columns, f"missing feature column: {col}"

    def test_respects_min_prior_games(self):
        df = make_matchup_gamelog(n_games=10)
        ds = build_feature_dataset(df, positions=POSITIONS, min_prior_games=5)
        # Games at index 5..9 qualify -> 5 rows per player, 2 players
        assert len(ds) == 10

    def test_rest_days_and_b2b(self):
        """Daily games -> 1 rest day -> flagged as back-to-back."""
        ds = build_feature_dataset(make_matchup_gamelog(n_games=10), positions=POSITIONS, min_prior_games=5)
        assert set(ds['rest_days']) == {1.0}
        assert set(ds['is_b2b']) == {1}

    def test_games_played_counts_prior_games(self):
        df = make_matchup_gamelog(n_games=10)
        ds = build_feature_dataset(df, positions=POSITIONS, min_prior_games=5)
        home = ds[ds['player'] == 'Home Player'].sort_values('game_date').reset_index(drop=True)
        assert home['games_played'].tolist() == [5, 6, 7, 8, 9]

    def test_fga_share_is_fraction_of_team_attempts(self):
        """Only one player per team here, so that player takes 100% of team FGA."""
        ds = build_feature_dataset(make_matchup_gamelog(n_games=10), positions=POSITIONS, min_prior_games=5)
        assert ds['fga_share_r5'].round(6).eq(1.0).all()

    def test_no_nan_or_inf_in_features(self):
        ds = build_feature_dataset(make_matchup_gamelog(n_games=10), positions=POSITIONS, min_prior_games=5)
        block = ds[FEATURE_COLUMNS].to_numpy(dtype=float)
        assert not np.isnan(block).any()
        assert not np.isinf(block).any()

    def test_volatility_zero_for_constant_scorer(self):
        constant = [20] * 10
        ds = build_feature_dataset(
            make_matchup_gamelog(n_games=10, home_pts=constant), positions=POSITIONS, min_prior_games=5
        )
        home = ds[ds['player'] == 'Home Player']
        assert home['pts_std_r5'].abs().max() == pytest.approx(0.0, abs=1e-9)


# Season boundaries

class TestSeasonHandling:
    def test_nba_season_end_year(self):
        """October onward belongs to the next calendar year's season."""
        dates = pd.to_datetime(['2025-10-22', '2025-12-25', '2026-03-10', '2026-06-01'])
        assert nba_season(pd.Series(dates)).tolist() == [2026, 2026, 2026, 2026]

    def test_september_belongs_to_prior_season_numbering(self):
        assert nba_season(pd.Series(pd.to_datetime(['2025-09-30']))).tolist() == [2025]

    def _two_season_gamelog(self):
        """Same player, 8 games in each of two seasons; scoring level differs sharply."""
        rows = []
        for season_start, pts in (('2024-11-01', 40), ('2025-11-01', 10)):
            dates = pd.date_range(start=season_start, periods=8, freq='D')
            for date in dates:
                game_id = f"{date.strftime('%Y%m%d')}0CHO"
                rows.append({
                    'PLAYER_NAME': 'Home Player', 'team': 'CHO', 'game_id': game_id,
                    'GAME_DATE': date.strftime('%Y-%m-%d'), 'PTS': pts, 'REB': 5, 'AST': 5,
                    'MIN': 30, 'FGA': 15, 'FG3A': 4, 'FTA': 5, 'STL': 1, 'BLK': 1, 'TOV': 2,
                })
                rows.append({
                    'PLAYER_NAME': 'Away Player', 'team': 'HOU', 'game_id': game_id,
                    'GAME_DATE': date.strftime('%Y-%m-%d'), 'PTS': 12, 'REB': 4, 'AST': 3,
                    'MIN': 25, 'FGA': 10, 'FG3A': 3, 'FTA': 4, 'STL': 1, 'BLK': 0, 'TOV': 2,
                })
        return pd.DataFrame(rows)

    def test_season_average_resets_across_seasons(self):
        """
        The prior season's 40-point games must not leak into the new season's
        baseline -- pts_season is the naive baseline we benchmark against.
        """
        ds = build_feature_dataset(self._two_season_gamelog(), positions=POSITIONS, min_prior_games=5)
        newer = ds[(ds['player'] == 'Home Player') & (ds['season'] == 2026)]
        assert not newer.empty
        # Every 2026 row must reflect only the 10-point season, never a 40/10 blend
        assert newer['pts_season'].max() == pytest.approx(10.0)
        assert newer['pts_r5'].max() == pytest.approx(10.0)

    def test_games_played_resets_each_season(self):
        ds = build_feature_dataset(self._two_season_gamelog(), positions=POSITIONS, min_prior_games=5)
        home = ds[ds['player'] == 'Home Player']
        for season in (2025, 2026):
            season_rows = home[home['season'] == season].sort_values('game_date')
            assert season_rows['games_played'].tolist() == [5, 6, 7]

    def test_both_seasons_produce_rows(self):
        ds = build_feature_dataset(self._two_season_gamelog(), positions=POSITIONS, min_prior_games=5)
        assert set(ds['season']) == {2025, 2026}


# Edge cases

class TestEdgeCases:
    def test_empty_gamelog_returns_empty_frame(self):
        empty = pd.DataFrame(columns=['PLAYER_NAME', 'GAME_DATE', 'game_id', 'team', 'PTS'])
        ds = build_feature_dataset(empty, positions={}, min_prior_games=5)
        assert ds.empty

    def test_too_few_games_yields_no_rows(self):
        ds = build_feature_dataset(make_matchup_gamelog(n_games=3), positions=POSITIONS, min_prior_games=5)
        assert ds.empty

    def test_zero_minutes_does_not_produce_inf(self):
        """A DNP-style row (0 minutes, 0 attempts) must not create inf ratios."""
        df = make_matchup_gamelog(n_games=10)
        df.loc[df['PLAYER_NAME'] == 'Home Player', 'MIN'] = 0
        df.loc[df['PLAYER_NAME'] == 'Home Player', 'FGA'] = 0
        ds = build_feature_dataset(df, positions=POSITIONS, min_prior_games=5)
        home = ds[ds['player'] == 'Home Player']
        assert np.isfinite(home[FEATURE_COLUMNS].to_numpy(dtype=float)).all()
        assert home['pts_per_min_r5'].eq(0.0).all()
