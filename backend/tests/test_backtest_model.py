"""
Tests for backend/backtest_model.py: backtesting the real NBAProjectionModel
ensemble against held-out historical games.
"""
import sys
import os
import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backtest_model import build_backtest_cases, run_model_backtest, summarize


def make_matchup_gamelog(home_team='CHO', away_team='HOU', n_games=8, game_id_prefix='20260101'):
    """
    Build a gamelog with two players (one per team) who play each other every
    game, with game_id encoding the home team as its last 3 chars (mirrors
    Basketball-Reference IDs, e.g. '202602190CHO').
    """
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


# Happy path

class TestBuildBacktestCases:
    def test_builds_one_case_per_qualifying_player(self):
        df = make_matchup_gamelog(n_games=8)
        cases = build_backtest_cases(df, min_prior_games=5)
        assert len(cases) == 2
        players = {c['player'] for c in cases}
        assert players == {'Home Player', 'Away Player'}

    def test_derives_is_home_from_game_id_suffix(self):
        df = make_matchup_gamelog(home_team='CHO', away_team='HOU', n_games=8)
        cases = build_backtest_cases(df, min_prior_games=5)
        home_case = next(c for c in cases if c['player'] == 'Home Player')
        away_case = next(c for c in cases if c['player'] == 'Away Player')
        assert home_case['is_home'] == 1
        assert away_case['is_home'] == 0

    def test_derives_opponent_from_shared_game_id(self):
        df = make_matchup_gamelog(home_team='CHO', away_team='HOU', n_games=8)
        cases = build_backtest_cases(df, min_prior_games=5)
        home_case = next(c for c in cases if c['player'] == 'Home Player')
        away_case = next(c for c in cases if c['player'] == 'Away Player')
        assert home_case['opponent'] == 'HOU'
        assert away_case['opponent'] == 'CHO'

    def test_holds_out_most_recent_game_only(self):
        df = make_matchup_gamelog(n_games=8)
        cases = build_backtest_cases(df, min_prior_games=5)
        home_case = next(c for c in cases if c['player'] == 'Home Player')
        # Last game had PTS = 20 + 7 = 27, so it must not appear in prior_games_df
        assert 27 not in home_case['prior_games_df']['PTS'].tolist()
        assert home_case['actual_pts'] == 27
        assert len(home_case['prior_games_df']) == 7

    def test_respects_max_players(self):
        df = make_matchup_gamelog(n_games=8)
        cases = build_backtest_cases(df, min_prior_games=5, max_players=1)
        assert len(cases) == 1


class TestRunModelBacktest:
    @patch('backtest_model.calculate_usage_rate', return_value=22.0)
    @patch('backtest_model.get_player_position', return_value='SG')
    @patch('backtest_model.get_opponent_defense_stats', return_value={
        'opp_pts_allowed': 24.0, 'opp_reb_allowed': 7.5, 'opp_ast_allowed': 5.5, 'opp_fd_allowed': 24.0,
    })
    def test_returns_prediction_per_case(self, mock_defense, mock_position, mock_usage):
        df = make_matchup_gamelog(n_games=8)
        results = run_model_backtest(df, min_prior_games=5)
        assert set(results.columns) == {'player', 'actual', 'baseline_pred', 'model_pred'}
        assert len(results) == 2

    @patch('backtest_model.calculate_usage_rate', return_value=22.0)
    @patch('backtest_model.get_player_position', return_value='SG')
    @patch('backtest_model.get_opponent_defense_stats', return_value={
        'opp_pts_allowed': 24.0, 'opp_reb_allowed': 7.5, 'opp_ast_allowed': 5.5, 'opp_fd_allowed': 24.0,
    })
    def test_baseline_pred_is_mean_of_prior_games(self, mock_defense, mock_position, mock_usage):
        df = make_matchup_gamelog(n_games=8)
        results = run_model_backtest(df, min_prior_games=5)
        home_row = results[results['player'] == 'Home Player'].iloc[0]
        # Prior games PTS = 20..26 -> mean = 23
        assert home_row['baseline_pred'] == pytest.approx(23.0)


class TestSummarize:
    def test_computes_mae_for_both_columns(self):
        df = pd.DataFrame({
            'player': ['A', 'B'],
            'actual': [20.0, 10.0],
            'baseline_pred': [22.0, 8.0],
            'model_pred': [19.0, 15.0],
        })
        summary = summarize(df)
        assert summary['n_predictions'] == 2
        assert summary['baseline_mae'] == pytest.approx(2.0)
        assert summary['model_mae'] == pytest.approx(3.0)


# Edge cases

class TestEdgeCases:
    def test_no_qualifying_players_returns_empty(self):
        df = make_matchup_gamelog(n_games=3)  # below default min_prior_games=5
        cases = build_backtest_cases(df, min_prior_games=5)
        assert cases == []

    def test_empty_results_summary(self):
        empty_df = pd.DataFrame(columns=['player', 'actual', 'baseline_pred', 'model_pred'])
        summary = summarize(empty_df)
        assert summary == {'n_predictions': 0, 'baseline_mae': None, 'model_mae': None}

    def test_unmatched_game_id_skipped(self):
        """A player whose games have no opponent row (e.g. bad/missing data) should be
        skipped, not crash, while properly paired players still produce cases."""
        paired_df = make_matchup_gamelog(n_games=8)

        dates = pd.date_range(start='2026-01-01', periods=8, freq='D')
        solo_rows = [{
            'PLAYER_NAME': 'Solo Player', 'team': 'SOL',
            'game_id': f"{date.strftime('%Y%m%d')}0SOL",  # no other team shares this game_id
            'GAME_DATE': date.strftime('%Y-%m-%d'), 'PTS': 15, 'REB': 4, 'AST': 3,
            'MIN': 25, 'FGA': 10, 'FG3A': 2, 'FTA': 3, 'STL': 1, 'BLK': 0, 'TOV': 1,
        } for date in dates]
        combined = pd.concat([paired_df, pd.DataFrame(solo_rows)], ignore_index=True)

        cases = build_backtest_cases(combined, min_prior_games=5)
        players_with_cases = {c['player'] for c in cases}
        assert 'Solo Player' not in players_with_cases
        assert 'Home Player' in players_with_cases
        assert 'Away Player' in players_with_cases


# Failure cases

class TestFailureCases:
    def test_missing_required_column_raises(self):
        df = pd.DataFrame({'PLAYER_NAME': ['A'], 'GAME_DATE': ['2026-01-01']})
        with pytest.raises(ValueError):
            build_backtest_cases(df)
