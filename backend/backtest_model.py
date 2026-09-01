"""
Backtest the actual NBAProjectionModel ensemble (not a heuristic proxy)
against real historical games, and compare its MAE on points to the naive
season-average baseline from evaluate.py.

For each player with enough game history, the most recent game is held out.
All prior games are used to build point-in-time rolling features (matching
`calculate_rolling_features`), and the real `is_home`/opponent are derived
from the held-out game's `game_id` (Basketball-Reference boxscore IDs end in
the home team's abbreviation). The full `NBAProjectionModel` pipeline
(`train_models_on_the_fly` + `ensemble_prediction`) is then run exactly as
it runs in production, and the ensemble prediction is compared to the
player's actual points in the held-out game.

Known limitation (disclosed, not hidden): `usage_rate` and opponent defense
stats are only available as current-season snapshots, not point-in-time
historical values, and per-game spread/total are not available historically
(only "today's" odds are scraped). Those three inputs are therefore held at
their current-season / neutral values for every backtest point, same as they
would be if you ran the live pipeline today. Everything else (rolling scoring
history, home/away, opponent) is real, point-in-time, and leakage-free.
"""
import os
import sys
import contextlib
import io
import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import (
    NBAProjectionModel,
    calculate_rolling_features,
    calculate_usage_rate,
    get_player_position,
    get_opponent_defense_stats,
)

GAMELOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'cached_player_gamelogs.csv')

NEUTRAL_SPREAD = 0
NEUTRAL_TOTAL = 225.0


def _home_team_from_game_id(game_id):
    """Basketball-Reference game IDs end with the home team's 3-letter abbreviation."""
    return str(game_id)[-3:]


def build_backtest_cases(gamelog_df, min_prior_games=5, max_players=None):
    """
    For each player with >= min_prior_games prior games, hold out their most
    recent game and return the prior games (for rolling features) plus the
    held-out game's actual PTS, opponent, and home/away flag.

    Returns a list of dicts: {player, prior_games_df, actual_pts, is_home, opponent}
    """
    required = {'PLAYER_NAME', 'GAME_DATE', 'game_id', 'team', 'PTS'}
    missing = required - set(gamelog_df.columns)
    if missing:
        raise ValueError(f"gamelog_df is missing required columns: {missing}")

    cases = []
    players = sorted(gamelog_df['PLAYER_NAME'].unique())
    if max_players:
        players = players[:max_players]

    for player in players:
        player_df = gamelog_df[gamelog_df['PLAYER_NAME'] == player].copy()
        player_df['GAME_DATE'] = pd.to_datetime(player_df['GAME_DATE'], errors='coerce')
        player_df = player_df.sort_values('GAME_DATE').reset_index(drop=True)

        if len(player_df) <= min_prior_games:
            continue

        held_out = player_df.iloc[-1]
        prior_games_df = player_df.iloc[:-1]

        home_team = _home_team_from_game_id(held_out['game_id'])
        player_team = str(held_out['team'])
        is_home = 1 if player_team == home_team else 0

        opponent_row = gamelog_df[
            (gamelog_df['game_id'] == held_out['game_id']) &
            (gamelog_df['team'] != player_team)
        ]
        if opponent_row.empty:
            continue
        opponent = opponent_row.iloc[0]['team']

        cases.append({
            'player': player,
            'prior_games_df': prior_games_df,
            'actual_pts': float(held_out['PTS']),
            'is_home': is_home,
            'opponent': opponent,
        })

    return cases


def run_model_backtest(gamelog_df, min_prior_games=5, max_players=None, verbose=False):
    """
    Run the real NBAProjectionModel ensemble against held-out games and
    return per-case predictions alongside the naive season-average baseline.
    """
    cases = build_backtest_cases(gamelog_df, min_prior_games=min_prior_games, max_players=max_players)
    results = []

    for case in cases:
        with contextlib.redirect_stdout(io.StringIO() if not verbose else sys.stdout):
            prior_games_df = case['prior_games_df']
            player_features = calculate_rolling_features(prior_games_df, window=5)

            baseline_pred = prior_games_df['PTS'].astype(float).mean()

            game_context = {
                'opponent': case['opponent'],
                'is_home': case['is_home'],
                'spread': NEUTRAL_SPREAD,
                'total': NEUTRAL_TOTAL,
            }

            position = get_player_position(case['player'])
            defense_stats = get_opponent_defense_stats(case['opponent'], position)
            real_usage_rate = calculate_usage_rate(case['player'])
            player_features['usage_rate'] = real_usage_rate

            ml_system = NBAProjectionModel()
            ml_system.initialize_models()

            trained_models, feature_columns = ml_system.train_models_on_the_fly(
                player_features, game_context, defense_stats, case['player'], 'Points'
            )

            final_feature_dict = {
                'pts_5g_avg': player_features.get('pts_roll_avg', 0),
                'reb_5g_avg': player_features.get('reb_roll_avg', 0),
                'ast_5g_avg': player_features.get('ast_roll_avg', 0),
                'mins_5g_avg': player_features.get('min_roll_avg', 0),
                'fga_5g_avg': player_features.get('fga_roll_avg', 0),
                'fg3a_5g_avg': player_features.get('fg3a_roll_avg', 0),
                'fta_5g_avg': player_features.get('fta_roll_avg', 0),
                'stl_5g_avg': player_features.get('stl_roll_avg', 0),
                'blk_5g_avg': player_features.get('blk_roll_avg', 0),
                'tov_5g_avg': player_features.get('tov_roll_avg', 0),
                'usage_rate': real_usage_rate,
                'is_home': game_context['is_home'],
                'spread': game_context['spread'],
                'total': game_context['total'],
                'opp_pts_allowed': defense_stats['opp_pts_allowed'],
                'opp_reb_allowed': defense_stats['opp_reb_allowed'],
                'opp_ast_allowed': defense_stats['opp_ast_allowed'],
                'opp_fd_allowed': defense_stats['opp_fd_allowed'],
            }
            prediction_input = pd.DataFrame([final_feature_dict])

            try:
                ensemble_pred, confidence, _ = ml_system.ensemble_prediction(
                    prediction_input, trained_models, feature_columns, 'Points'
                )
            except ValueError:
                continue

        results.append({
            'player': case['player'],
            'actual': case['actual_pts'],
            'baseline_pred': baseline_pred,
            'model_pred': ensemble_pred,
        })

    return pd.DataFrame(results, columns=['player', 'actual', 'baseline_pred', 'model_pred'])


def summarize(results_df):
    if results_df.empty:
        return {'n_predictions': 0, 'baseline_mae': None, 'model_mae': None}
    return {
        'n_predictions': len(results_df),
        'baseline_mae': mean_absolute_error(results_df['actual'], results_df['baseline_pred']),
        'model_mae': mean_absolute_error(results_df['actual'], results_df['model_pred']),
    }


def run_backtest(gamelog_path=GAMELOG_PATH, min_prior_games=5, max_players=None):
    gamelog_df = pd.read_csv(gamelog_path)
    results_df = run_model_backtest(gamelog_df, min_prior_games=min_prior_games, max_players=max_players)
    summary = summarize(results_df)

    print("=" * 60)
    print("REAL ENSEMBLE MODEL BACKTEST: POINTS")
    print("=" * 60)
    print(f"Held-out predictions evaluated: {summary['n_predictions']}")
    if summary['n_predictions'] == 0:
        print("No predictions could be generated.")
        return summary, results_df

    print(f"Naive season-average baseline MAE: {summary['baseline_mae']:.3f}")
    print(f"NBAProjectionModel ensemble MAE:    {summary['model_mae']:.3f}")

    diff = summary['baseline_mae'] - summary['model_mae']
    if diff > 0:
        print(f"\nEnsemble model beats the naive baseline by {diff:.3f} MAE.")
    elif diff < 0:
        print(f"\nNaive baseline beats the ensemble model by {-diff:.3f} MAE.")
    else:
        print("\nEnsemble model and naive baseline are tied.")

    return summary, results_df


if __name__ == "__main__":
    run_backtest()
