"""
Train NBAProjectionModel's ensemble architecture on REAL historical outcomes
instead of the synthetic, formula-derived targets used by
`NBAProjectionModel.create_synthetic_training_data`.

Why this exists: the production training path in model.py never learns from
real points scored — it generates synthetic targets via a hand-written
formula (`pts_5g_avg * usage_factor * opp_factor * home_boost * total_factor
* noise`) and fits models to reproduce that formula. A backtest of that path
(backend/backtest_model.py) confirmed it loses to a naive season-average
baseline (MAE 5.485 vs. 4.950 on points). This module builds a real,
walk-forward, leakage-free supervised dataset from cached_player_gamelogs.csv
and trains the same model architecture on actual outcomes.

Dataset construction (per player, chronological):
  - For every game after the player's first `min_prior_games` games, build
    features from ONLY strictly-prior games (rolling averages over the last
    `rolling_window` of them), and label = actual PTS scored in that game.
  - The player's single most recent game is held out as the test set (same
    protocol as backtest_model.py), so evaluation MAE is on truly unseen
    games. All earlier games, pooled across all players, form the training
    set.

Known limitation (disclosed, not hidden): usage_rate and opponent defense
stats are current-season snapshots, not point-in-time historical values, and
spread/total are held at a neutral default since historical odds aren't
scraped. Rolling scoring history, home/away, and opponent are real and
point-in-time.
"""
import os
import sys
import contextlib
import io
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import (
    NBAProjectionModel,
    calculate_rolling_features,
    calculate_usage_rate,
    get_player_position,
    get_opponent_defense_stats,
)

GAMELOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cached_player_gamelogs.csv')

NEUTRAL_SPREAD = 0
NEUTRAL_TOTAL = 225.0

FEATURE_COLUMNS = [
    'pts_5g_avg', 'reb_5g_avg', 'ast_5g_avg', 'mins_5g_avg', 'fga_5g_avg',
    'fg3a_5g_avg', 'fta_5g_avg', 'stl_5g_avg', 'blk_5g_avg', 'tov_5g_avg',
    'usage_rate', 'is_home', 'spread', 'total',
    'opp_pts_allowed', 'opp_reb_allowed', 'opp_ast_allowed', 'opp_fd_allowed',
]


def _build_opponent_map(gamelog_df):
    """Map game_id -> {team: opponent_team} for games with exactly 2 teams, plus home team per game_id."""
    opponent_map = {}
    home_team_map = {}
    for game_id, group in gamelog_df.groupby('game_id'):
        teams = group['team'].unique().tolist()
        if len(teams) != 2:
            continue
        opponent_map[game_id] = {teams[0]: teams[1], teams[1]: teams[0]}
        home_team_map[game_id] = str(game_id)[-3:]
    return opponent_map, home_team_map


def build_training_dataset(gamelog_df, min_prior_games=5, rolling_window=5, verbose=False):
    """
    Build a walk-forward, leakage-free dataset of (features, actual PTS) rows
    from real historical games, with an `is_test` flag marking each player's
    most recent game as the held-out evaluation row.

    Returns a DataFrame with FEATURE_COLUMNS + ['player', 'actual_pts', 'baseline_pred', 'is_test'].
    """
    required = {'PLAYER_NAME', 'GAME_DATE', 'game_id', 'team', 'PTS'}
    missing = required - set(gamelog_df.columns)
    if missing:
        raise ValueError(f"gamelog_df is missing required columns: {missing}")

    opponent_map, home_team_map = _build_opponent_map(gamelog_df)

    usage_cache = {}
    position_cache = {}
    defense_cache = {}

    def cached_usage(player):
        if player not in usage_cache:
            usage_cache[player] = calculate_usage_rate(player)
        return usage_cache[player]

    def cached_position(player):
        if player not in position_cache:
            position_cache[player] = get_player_position(player)
        return position_cache[player]

    def cached_defense(opponent, position):
        key = (opponent, position)
        if key not in defense_cache:
            defense_cache[key] = get_opponent_defense_stats(opponent, position)
        return defense_cache[key]

    rows = []

    stdout_ctx = contextlib.redirect_stdout(io.StringIO() if not verbose else sys.stdout)
    with stdout_ctx:
        for player, player_df in gamelog_df.groupby('PLAYER_NAME'):
            player_df = player_df.copy()
            player_df['GAME_DATE'] = pd.to_datetime(player_df['GAME_DATE'], errors='coerce')
            player_df = player_df.sort_values('GAME_DATE').reset_index(drop=True)

            if len(player_df) <= min_prior_games:
                continue

            position = cached_position(player)
            usage_rate = cached_usage(player)

            for i in range(min_prior_games, len(player_df)):
                current = player_df.iloc[i]
                prior_games_df = player_df.iloc[:i]

                game_id = current['game_id']
                team = str(current['team'])
                if game_id not in opponent_map or team not in opponent_map[game_id]:
                    continue
                opponent = opponent_map[game_id][team]
                is_home = 1 if team == home_team_map[game_id] else 0

                rolling_feats = calculate_rolling_features(prior_games_df, window=rolling_window)
                defense_stats = cached_defense(opponent, position)
                baseline_pred = prior_games_df['PTS'].astype(float).mean()

                row = {
                    'player': player,
                    'game_date': current['GAME_DATE'],
                    'pts_5g_avg': rolling_feats.get('pts_roll_avg', 0),
                    'reb_5g_avg': rolling_feats.get('reb_roll_avg', 0),
                    'ast_5g_avg': rolling_feats.get('ast_roll_avg', 0),
                    'mins_5g_avg': rolling_feats.get('min_roll_avg', 0),
                    'fga_5g_avg': rolling_feats.get('fga_roll_avg', 0),
                    'fg3a_5g_avg': rolling_feats.get('fg3a_roll_avg', 0),
                    'fta_5g_avg': rolling_feats.get('fta_roll_avg', 0),
                    'stl_5g_avg': rolling_feats.get('stl_roll_avg', 0),
                    'blk_5g_avg': rolling_feats.get('blk_roll_avg', 0),
                    'tov_5g_avg': rolling_feats.get('tov_roll_avg', 0),
                    'usage_rate': usage_rate,
                    'is_home': is_home,
                    'spread': NEUTRAL_SPREAD,
                    'total': NEUTRAL_TOTAL,
                    'opp_pts_allowed': defense_stats['opp_pts_allowed'],
                    'opp_reb_allowed': defense_stats['opp_reb_allowed'],
                    'opp_ast_allowed': defense_stats['opp_ast_allowed'],
                    'opp_fd_allowed': defense_stats['opp_fd_allowed'],
                    'actual_pts': float(current['PTS']),
                    'baseline_pred': baseline_pred,
                    'is_test': (i == len(player_df) - 1),
                }
                rows.append(row)

    return pd.DataFrame(rows)


def train_and_evaluate(gamelog_df, min_prior_games=5, rolling_window=5, verbose=False):
    """
    Build the real dataset, train NBAProjectionModel's ensemble on the
    pooled training rows (all games except each player's held-out last
    game), predict on the held-out test rows, and return an MAE summary
    alongside the per-row predictions.
    """
    dataset = build_training_dataset(
        gamelog_df, min_prior_games=min_prior_games, rolling_window=rolling_window, verbose=verbose
    )

    if dataset.empty:
        return {'n_train': 0, 'n_test': 0, 'baseline_mae': None, 'model_mae': None}, dataset

    train_df = dataset[~dataset['is_test']]
    test_df = dataset[dataset['is_test']]

    if train_df.empty or test_df.empty:
        return {'n_train': len(train_df), 'n_test': len(test_df), 'baseline_mae': None, 'model_mae': None}, dataset

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df['actual_pts']
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df['actual_pts']

    ml_system = NBAProjectionModel()
    ml_system.initialize_models()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    predictions = {}
    stdout_ctx = contextlib.redirect_stdout(io.StringIO() if not verbose else sys.stdout)
    with stdout_ctx:
        for name, mdl in ml_system.models.items():
            if name in ['linear', 'bayesian']:
                mdl.fit(X_train_scaled, y_train)
                predictions[name] = mdl.predict(X_test_scaled)
            else:
                mdl.fit(X_train, y_train)
                predictions[name] = mdl.predict(X_test)

    ensemble_pred = np.mean(list(predictions.values()), axis=0)

    results_df = test_df[['player', 'actual_pts', 'baseline_pred']].copy()
    results_df = results_df.rename(columns={'actual_pts': 'actual'})
    results_df['model_pred'] = ensemble_pred

    summary = {
        'n_train': len(train_df),
        'n_test': len(test_df),
        'baseline_mae': mean_absolute_error(results_df['actual'], results_df['baseline_pred']),
        'model_mae': mean_absolute_error(results_df['actual'], results_df['model_pred']),
    }

    return summary, results_df


def run_training_evaluation(gamelog_path=GAMELOG_PATH, min_prior_games=5, rolling_window=5):
    gamelog_df = pd.read_csv(gamelog_path)
    summary, results_df = train_and_evaluate(gamelog_df, min_prior_games=min_prior_games, rolling_window=rolling_window)

    print("=" * 60)
    print("REAL-DATA-TRAINED ENSEMBLE: POINTS")
    print("=" * 60)
    print(f"Training rows (pooled, real outcomes): {summary['n_train']}")
    print(f"Held-out test rows (most recent game per player): {summary['n_test']}")
    if not summary['n_test']:
        print("No test predictions could be generated.")
        return summary, results_df

    print(f"Naive season-average baseline MAE: {summary['baseline_mae']:.3f}")
    print(f"Real-data-trained ensemble MAE:     {summary['model_mae']:.3f}")

    diff = summary['baseline_mae'] - summary['model_mae']
    if diff > 0:
        print(f"\nEnsemble trained on real outcomes beats the naive baseline by {diff:.3f} MAE.")
    elif diff < 0:
        print(f"\nNaive baseline beats the ensemble by {-diff:.3f} MAE.")
    else:
        print("\nEnsemble and naive baseline are tied.")

    return summary, results_df


if __name__ == "__main__":
    run_training_evaluation()
