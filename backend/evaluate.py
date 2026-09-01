"""
Walk-forward evaluation of points predictions against a naive season-average baseline.

For each player, games are ordered chronologically. For every game after the
first, two predictions are made using only information available *before*
that game:
  - naive baseline: the player's average PTS over all prior games this season
  - rolling model:  the player's average PTS over the last N prior games
                     (mirrors the `pts_5g_avg` feature used by NBAProjectionModel)

Both predictions are compared against the actual PTS scored, and MAE is
reported for each so the rolling-average model can be judged against the
naive baseline it is meant to beat.
"""
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error

GAMELOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'cached_player_gamelogs.csv')


def compute_predictions(gamelog_df, target_col='PTS', player_col='PLAYER_NAME',
                         date_col='GAME_DATE', rolling_window=5):
    """
    Build walk-forward predictions for every game (per player) that has at
    least one prior game to derive predictions from.

    Returns a DataFrame with columns: player, actual, baseline_pred, model_pred
    """
    required_cols = {target_col, player_col, date_col}
    missing = required_cols - set(gamelog_df.columns)
    if missing:
        raise ValueError(f"gamelog_df is missing required columns: {missing}")

    records = []

    for player, player_df in gamelog_df.groupby(player_col):
        player_df = player_df.copy()
        player_df[date_col] = pd.to_datetime(player_df[date_col], errors='coerce')
        player_df = player_df.sort_values(date_col).reset_index(drop=True)
        actuals = player_df[target_col].astype(float).tolist()

        for i in range(1, len(actuals)):
            prior = actuals[:i]
            baseline_pred = sum(prior) / len(prior)
            recent = prior[-rolling_window:]
            model_pred = sum(recent) / len(recent)

            records.append({
                'player': player,
                'actual': actuals[i],
                'baseline_pred': baseline_pred,
                'model_pred': model_pred,
            })

    return pd.DataFrame(records, columns=['player', 'actual', 'baseline_pred', 'model_pred'])


def evaluate_points_mae(gamelog_df, rolling_window=5):
    """
    Compute MAE on points for the naive season-average baseline and the
    rolling-average model proxy.

    Returns a dict: {'n_predictions', 'baseline_mae', 'model_mae'}
    """
    predictions_df = compute_predictions(gamelog_df, target_col='PTS', rolling_window=rolling_window)

    if predictions_df.empty:
        return {'n_predictions': 0, 'baseline_mae': None, 'model_mae': None}

    baseline_mae = mean_absolute_error(predictions_df['actual'], predictions_df['baseline_pred'])
    model_mae = mean_absolute_error(predictions_df['actual'], predictions_df['model_pred'])

    return {
        'n_predictions': len(predictions_df),
        'baseline_mae': baseline_mae,
        'model_mae': model_mae,
    }


def run_evaluation(gamelog_path=GAMELOG_PATH, rolling_window=5):
    gamelog_df = pd.read_csv(gamelog_path)
    results = evaluate_points_mae(gamelog_df, rolling_window=rolling_window)

    print("=" * 60)
    print("POINTS PREDICTION EVALUATION")
    print("=" * 60)
    print(f"Predictions evaluated: {results['n_predictions']}")
    if results['n_predictions'] == 0:
        print("No predictions could be generated (not enough game history).")
        return results

    print(f"Naive season-average baseline MAE: {results['baseline_mae']:.3f}")
    print(f"Rolling {rolling_window}-game average MAE:      {results['model_mae']:.3f}")

    diff = results['baseline_mae'] - results['model_mae']
    if diff > 0:
        print(f"\nRolling average beats the naive baseline by {diff:.3f} MAE.")
    elif diff < 0:
        print(f"\nNaive baseline beats the rolling average by {-diff:.3f} MAE.")
    else:
        print("\nRolling average and naive baseline are tied.")

    return results


if __name__ == "__main__":
    run_evaluation()
