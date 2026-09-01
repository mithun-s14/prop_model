"""
Train and evaluate the points model on engineered point-in-time features.

Fourth iteration in the evaluation chain:
  1. evaluate.py         -- naive baseline vs. a rolling-average heuristic
  2. backtest_model.py   -- production synthetic-trained ensemble (lost to baseline)
  3. train_real_model.py -- same architecture, trained on real outcomes (tied baseline)
  4. this module         -- real outcomes + engineered point-in-time features

Method
------
Features come from features.py and are leakage-free by construction. Feature
blocks and the ridge penalty are chosen by greedy forward selection on a
VALIDATION split carved out of the training data (`select_feature_blocks`),
so the test set is never used for tuning. `SELECTED_BLOCKS` / `SELECTED_ALPHA`
record what that procedure chose on the current data.

Ridge is the estimator rather than NBAProjectionModel's tree ensemble because
the tree models overfit badly at this sample size (~2.6k rows): on validation
they scored 5.19-5.25 MAE against a 5.11 baseline, while unregularized
LinearRegression blew up to 7.02 through collinearity. `evaluate_split` can
still run the original ensemble for comparison via `use_ensemble=True`.

Evaluation protocols
--------------------
  - `temporal` : train on games before a cutoff date, test on games after it.
                 No training row post-dates any test row; the honest estimate
                 of live performance.
  - `holdout`  : each player's most recent game is the test row. Matches the
                 protocol in backtest_model.py / train_real_model.py so MAE is
                 directly comparable to those runs.

Every result is reported against the naive season-average baseline on exactly
the same rows, with a paired t-test and a 95% confidence interval, because
the effects at this sample size are small enough that a bare MAE delta is
easy to over-read.
"""
import os
import sys
import contextlib
import io
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import build_feature_dataset, nba_season, FEATURE_COLUMNS
from model import NBAProjectionModel, get_player_position

GAMELOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'cached_player_gamelogs.csv')

# Thematic groups of features, the unit of greedy selection. Selecting whole
# blocks rather than individual columns keeps the search small enough that it
# does not simply overfit the validation split.
FEATURE_BLOCKS = {
    'PTS': ['pts_r3', 'pts_r5', 'pts_r10', 'pts_season'],
    'MIN': ['min_r3', 'min_r5', 'min_season', 'min_trend'],
    'VOL': ['fga_r5', 'fta_r5', 'fga_share_r5', 'pts_std_r5'],
    'SHOT': ['fg3a_r5', 'fg3a_rate_r5', 'fta_rate_r5'],
    'EFF': ['pts_per_min_r5', 'pts_per_fga_r5', 'fga_per_min_r5'],
    'CTX': ['is_home', 'rest_days', 'is_b2b', 'games_played'],
    'BOX': ['reb_r5', 'ast_r5', 'tov_r5', 'stl_r5', 'blk_r5'],
    'DEF': ['opp_pts_allowed_pit', 'opp_pts_allowed_pos_pit'],
    'MOM': ['momentum_3_vs_season'],
}

ALPHA_GRID = [3, 10, 30, 100, 300, 1000]

# Chosen by select_feature_blocks() on a validation split of the current data.
# Notably DEF (opponent defense) was NOT selected -- it made validation MAE worse.
SELECTED_BLOCKS = ['PTS', 'VOL', 'SHOT']
SELECTED_ALPHA = 30

_SCALED_MODELS = {'linear', 'bayesian'}


def load_positions(player_names, verbose=False):
    """Resolve each player's position once, silencing model.py's chatty lookups."""
    with contextlib.redirect_stdout(io.StringIO() if not verbose else sys.stdout):
        return {name: get_player_position(name) for name in player_names}


def columns_for_blocks(blocks):
    """Flatten a list of block names into their feature columns."""
    return [col for block in blocks for col in FEATURE_BLOCKS[block]]


def split_holdout(dataset):
    """Test set = each player's most recent game (matches earlier backtests)."""
    ordered = dataset.sort_values(['player', 'game_date'])
    is_last = ordered.groupby('player').cumcount(ascending=False) == 0
    return ordered[~is_last], ordered[is_last]


def split_temporal(dataset, test_fraction=0.2):
    """Test set = all games on/after the cutoff date; train = everything before."""
    dates = np.sort(dataset['game_date'].unique())
    if len(dates) < 2:
        return dataset.iloc[0:0], dataset.iloc[0:0]
    cutoff = dates[int(len(dates) * (1 - test_fraction))]
    return dataset[dataset['game_date'] < cutoff], dataset[dataset['game_date'] >= cutoff]


def split_by_season(dataset, test_seasons=None):
    """
    Test set = the most recent season; train = all earlier seasons.

    The most realistic protocol once the logs span multiple seasons: it mimics
    deploying a model trained on completed seasons against a new one. Returns
    empty frames when only a single season is present.
    """
    if 'season' not in dataset.columns:
        return dataset.iloc[0:0], dataset.iloc[0:0]
    seasons = np.sort(dataset['season'].unique())
    if len(seasons) < 2:
        return dataset.iloc[0:0], dataset.iloc[0:0]
    test_seasons = test_seasons or [seasons[-1]]
    is_test = dataset['season'].isin(test_seasons)
    return dataset[~is_test], dataset[is_test]


def fit_ridge(train_df, test_df, columns, alpha):
    """Standardize, fit ridge on the training rows, predict the test rows."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[columns])
    X_test = scaler.transform(test_df[columns])
    model = Ridge(alpha=alpha).fit(X_train, train_df['actual_pts'])
    return model.predict(X_test)


def fit_ensemble(train_df, test_df, columns, verbose=False):
    """Fit NBAProjectionModel's full architecture, for comparison against ridge."""
    ml_system = NBAProjectionModel()
    ml_system.initialize_models()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[columns])
    X_test_scaled = scaler.transform(test_df[columns])
    y_train = train_df['actual_pts']

    per_model = {}
    with contextlib.redirect_stdout(io.StringIO() if not verbose else sys.stdout):
        for name, mdl in ml_system.models.items():
            if name in _SCALED_MODELS:
                mdl.fit(X_train_scaled, y_train)
                per_model[name] = mdl.predict(X_test_scaled)
            else:
                mdl.fit(train_df[columns], y_train)
                per_model[name] = mdl.predict(test_df[columns])

    return np.mean(list(per_model.values()), axis=0), per_model


def select_feature_blocks(train_df, val_df, seed_blocks=('PTS',), alpha_grid=None):
    """
    Greedy forward selection over FEATURE_BLOCKS, scored on a validation split.

    Starts from `seed_blocks` and repeatedly adds whichever remaining block
    most improves validation MAE, stopping when no block helps. Returns
    (chosen_blocks, chosen_alpha, val_mae).
    """
    alpha_grid = alpha_grid or ALPHA_GRID
    chosen = list(seed_blocks)

    def best_for(blocks):
        cols = columns_for_blocks(blocks)
        scored = [
            (mean_absolute_error(val_df['actual_pts'], fit_ridge(train_df, val_df, cols, a)), a)
            for a in alpha_grid
        ]
        return min(scored)

    current_mae, current_alpha = best_for(chosen)

    while True:
        candidates = []
        for block in FEATURE_BLOCKS:
            if block in chosen:
                continue
            mae, alpha = best_for(chosen + [block])
            candidates.append((mae, block, alpha))
        if not candidates:
            break
        candidates.sort(key=lambda c: c[0])
        best_mae, best_block, best_alpha = candidates[0]
        if best_mae >= current_mae - 1e-6:
            break
        chosen.append(best_block)
        current_mae, current_alpha = best_mae, best_alpha

    return chosen, current_alpha, current_mae


def compare_to_baseline(actual, model_pred, baseline_pred):
    """
    Paired comparison of model vs. naive baseline absolute errors.

    Returns MAEs, the mean improvement, a paired t-test p-value and a 95% CI.
    Paired stats matter here: the two predictors are highly correlated, so an
    unpaired view would badly overstate the uncertainty of their difference.
    """
    actual = np.asarray(actual, dtype=float)
    abs_err_model = np.abs(actual - np.asarray(model_pred, dtype=float))
    abs_err_baseline = np.abs(actual - np.asarray(baseline_pred, dtype=float))
    diff = abs_err_baseline - abs_err_model  # positive => model is better

    n = len(diff)
    stderr = diff.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
    p_value = stats.ttest_rel(abs_err_baseline, abs_err_model).pvalue if n > 1 else float('nan')

    return {
        'n_test': n,
        'baseline_mae': abs_err_baseline.mean(),
        'model_mae': abs_err_model.mean(),
        'improvement': diff.mean(),
        'improvement_pct': 100 * diff.mean() / abs_err_baseline.mean() if abs_err_baseline.mean() else 0.0,
        'p_value': p_value,
        'ci_low': diff.mean() - 1.96 * stderr,
        'ci_high': diff.mean() + 1.96 * stderr,
        'win_rate': float((diff > 0).mean()),
    }


def evaluate_split(train_df, test_df, blocks=None, alpha=SELECTED_ALPHA,
                   use_ensemble=False, verbose=False):
    """Fit on train_df, predict test_df, and compare against the naive baseline."""
    blocks = blocks or SELECTED_BLOCKS
    columns = columns_for_blocks(blocks)

    if train_df.empty or test_df.empty:
        return {'n_train': len(train_df), 'n_test': len(test_df),
                'baseline_mae': None, 'model_mae': None}, pd.DataFrame()

    if use_ensemble:
        predictions, _ = fit_ensemble(train_df, test_df, columns, verbose=verbose)
    else:
        predictions = fit_ridge(train_df, test_df, columns, alpha)

    summary = compare_to_baseline(test_df['actual_pts'], predictions, test_df['baseline_pred'])
    summary['n_train'] = len(train_df)

    results_df = pd.DataFrame({
        'player': test_df['player'].values,
        'game_date': test_df['game_date'].values,
        'actual': test_df['actual_pts'].values,
        'baseline_pred': test_df['baseline_pred'].values,
        'model_pred': predictions,
    })
    return summary, results_df


def noise_floor_analysis(gamelog_df, test_df):
    """
    How much headroom actually exists above the naive baseline?

    Compares the baseline against two *leaky* oracles that are impossible in
    practice (they use the future) to bound what any honest model could reach:
      - full-season mean: knows each player's average over the whole dataset
      - test-period mean: knows each player's average over the test games
    """
    df = gamelog_df.copy()
    df['season'] = nba_season(pd.to_datetime(df['GAME_DATE'], format='mixed', errors='coerce'))

    # Group by (player, season), not player: averaging a player across several
    # seasons is not an oracle at all -- their form two years ago says little
    # about this season, so a career-wide mean can score WORSE than the naive
    # baseline and make headroom look negative.
    by_player_season = df.groupby(['PLAYER_NAME', 'season'])['PTS']
    within_player_std = by_player_season.std()[by_player_season.size() >= 6]
    full_season_mean = by_player_season.mean()

    if 'season' in test_df.columns:
        keys = pd.MultiIndex.from_arrays([test_df['player'], test_df['season']])
        oracle_full = pd.Series(full_season_mean.reindex(keys).to_numpy(), index=test_df.index)
        oracle_test = test_df.groupby(['player', 'season'])['actual_pts'].transform('mean')
    else:  # single-season logs: player alone is already the right key
        oracle_full = test_df['player'].map(full_season_mean.droplevel('season'))
        oracle_test = test_df.groupby('player')['actual_pts'].transform('mean')
    oracle_full = oracle_full.fillna(full_season_mean.mean())

    return {
        'mean_within_player_std': within_player_std.mean(),
        # E|X-mu| = sigma*sqrt(2/pi) for a Gaussian: the MAE of a perfect mean-predictor
        'gaussian_mae_floor': 0.7979 * within_player_std.mean(),
        'baseline_mae': mean_absolute_error(test_df['actual_pts'], test_df['baseline_pred']),
        'oracle_full_season_mae': mean_absolute_error(test_df['actual_pts'], oracle_full),
        'oracle_test_period_mae': mean_absolute_error(test_df['actual_pts'], oracle_test),
    }


def _print_summary(label, summary):
    print(f"\n--- {label} split ---")
    print(f"  train rows: {summary.get('n_train')}   test rows: {summary.get('n_test')}")
    if summary.get('baseline_mae') is None:
        print("  Not enough data to evaluate this split.")
        return
    print(f"  naive season-average baseline MAE : {summary['baseline_mae']:.3f}")
    print(f"  engineered-feature model MAE      : {summary['model_mae']:.3f}")
    print(f"  improvement                       : {summary['improvement']:+.3f} "
          f"({summary['improvement_pct']:+.2f}%)")
    print(f"  paired t-test p                   : {summary['p_value']:.4f}")
    print(f"  95% CI of improvement             : "
          f"[{summary['ci_low']:+.3f}, {summary['ci_high']:+.3f}]")
    print(f"  model better on                   : {100 * summary['win_rate']:.1f}% of rows")
    significant = summary['p_value'] < 0.05
    print(f"  => {'STATISTICALLY SIGNIFICANT' if significant else 'NOT statistically significant'} "
          f"at p<0.05")


def run(gamelog_path=GAMELOG_PATH, min_prior_games=5, reselect=True, verbose=False):
    gamelog_df = pd.read_csv(gamelog_path)
    positions = load_positions(gamelog_df['PLAYER_NAME'].astype(str).str.strip().unique(), verbose=verbose)
    dataset = build_feature_dataset(gamelog_df, positions=positions, min_prior_games=min_prior_games)

    print("=" * 66)
    print("ENGINEERED POINT-IN-TIME FEATURES: POINTS PREDICTION")
    print("=" * 66)
    print(f"dataset rows: {len(dataset)}   candidate features: {len(FEATURE_COLUMNS)}   "
          f"min_prior_games: {min_prior_games}")

    train_full, test_df = split_temporal(dataset, test_fraction=0.2)
    blocks, alpha = SELECTED_BLOCKS, SELECTED_ALPHA

    if reselect:
        # Select on a validation split carved from training data only.
        inner_train, inner_val = split_temporal(train_full, test_fraction=0.25)
        blocks, alpha, val_mae = select_feature_blocks(inner_train, inner_val)
        val_baseline = mean_absolute_error(inner_val['actual_pts'], inner_val['baseline_pred'])
        print(f"\nfeature selection (on validation, test never touched):")
        print(f"  validation rows: {len(inner_val)}   baseline MAE {val_baseline:.3f}")
        print(f"  chosen blocks: {blocks}   alpha: {alpha}   val MAE {val_mae:.3f}")
        print(f"  columns used: {len(columns_for_blocks(blocks))} of {len(FEATURE_COLUMNS)}")

    protocols = [
        ('temporal', (train_full, test_df)),
        ('holdout', split_holdout(dataset)),
    ]
    # Only meaningful once the logs span more than one season.
    season_train, season_test = split_by_season(dataset)
    if not season_test.empty:
        protocols.append(('by-season', (season_train, season_test)))

    summaries = {}
    for label, (tr, te) in protocols:
        summary, _ = evaluate_split(tr, te, blocks=blocks, alpha=alpha, verbose=verbose)
        summaries[label] = summary
        _print_summary(label, summary)

    floor = noise_floor_analysis(gamelog_df, test_df)
    print("\n--- how much headroom exists (temporal test set) ---")
    print(f"  mean within-player PTS std        : {floor['mean_within_player_std']:.2f}")
    print(f"  Gaussian MAE floor (perfect mean) : {floor['gaussian_mae_floor']:.3f}")
    print(f"  naive baseline MAE                : {floor['baseline_mae']:.3f}")
    print(f"  LEAKY oracle, full-season mean    : {floor['oracle_full_season_mae']:.3f}")
    print(f"  LEAKY oracle, test-period mean    : {floor['oracle_test_period_mae']:.3f}")
    headroom = floor['baseline_mae'] - floor['oracle_full_season_mae']
    captured = summaries['temporal']['improvement']
    print(f"\n  realistic headroom over baseline  : {headroom:.3f} MAE")
    if headroom > 0:
        print(f"  captured by this model            : {captured:.3f} "
              f"({100 * captured / headroom:.0f}% of it)")

    summaries['noise_floor'] = floor
    return summaries, dataset


if __name__ == "__main__":
    run()
