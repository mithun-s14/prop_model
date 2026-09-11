"""
Train and evaluate preseason season-average projections for fantasy drafting.

Companion to season_features.py. See markdowns/season_projection_model.md.

Method
------
Ridge on season-pair features, one model per target, fit to the CHANGE from the
previous season (so the penalty shrinks toward carry-forward, not the league
mean), evaluated against three baselines on identical rows:

  1. carry-forward            pred = previous season's value
  2. per-category shrunk      pred = w*prev + (1-w)*league_mean, w fit on TRAIN only
  3. minutes-weighted         pred = per36_prev * min_prev / 36

Ridge rather than a tree ensemble because of sample size: train_engineered_model.py
already found the ensemble overfitting at ~2.6k rows while unregularized
LinearRegression blew up through collinearity, and this problem has fewer rows
and more collinear features (pts_prev, pts_last20 and pts_per36_prev all measure
one thing). Alpha is selected per target -- the low-persistence categories (STL,
FG%, FT%) want heavier regularization than PTS.

Shrinkage is not optional: every category's optimal w is below 1, ranging from
0.375 (GP) to 0.925 (3PM), so a single global constant would be wrong nearly
everywhere. That is why baseline 2 exists and is the real bar.

Evaluation
----------
Ranking accuracy is the primary metric, not MAE. Nobody drafts on absolute
error; what matters is whether the right players land in the right order.
Reported: Spearman rho and top-N retention on 9-cat value, then per-category
MAE with paired t-tests against the baselines.

Protocols:
  - leave-one-transition-out : hold out one season pair, train on the rest. No
                               row-level leakage, but folds before the last train
                               on LATER seasons too, and block/alpha selection
                               uses all rows -- both mildly optimistic.
  - grouped k-fold           : pooled, split so a player never appears in both
                               train and test. Used for alpha selection and
                               residual quantiles. Mildly optimistic -- label it.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from season_features import (
    ALL_TARGETS,
    COUNTING_TARGETS,
    GAMELOG_PATH,
    SEASON_FEATURE_BLOCKS,
    build_inference_features,
    build_season_dataset,
    load_player_info,
    load_positions,
)

from fantasy_value import DEFAULT_POOL_SIZE as _SHIPPED_POOL

PROJECTIONS_PATH = os.path.join(os.path.dirname(GAMELOG_PATH), 'season_projections_2027.csv')

ALPHA_GRID = [1, 3, 10, 30, 100, 300]
# Rounded so the fitted weight is a clean value: np.arange's endpoint lands on
# 1.3000000000000003, which reads as out-of-range wherever w is bounds-checked.
SHRINK_GRID = np.round(np.arange(0.30, 1.301, 0.025), 4)

# Interval half-widths come from cross-validated residual quantiles.
INTERVAL_QUANTILES = (0.10, 0.90)

# Top-of-board blend (design doc section 15.4). Ridge orders the middle and back
# of the pool better than carry-forward but not the first two rounds, so for
# players inside the previous season's top TOP_TIER_CUTOFF by 9-cat value the
# shipped projection is a 50/50 average of the model and their own prior season.
# Measured on all 7 transitions at the shipped pool size: top-25 116 -> 119 hits
# (carry-forward 117), top-12 51 -> 54 (carry-forward 52), rho and top-50/100/150
# unchanged, PTS MAE 2.2290 -> 2.2298. Cutoff 25 was chosen by nested
# leave-one-transition-out (selected on the other six folds, it wins 7/7); the
# weight is a product call within a one-hit spread -- 0.0 scores one more top-25
# hit but publishes last season's raw numbers as an elite player's projection and
# gives up bust/breakout recall (73.9% vs 75.7%).
TOP_TIER_CUTOFF = 25
TOP_TIER_MODEL_WEIGHT = 0.5
TOP_TIER_FLAG = 'TOP_TIER_BLEND'


def columns_for_blocks(blocks):
    """Flatten block names into their feature columns."""
    return [c for b in blocks for c in SEASON_FEATURE_BLOCKS[b]]


ALL_BLOCKS = list(SEASON_FEATURE_BLOCKS)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def carry_forward_baseline(test_df, target):
    """pred = the player's previous-season value. The number to beat."""
    return test_df[f'baseline_{target}'].to_numpy(dtype=float)


def fit_shrink_weight(train_df, target):
    """
    Fit w for `pred = w*prev + (1-w)*mean` on TRAINING rows only.

    Returns (w, league_mean). Regression to the mean is real and per-category:
    the optimal w ranges from 0.375 (GP) to 0.925 (3PM) on this data.
    """
    prev = train_df[f'baseline_{target}'].to_numpy(dtype=float)
    actual = train_df[f'target_{target}'].to_numpy(dtype=float)
    league_mean = float(prev.mean())
    best_w, best_mae = 1.0, np.inf
    for w in SHRINK_GRID:
        mae = np.abs(actual - (w * prev + (1 - w) * league_mean)).mean()
        if mae < best_mae:
            best_w, best_mae = float(w), mae
    return best_w, league_mean


def shrunk_baseline(train_df, test_df, target):
    """Per-category shrunk carry-forward. The real bar for the model to clear."""
    w, league_mean = fit_shrink_weight(train_df, target)
    prev = test_df[f'baseline_{target}'].to_numpy(dtype=float)
    return w * prev + (1 - w) * league_mean


def minutes_weighted_baseline(train_df, test_df, target):
    """
    pred = prior per-36 rate * PROJECTED minutes / 36.

    The projected minutes must be a real projection, not the prior season's.
    Using min_prev makes this baseline algebraically identical to carry-forward:
    per36_prev is defined as stat_mean * 36 / min_mean, so per36_prev * min_prev / 36
    collapses back to stat_mean exactly. (Verified on the real data -- the two
    columns matched to floating-point precision, which is what surfaced it.)
    Minutes are therefore projected with the shrunk baseline, making this a
    genuine alternative: "the per-36 rate persists, but minutes regress."
    """
    rate_col = f'{target}_per36_prev'
    if rate_col not in test_df.columns:
        return carry_forward_baseline(test_df, target)
    projected_minutes = shrunk_baseline(train_df, test_df, 'min')
    return test_df[rate_col].to_numpy(dtype=float) * projected_minutes / 36.0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def fit_ridge(train_df, test_df, columns, target, alpha, use_weights=True,
              residual=True):
    """
    Standardize, fit weighted ridge on train rows, predict test rows.

    With `residual=True` the model is fit to `target - carry_forward` and the
    carry-forward is added back. The ridge penalty then pulls an uncertain
    prediction toward the player's own previous season rather than toward the
    league-average intercept -- the prior this problem actually supports. On
    the 7-transition evaluation it matched or beat the level-target fit on every
    ranking metric at no MAE cost (section 15 of the design doc), though the
    ranking gain is within noise.
    """
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[columns])
    x_test = scaler.transform(test_df[columns])
    weights = train_df['sample_weight'].to_numpy(dtype=float) if use_weights else None
    y = train_df[f'target_{target}'].to_numpy(dtype=float)
    if residual:
        y = y - train_df[f'baseline_{target}'].to_numpy(dtype=float)
    model = Ridge(alpha=alpha)
    model.fit(x_train, y, sample_weight=weights)
    pred = model.predict(x_test)
    if residual:
        pred = pred + test_df[f'baseline_{target}'].to_numpy(dtype=float)
    return pred


def grouped_kfold_indices(dataset, k=5, seed=0):
    """
    K folds split by PLAYER, so a player's two season-pairs never straddle a
    fold boundary. Their rows are correlated; splitting them would leak.
    """
    players = dataset['player'].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(players)
    buckets = {p: i % k for i, p in enumerate(shuffled)}
    fold_of = dataset['player'].map(buckets).to_numpy()
    for fold in range(k):
        yield np.where(fold_of != fold)[0], np.where(fold_of == fold)[0]


def grouped_kfold_predictions(dataset, target, blocks, alpha, k=5):
    """Out-of-fold predictions for every row, used for alpha choice and intervals."""
    columns = columns_for_blocks(blocks)
    out = np.full(len(dataset), np.nan)
    for train_idx, test_idx in grouped_kfold_indices(dataset, k=k):
        out[test_idx] = fit_ridge(dataset.iloc[train_idx], dataset.iloc[test_idx],
                                  columns, target, alpha)
    return out


def select_alpha(dataset, target, blocks, k=5):
    """Choose alpha per target by grouped-CV MAE."""
    scored = []
    for alpha in ALPHA_GRID:
        pred = grouped_kfold_predictions(dataset, target, blocks, alpha, k=k)
        scored.append((mean_absolute_error(dataset[f'target_{target}'], pred), alpha))
    return min(scored)[1]


def select_blocks(dataset, target, seed_blocks=('PRIOR',), k=5):
    """
    Greedy forward selection over feature BLOCKS, scored by grouped CV.

    Blocks rather than individual columns: at this sample size a column-level
    search would simply select noise.
    """
    chosen = list(seed_blocks)

    def score(blocks):
        best = np.inf
        best_alpha = ALPHA_GRID[0]
        for alpha in ALPHA_GRID:
            pred = grouped_kfold_predictions(dataset, target, blocks, alpha, k=k)
            mae = mean_absolute_error(dataset[f'target_{target}'], pred)
            if mae < best:
                best, best_alpha = mae, alpha
        return best, best_alpha

    current_mae, current_alpha = score(chosen)
    while True:
        candidates = []
        for block in ALL_BLOCKS:
            if block in chosen:
                continue
            mae, alpha = score(chosen + [block])
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


# ---------------------------------------------------------------------------
# Comparison statistics
# ---------------------------------------------------------------------------

def compare_to_baseline(actual, model_pred, baseline_pred):
    """
    Paired comparison of absolute errors. Paired because the two predictors are
    highly correlated -- an unpaired view would badly overstate the uncertainty
    of their difference.
    """
    actual = np.asarray(actual, dtype=float)
    err_model = np.abs(actual - np.asarray(model_pred, dtype=float))
    err_base = np.abs(actual - np.asarray(baseline_pred, dtype=float))
    diff = err_base - err_model            # positive => model better

    n = len(diff)
    stderr = diff.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
    p_value = stats.ttest_rel(err_base, err_model).pvalue if n > 1 else float('nan')
    return {
        'n_test': n,
        'baseline_mae': float(err_base.mean()),
        'model_mae': float(err_model.mean()),
        'improvement': float(diff.mean()),
        'improvement_pct': float(100 * diff.mean() / err_base.mean()) if err_base.mean() else 0.0,
        'p_value': float(p_value),
        'ci_low': float(diff.mean() - 1.96 * stderr),
        'ci_high': float(diff.mean() + 1.96 * stderr),
        'win_rate': float((diff > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Ranking metrics -- the primary evaluation (nobody drafts on MAE)
# ---------------------------------------------------------------------------

def _value_frame_from(df, prefix):
    """Assemble the columns fantasy_value expects from a projection prefix."""
    from fantasy_value import RATIO_CATEGORIES

    out = pd.DataFrame(index=df.index)
    mapping = {'PTS': 'pts', 'REB': 'reb', 'AST': 'ast', 'STL': 'stl',
               'BLK': 'blk', 'FG3M': 'fg3m', 'TOV': 'tov'}
    for cat, stat in mapping.items():
        out[cat] = df[f'{prefix}{stat}'].to_numpy(dtype=float)
    fgm = df[f'{prefix}fgm'].to_numpy(dtype=float)
    fga = df[f'{prefix}fga'].to_numpy(dtype=float)
    ftm = df[f'{prefix}ftm'].to_numpy(dtype=float)
    fta = df[f'{prefix}fta'].to_numpy(dtype=float)
    out['FGA'], out['FTA'] = fga, fta
    out['FG_PCT'] = np.divide(fgm, fga, out=np.zeros_like(fgm), where=fga > 0)
    out['FT_PCT'] = np.divide(ftm, fta, out=np.zeros_like(ftm), where=fta > 0)
    _ = RATIO_CATEGORIES
    return out


def ranking_metrics(actual_frame, predicted_frame, pool_size=_SHIPPED_POOL,
                    top_ns=(12, 25, 50, 100, 150)):
    """
    Spearman rho and top-N retention of 9-cat value.

    Both frames are scored with the same z-score machinery so the comparison is
    apples to apples. Actual value is computed from realized season averages.

    `pool_size` defaults to the pool the shipped board is valued against
    (`fantasy_value.DEFAULT_POOL_SIZE`), NOT to None. Scoring over all ~450
    projected players measures a ranking nobody ever sees, and it is not a
    neutral choice: z-scores taken over the whole pool understate the model's
    top-of-board accuracy, because the spread of the back of the pool sets the
    scale. The same folds score top-25 at 109/175 over the full pool and 116/175
    over the top 156 -- most of the "top-25 regression" in section 15.1 was this
    mismatch, not the model. Pass None explicitly to reproduce the old figures.
    """
    from fantasy_value import category_zscores, total_value

    actual_value = total_value(category_zscores(actual_frame, pool_size=pool_size))
    pred_value = total_value(category_zscores(predicted_frame, pool_size=pool_size))
    ok = actual_value.notna() & pred_value.notna()
    actual_value, pred_value = actual_value[ok], pred_value[ok]

    if len(actual_value) < 3:
        return {'n': len(actual_value), 'spearman': float('nan'), 'top_n': {}}

    rho = stats.spearmanr(pred_value, actual_value).statistic
    actual_rank = actual_value.rank(ascending=False)
    pred_rank = pred_value.rank(ascending=False)

    retention = {}
    for n in top_ns:
        if n > len(actual_value):
            continue
        top_actual = set(actual_rank.nsmallest(n).index)
        top_pred = set(pred_rank.nsmallest(n).index)
        retention[n] = len(top_actual & top_pred) / n

    in_top100 = actual_rank <= min(100, len(actual_rank))
    mare = float((pred_rank - actual_rank).abs()[in_top100].mean())

    return {'n': int(len(actual_value)), 'spearman': float(rho),
            'top_n': retention, 'mean_abs_rank_err_top100': mare}


def bust_breakout_recall(actual_frame, predicted_frame, baseline_frame,
                         pool_size=_SHIPPED_POOL, threshold_sd=1.0):
    """
    Of players whose value moved more than `threshold_sd` year-over-year, what
    fraction did the model call in the right direction?

    Directly the thing a drafter wants, and the thing carry-forward cannot do by
    construction: carry-forward predicts no change at all.
    """
    from fantasy_value import category_zscores, total_value

    actual = total_value(category_zscores(actual_frame, pool_size=pool_size))
    pred = total_value(category_zscores(predicted_frame, pool_size=pool_size))
    prior = total_value(category_zscores(baseline_frame, pool_size=pool_size))
    ok = actual.notna() & pred.notna() & prior.notna()
    actual, pred, prior = actual[ok], pred[ok], prior[ok]

    actual_move = actual - prior
    movers = actual_move.abs() > threshold_sd * actual_move.std()
    if movers.sum() == 0:
        return {'n_movers': 0, 'recall': float('nan')}
    pred_move = pred - prior
    correct = np.sign(pred_move[movers]) == np.sign(actual_move[movers])
    return {'n_movers': int(movers.sum()), 'recall': float(correct.mean())}


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

def leave_one_transition_out(dataset, blocks_by_target=None, alpha_by_target=None):
    """
    Hold out one season transition, train on the others. No player-season is on
    both sides, but earlier held-out seasons are fit using later ones, so this
    is mildly optimistic relative to a strict forward-only split.
    """
    transitions = sorted(dataset['target_season'].unique())
    results = []
    for held_out in transitions:
        test_df = dataset[dataset['target_season'] == held_out]
        train_df = dataset[dataset['target_season'] != held_out]
        if len(train_df) < 30 or len(test_df) < 30:
            continue
        results.append((held_out, train_df, test_df))
    return results


def evaluate_transition(train_df, test_df, blocks_by_target, alpha_by_target):
    """All targets, all baselines, on one train/test split."""
    predictions, summaries = {}, {}
    for target in ALL_TARGETS:
        blocks = blocks_by_target.get(target, ['PRIOR'])
        alpha = alpha_by_target.get(target, 30)
        predictions[target] = fit_ridge(train_df, test_df, columns_for_blocks(blocks),
                                        target, alpha)
    # Makes come from shrunk rate x projected attempts, exactly as in production
    predictions, _ = project_makes_from_rates(train_df, test_df, predictions)
    predictions, _ = blend_top_tier(test_df, predictions)

    for target in ALL_TARGETS:
        blocks = blocks_by_target.get(target, ['PRIOR'])
        alpha = alpha_by_target.get(target, 30)
        pred = predictions[target]
        actual = test_df[f'target_{target}']
        summaries[target] = {
            'vs_carry_forward': compare_to_baseline(actual, pred, carry_forward_baseline(test_df, target)),
            'vs_shrunk': compare_to_baseline(actual, pred, shrunk_baseline(train_df, test_df, target)),
            'vs_minutes': compare_to_baseline(actual, pred, minutes_weighted_baseline(train_df, test_df, target)),
            'blocks': blocks, 'alpha': alpha,
        }
    return predictions, summaries


def baseline_report(dataset):
    """
    Per-category baseline table, computed before any model is fitted.

    Establishing the bar first is deliberate: if the model never clears these,
    the shrunk baseline is the deliverable and knowing that early prevents
    wasted work.
    """
    rows = []
    for target in ALL_TARGETS:
        actual = dataset[f'target_{target}'].to_numpy(dtype=float)
        prev = dataset[f'baseline_{target}'].to_numpy(dtype=float)
        w, mean = fit_shrink_weight(dataset, target)
        shrunk = w * prev + (1 - w) * mean
        minutes = minutes_weighted_baseline(dataset, dataset, target)
        corr = np.corrcoef(prev, actual)[0, 1] if len(actual) > 2 else float('nan')
        rows.append({
            'target': target,
            'n': len(actual),
            'corr': corr,
            'sd_next': actual.std(),
            'mae_carry_forward': np.abs(actual - prev).mean(),
            'shrink_w': w,
            'mae_shrunk': np.abs(actual - shrunk).mean(),
            'mae_minutes': np.abs(actual - minutes).mean(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Intervals, ratio derivation, projection
# ---------------------------------------------------------------------------

def residual_quantiles(dataset, target, blocks, alpha, k=5,
                       quantiles=INTERVAL_QUANTILES):
    """
    Empirical residual quantiles from out-of-fold predictions.

    Not a Bayesian posterior -- an honest empirical band. A single-number season
    projection with no error bar invites more confidence than this much data
    supports.
    """
    pred = grouped_kfold_predictions(dataset, target, blocks, alpha, k=k)
    residuals = dataset[f'target_{target}'].to_numpy(dtype=float) - pred
    low, high = np.quantile(residuals, quantiles)
    return float(low), float(high)


# Pseudo-attempts of league-average shooting added to a player's prior season.
RATE_PRIOR_K_GRID = [0, 10, 25, 50, 100, 200, 400, 800]
RATE_COMPONENTS = [('fgm', 'fga'), ('ftm', 'fta')]


def _season_rate_inputs(df, makes, attempts):
    """Prior-season rate, season attempt volume, and next-season rate (if present)."""
    gp = df['gp_prev'].to_numpy(dtype=float)
    made = df[f'baseline_{makes}'].to_numpy(dtype=float) * gp
    tried = df[f'baseline_{attempts}'].to_numpy(dtype=float) * gp
    target = None
    if f'target_{attempts}' in df.columns:
        t_att = df[f'target_{attempts}'].to_numpy(dtype=float)
        target = np.divide(df[f'target_{makes}'].to_numpy(dtype=float), t_att,
                           out=np.full_like(t_att, np.nan), where=t_att > 0)
    return made, tried, target


def fit_rate_prior(train_df, makes, attempts):
    """
    Fit k for `rate = (made + k*league) / (attempts + k)` on TRAINING rows only.

    Scored by next-season attempt-weighted MAE, because fantasy impact is
    (rate - league) * attempts. Returns (k, league_rate).
    """
    made, tried, target = _season_rate_inputs(train_df, makes, attempts)
    league = float(made.sum() / tried.sum()) if tried.sum() > 0 else 0.0
    weight = train_df[f'target_{attempts}'].to_numpy(dtype=float)
    ok = np.isfinite(target) & (weight > 0)
    best_k, best_err = RATE_PRIOR_K_GRID[0], np.inf
    for k in RATE_PRIOR_K_GRID:
        denom = tried + k
        rate = np.divide(made + k * league, denom, out=np.full_like(denom, league),
                         where=denom > 0)
        err = np.average(np.abs(target[ok] - rate[ok]), weights=weight[ok])
        if err < best_err:
            best_k, best_err = k, err
    return best_k, league


def shrunk_rate(df, makes, attempts, k, league):
    """Volume-aware shrinkage: a player with few attempts is pulled harder to the league."""
    made, tried, _ = _season_rate_inputs(df, makes, attempts)
    denom = tried + k
    return np.divide(made + k * league, denom, out=np.full_like(denom, league),
                     where=denom > 0)


def project_makes_from_rates(train_df, test_df, predictions):
    """
    Replace projected makes with shrunk_rate * projected attempts.

    Why not the FGM/FTM models: dividing two independently noisy projections
    amplifies noise, worst for low-volume shooters. On the 7-transition
    evaluation the model-ratio FG% lost to shrunk carry-forward (MAE 0.0393 vs
    0.0358); volume-aware shrinkage matched the best method on FG% and beat
    every alternative on FT% (0.0606). Section 4's rule still holds -- FG% is
    exactly makes/attempts, and volume is projected separately by the model.

    `predictions` maps target -> array; returns a copy with makes replaced and
    the fitted (k, league) per rate.
    """
    out = dict(predictions)
    priors = {}
    for makes, attempts in RATE_COMPONENTS:
        k, league = fit_rate_prior(train_df, makes, attempts)
        rate = shrunk_rate(test_df, makes, attempts, k, league)
        out[makes] = rate * np.clip(np.asarray(predictions[attempts], dtype=float), 0, None)
        priors[makes] = (k, league)
    return out, priors


def prior_value_rank(df, prefix='baseline_'):
    """
    Rank players by their PREVIOUS season's 9-cat value, best first.

    Uses only prior-season columns, so it is available at inference time -- this
    is the tier a drafter already knows going in, not a model output.
    """
    from fantasy_value import category_zscores, total_value

    value = total_value(category_zscores(_value_frame_from(df, prefix),
                                         pool_size=_SHIPPED_POOL))
    return value.rank(ascending=False, na_option='bottom')


def blend_top_tier(test_df, predictions, cutoff=TOP_TIER_CUTOFF,
                   weight=TOP_TIER_MODEL_WEIGHT):
    """
    Pull projections for last season's top `cutoff` players toward carry-forward.

    Ridge wins on rho and top-50 through top-150 but loses the first two rounds,
    which is where a draft is decided. Averaging the model with the player's own
    prior season inside the top tier recovers that without touching the rest of
    the board. Returns (predictions, mask) where `mask` marks the blended rows.

    Blending every category rather than just the ranking value is deliberate: the
    board's displayed per-category numbers and the order it sorts by have to come
    from the same projection, or a user who checks the math finds they disagree.
    """
    rank = prior_value_rank(test_df).to_numpy(dtype=float)
    mask = rank <= cutoff
    if not mask.any():
        return predictions, mask
    blended = {}
    for target in ALL_TARGETS:
        model = np.asarray(predictions[target], dtype=float)
        carry = test_df[f'baseline_{target}'].to_numpy(dtype=float)
        # NaN carry-forward (no usable prior season) keeps the model's number
        pull = np.where(mask & np.isfinite(carry),
                        weight * model + (1.0 - weight) * carry, model)
        blended[target] = pull
    return blended, mask


def derive_ratio_projections(projections):
    """
    FG% and FT% from projected components -- never fitted directly.

    A projected attempt count of 0 yields the league rate rather than a division
    by zero or an inf.
    """
    df = projections.copy()
    for pct, makes, attempts in [('pred_fg_pct', 'pred_fgm', 'pred_fga'),
                                 ('pred_ft_pct', 'pred_ftm', 'pred_fta')]:
        m = df[makes].to_numpy(dtype=float)
        a = df[attempts].to_numpy(dtype=float)
        total_a = a.sum()
        league = float(m.sum() / total_a) if total_a > 0 else 0.0
        df[pct] = np.divide(m, a, out=np.full_like(m, league), where=a > 0)
    return df


def clip_to_sane_bounds(projections):
    """
    Clamp projections to physically plausible ranges.

    A projection outside these is a bug, not a bold take. Ridge is unbounded and
    will happily extrapolate a negative steal rate.
    """
    bounds = {
        'pred_pts': (0, 45), 'pred_reb': (0, 20), 'pred_ast': (0, 15),
        'pred_stl': (0, 4), 'pred_blk': (0, 5), 'pred_fg3m': (0, 7),
        'pred_tov': (0, 7), 'pred_min': (0, 42), 'pred_gp': (0, 82),
        'pred_fgm': (0, 20), 'pred_fga': (0, 35),
        'pred_ftm': (0, 15), 'pred_fta': (0, 20),
        'pred_fg_pct': (0.25, 0.75), 'pred_ft_pct': (0.35, 1.0),
    }
    df = projections.copy()
    for column, (low, high) in bounds.items():
        if column in df.columns:
            df[column] = df[column].clip(low, high)
        for suffix in ('_low', '_high'):
            col = f'{column}{suffix}'
            if col in df.columns:
                df[col] = df[col].clip(low, high)
    return df


def fit_and_project(dataset, inference_df, blocks_by_target, alpha_by_target, k=5):
    """Fit each target on the full dataset and project the inference rows."""
    out = inference_df[['player', 'player_id', 'team_prev', 'position',
                        'age', 'gp_prev']].copy().reset_index(drop=True)

    predictions = {}
    for target in ALL_TARGETS:
        columns = columns_for_blocks(blocks_by_target.get(target, ['PRIOR']))
        predictions[target] = fit_ridge(dataset, inference_df, columns, target,
                                        alpha_by_target.get(target, 30))
    predictions, _ = project_makes_from_rates(dataset, inference_df, predictions)
    predictions, top_tier = blend_top_tier(inference_df, predictions)

    for target in ALL_TARGETS:
        blocks = blocks_by_target.get(target, ['PRIOR'])
        alpha = alpha_by_target.get(target, 30)
        pred = predictions[target]
        # Band half-widths from the fitted model's CV residuals, centred on the
        # shipped point estimate (for makes, the rate-based one).
        low, high = residual_quantiles(dataset, target, blocks, alpha, k=k)
        out[f'pred_{target}'] = pred
        out[f'pred_{target}_low'] = pred + low
        out[f'pred_{target}_high'] = pred + high
        out[f'{target}_prev'] = inference_df[f'baseline_{target}'].to_numpy(dtype=float)

    out = derive_ratio_projections(out)
    out = clip_to_sane_bounds(out)
    out['n_prior_seasons'] = 1 + inference_df['has_prev2'].to_numpy(dtype=int)
    out['team_changed'] = inference_df['team_changed'].to_numpy(dtype=int)
    out['role_change'] = ''
    # Recorded, not hidden: a user comparing two players in the first two rounds
    # should be able to see which projections were pulled toward last season.
    out['flags'] = np.where(top_tier, TOP_TIER_FLAG, '')
    return out


def attach_fantasy_value(projections, pool_size=None, punt=None):
    """Attach 9-cat z-scores and league values to a projection frame."""
    from fantasy_value import DEFAULT_POOL_SIZE, build_value_frame

    frame = projections.copy()
    renames = {'pred_pts': 'PTS', 'pred_reb': 'REB', 'pred_ast': 'AST',
               'pred_stl': 'STL', 'pred_blk': 'BLK', 'pred_fg3m': 'FG3M',
               'pred_tov': 'TOV', 'pred_fg_pct': 'FG_PCT', 'pred_ft_pct': 'FT_PCT',
               'pred_fga': 'FGA', 'pred_fta': 'FTA'}
    for src, dst in renames.items():
        frame[dst] = frame[src]
    valued = build_value_frame(
        frame, pool_size=pool_size if pool_size is not None else DEFAULT_POOL_SIZE,
        punt=punt,
    )
    return valued.drop(columns=list(renames.values()))


def project_next_season(gamelog_df=None, feature_season=2026, gamelog_path=GAMELOG_PATH,
                        roster_changes=None, blocks_by_target=None,
                        alpha_by_target=None, verbose=True):
    """
    Full preseason projection for `feature_season + 1`.

    Fits on every available season transition, projects the players with a
    qualifying `feature_season`, derives FG%/FT% from components, attaches
    fantasy value, and applies roster changes if supplied.
    """
    if gamelog_df is None:
        gamelog_df = pd.read_csv(gamelog_path)
    positions = load_positions()
    player_info = load_player_info()

    dataset = build_season_dataset(gamelog_df, player_info=player_info, positions=positions)
    inference = build_inference_features(gamelog_df, feature_season,
                                         player_info=player_info, positions=positions)
    if dataset.empty or inference.empty:
        return pd.DataFrame()

    blocks_by_target = blocks_by_target or {}
    alpha_by_target = alpha_by_target or {}
    for target in ALL_TARGETS:
        if target not in blocks_by_target:
            blocks, alpha, _ = select_blocks(dataset, target)
            blocks_by_target[target] = blocks
            alpha_by_target.setdefault(target, alpha)
            if verbose:
                print(f"  {target:5s} blocks={'+'.join(blocks):28s} alpha={alpha}")

    projections = fit_and_project(dataset, inference, blocks_by_target, alpha_by_target)

    if roster_changes is not None and len(roster_changes):
        from roster_changes import apply_to_projections
        projections = apply_to_projections(projections, roster_changes)

    return attach_fantasy_value(projections)


def output_columns():
    """The projection CSV schema, in order (design doc section 11)."""
    columns = ['player', 'player_id', 'team_prev', 'position', 'age', 'gp_prev']
    for target in ALL_TARGETS:
        if target != 'gp':                      # gp_prev is already an id column
            columns.append(f'{target}_prev')
        columns += [f'pred_{target}', f'pred_{target}_low', f'pred_{target}_high']
    columns += ['pred_fg_pct', 'pred_ft_pct']
    from fantasy_value import NINE_CAT
    columns += [f'z_{c}' for c in NINE_CAT] + ['value_total', 'value_points_league']
    columns += ['n_prior_seasons', 'team_changed', 'role_change', 'flags']
    return columns


def write_projections(board, path=PROJECTIONS_PATH, excluded=None):
    """
    Write the draft board in schema order, best value first.

    Rookies (NaN value) sort last rather than being ranked on fabricated zeros.
    Players dropped by the roster file go to a sibling `_excluded.csv` with a
    reason code, so nobody vanishes without a record.
    """
    board = board.reindex(columns=output_columns())
    board = board.sort_values('value_total', ascending=False, na_position='last')
    board.to_csv(path, index=False)
    if excluded is not None:
        excluded.to_csv(path.replace('.csv', '_excluded.csv'), index=False)
    return board


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_baseline_table(report):
    print("\n--- per-category baselines (all rows, before any model) ---")
    header = (f"  {'target':6s} {'n':>4s} {'corr':>6s} {'sd':>7s} "
              f"{'carry':>7s} {'w':>6s} {'shrunk':>7s} {'min-wt':>7s}")
    print(header)
    for row in report.itertuples():
        print(f"  {row.target:6s} {row.n:4d} {row.corr:6.3f} {row.sd_next:7.3f} "
              f"{row.mae_carry_forward:7.3f} {row.shrink_w:6.3f} "
              f"{row.mae_shrunk:7.3f} {row.mae_minutes:7.3f}")


def _print_target_summary(target, summary):
    cf = summary['vs_carry_forward']
    sh = summary['vs_shrunk']
    verdict = 'BEATS' if sh['improvement'] > 0 and sh['p_value'] < 0.05 else '     '
    print(f"  {target:5s} model {cf['model_mae']:7.3f} | carry {cf['baseline_mae']:7.3f} "
          f"({cf['improvement']:+.3f}) | shrunk {sh['baseline_mae']:7.3f} "
          f"({sh['improvement']:+.3f}, p={sh['p_value']:.3f}) {verdict}")


def run(gamelog_path=GAMELOG_PATH, verbose=True, select=True):
    """Full evaluation report: baselines, per-target models, ranking metrics."""
    gamelog_df = pd.read_csv(gamelog_path)
    positions = load_positions()
    player_info = load_player_info()
    dataset = build_season_dataset(gamelog_df, player_info=player_info, positions=positions)

    print("=" * 78)
    print("SEASON PROJECTION MODEL -- PRESEASON FANTASY PROJECTIONS")
    print("=" * 78)
    if dataset.empty:
        print("Not enough data: need at least two seasons of game logs.")
        return {}
    transitions = sorted(dataset['target_season'].unique())
    print(f"rows: {len(dataset)}   transitions: {transitions}   "
          f"players: {dataset['player'].nunique()}")

    report = baseline_report(dataset)
    _print_baseline_table(report)

    blocks_by_target, alpha_by_target = {}, {}
    if select:
        print("\n--- block selection (grouped CV, per target) ---")
        for target in ALL_TARGETS:
            blocks, alpha, mae = select_blocks(dataset, target)
            blocks_by_target[target] = blocks
            alpha_by_target[target] = alpha
            print(f"  {target:5s} {'+'.join(blocks):32s} alpha={alpha:<4d} cv_mae={mae:.3f}")
    else:
        for target in ALL_TARGETS:
            blocks_by_target[target] = ['PRIOR']
            alpha_by_target[target] = 30

    print("\n--- leave-one-transition-out ---")
    all_summaries = {}
    for held_out, train_df, test_df in leave_one_transition_out(dataset):
        print(f"\n  test season {held_out}  (train {len(train_df)} / test {len(test_df)})")
        predictions, summaries = evaluate_transition(train_df, test_df,
                                                     blocks_by_target, alpha_by_target)
        for target in ALL_TARGETS:
            _print_target_summary(target, summaries[target])
        all_summaries[held_out] = summaries

        actual = _value_frame_from(test_df, 'target_')
        pred_frame = pd.DataFrame({f'pred_{t}': predictions[t] for t in ALL_TARGETS})
        pred_frame = derive_ratio_projections(pred_frame)
        model_frame = _value_frame_from(
            pred_frame.rename(columns={f'pred_{t}': f'm_{t}' for t in ALL_TARGETS}), 'm_')
        base_frame = _value_frame_from(test_df, 'baseline_')

        model_rank = ranking_metrics(actual, model_frame)
        base_rank = ranking_metrics(actual, base_frame)
        print(f"    9-cat rank  model rho {model_rank['spearman']:.3f} "
              f"vs carry-forward rho {base_rank['spearman']:.3f}")
        for n in sorted(model_rank['top_n']):
            print(f"      top-{n:<4d} retention  model {100*model_rank['top_n'][n]:5.1f}%  "
                  f"carry-forward {100*base_rank['top_n'][n]:5.1f}%")
        bb = bust_breakout_recall(actual, model_frame, base_frame)
        print(f"    bust/breakout direction recall: {100*bb['recall']:.1f}% "
              f"of {bb['n_movers']} movers")
        all_summaries[held_out]['_ranking'] = {'model': model_rank, 'baseline': base_rank,
                                              'bust_breakout': bb}

    return {'dataset': dataset, 'baselines': report, 'summaries': all_summaries,
            'blocks': blocks_by_target, 'alphas': alpha_by_target}


def main(argv=None):
    """
    python train_season_model.py              evaluation report (section 8)
    python train_season_model.py --project    write season_projections_2027.csv
    """
    import argparse
    from roster_changes import excluded_players, load_roster_changes, validate

    parser = argparse.ArgumentParser(description='Preseason fantasy projections')
    parser.add_argument('--project', action='store_true',
                        help='fit on every transition and write the projection CSV')
    parser.add_argument('--feature-season', type=int, default=2026)
    parser.add_argument('--out', default=PROJECTIONS_PATH)
    args = parser.parse_args(argv)

    if not args.project:
        run()
        return 0

    gamelog_df = pd.read_csv(GAMELOG_PATH)
    changes = load_roster_changes(gamelog_df=gamelog_df)
    report = validate(changes, gamelog_df)
    if not report.ok:
        # Section 10.6: nothing reaches the model until the file validates
        print(report)
        print("roster_changes_2027.csv failed validation -- nothing written")
        return 1

    board = project_next_season(gamelog_df, feature_season=args.feature_season,
                                roster_changes=changes)
    board = write_projections(board, args.out, excluded=excluded_players(changes))
    print(f"wrote {len(board)} players to {args.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
