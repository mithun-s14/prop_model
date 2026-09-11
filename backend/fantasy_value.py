"""
Fantasy value aggregation: per-category z-scores, punt profiles, points leagues.

Turns per-category projections into the single comparable number a draft board
sorts on, plus the nine components behind it.

Two decisions matter here (markdowns/season_projection_model.md section 7):

  1. Percentage categories use VOLUME-WEIGHTED IMPACT, not the raw rate. A
     player's FG% contribution is (their FG% - league FG%) * their FGA. A center
     shooting .620 on 4 attempts is nearly irrelevant; one shooting .580 on 18
     swings the category. Ranking on the raw rate is the single most common flaw
     in naive fantasy value calculations.

  2. Rows flagged as unprojected (rookies, emitted as 0.0) are EXCLUDED from the
     pool mean and standard deviation. Zero-rows entering the pool shift every
     other player's z-scores -- measured at up to 1.23 of value and +7.3% pool
     SD with 60 rookies. They stay in the output with NaN value so the app can
     render them as "not projected" rather than as bottom-ranked players.
"""
import numpy as np
import pandas as pd

# 9-cat. TOV is negative (fewer is better); FG_PCT/FT_PCT are volume-weighted.
COUNTING_CATEGORIES = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M']
NEGATIVE_CATEGORIES = ['TOV']
RATIO_CATEGORIES = {'FG_PCT': 'FGA', 'FT_PCT': 'FTA'}
NINE_CAT = COUNTING_CATEGORIES + NEGATIVE_CATEGORIES + list(RATIO_CATEGORIES)

DEFAULT_POINTS_FORMULA = {
    'PTS': 1.0, 'REB': 1.2, 'AST': 1.5, 'STL': 3.0, 'BLK': 3.0, 'TOV': -1.0,
}

# Rows carrying any of these flags are kept in the output but excluded from the
# z-score pool. See module docstring.
DEFAULT_EXCLUDE_FLAGS = ('NO_PRIOR_SEASON',)

DEFAULT_POOL_SIZE = 156      # 12 teams x 13 roster spots


def _flagged(df, exclude_flags):
    """Boolean mask of rows carrying any excluded flag."""
    if 'flags' not in df.columns or not exclude_flags:
        return pd.Series(False, index=df.index)
    text = df['flags'].fillna('').astype(str)
    mask = pd.Series(False, index=df.index)
    for flag in exclude_flags:
        mask |= text.str.contains(flag, regex=False)
    return mask


def _ratio_impact(df, pct_col, att_col):
    """
    Volume-weighted impact of a percentage category.

    (player rate - league rate) * player attempts, where the league rate is
    itself attempt-weighted. This is what makes a high-volume .580 shooter beat
    a low-volume .620 one.
    """
    rate = pd.to_numeric(df[pct_col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    attempts = pd.to_numeric(df[att_col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    total = attempts.sum()
    league_rate = float((rate * attempts).sum() / total) if total > 0 else 0.0
    return (rate - league_rate) * attempts


def category_zscores(projections, pool_size=DEFAULT_POOL_SIZE,
                     exclude_flags=DEFAULT_EXCLUDE_FLAGS, value_column=None):
    """
    Per-category z-scores over the projected player pool.

    `pool_size` defines the replacement-level pool the z-scores are computed
    against (default 12 teams x 13 spots). Z-scores over all ~450 projected
    players differ materially from z-scores over the top 156, and the
    replacement-level pool is the correct one for draft valuation.

    Returns a DataFrame of z_<CATEGORY> columns aligned to `projections`.
    Excluded rows get NaN.
    """
    df = projections.copy().reset_index(drop=True)
    excluded = _flagged(df, exclude_flags)
    eligible = df[~excluded]
    if eligible.empty:
        return pd.DataFrame({f'z_{c}': np.full(len(df), np.nan) for c in NINE_CAT})

    # Rank by a rough proxy first to pick the pool, then compute z over that pool
    proxy_col = value_column if value_column in eligible else 'PTS'
    if pool_size and pool_size < len(eligible):
        pool_idx = eligible[proxy_col].astype(float).nlargest(pool_size).index
    else:
        pool_idx = eligible.index
    pool = df.loc[pool_idx]

    out = pd.DataFrame(index=df.index)
    for cat in COUNTING_CATEGORIES + NEGATIVE_CATEGORIES:
        values = pd.to_numeric(df[cat], errors='coerce').astype(float)
        mean = pd.to_numeric(pool[cat], errors='coerce').astype(float).mean()
        sd = pd.to_numeric(pool[cat], errors='coerce').astype(float).std()
        z = (values - mean) / sd if sd and sd > 0 else values * 0.0
        out[f'z_{cat}'] = -z if cat in NEGATIVE_CATEGORIES else z

    for pct_col, att_col in RATIO_CATEGORIES.items():
        impact_all = _ratio_impact(df, pct_col, att_col)
        impact_pool = impact_all[df.index.isin(pool_idx)]
        # ddof=1 to match pandas' default for the counting categories above.
        # numpy's default ddof=0 would standardize the ratio categories by a
        # slightly smaller divisor, silently giving them ~0.8% more weight in
        # the 9-cat total at n=60 -- a real thumb on the scale, not rounding.
        mean = impact_pool.mean()
        sd = impact_pool.std(ddof=1) if len(impact_pool) > 1 else 0.0
        out[f'z_{pct_col}'] = (impact_all - mean) / sd if sd > 0 else impact_all * 0.0

    out.loc[excluded, :] = np.nan
    return out


def total_value(z_df, punt=None):
    """
    Sum of per-category z-scores, optionally punting categories.

    Because value is a plain sum, a punt strategy is just a re-sum over a
    subset -- no re-projection needed. That is why the output ships the nine
    individual z-scores rather than only the total.
    """
    punt = set(punt or [])
    columns = [f'z_{c}' for c in NINE_CAT if c not in punt]
    missing = [c for c in columns if c not in z_df.columns]
    if missing:
        raise ValueError(f"z-score frame is missing columns: {missing}")
    return z_df[columns].sum(axis=1, min_count=len(columns))


def points_league_value(projections, formula=None):
    """Apply a linear points-league scoring formula to projected components."""
    formula = formula or DEFAULT_POINTS_FORMULA
    missing = [c for c in formula if c not in projections.columns]
    if missing:
        raise ValueError(f"projections are missing columns: {missing}")
    total = pd.Series(0.0, index=projections.index)
    for column, weight in formula.items():
        total = total + pd.to_numeric(projections[column], errors='coerce').fillna(0.0) * weight
    excluded = _flagged(projections, DEFAULT_EXCLUDE_FLAGS)
    total[excluded.to_numpy()] = np.nan
    return total


def build_value_frame(projections, pool_size=DEFAULT_POOL_SIZE, punt=None,
                      exclude_flags=DEFAULT_EXCLUDE_FLAGS):
    """Attach z-scores, 9-cat total, and points-league value to a projection frame."""
    df = projections.copy().reset_index(drop=True)
    z = category_zscores(df, pool_size=pool_size, exclude_flags=exclude_flags)
    df = pd.concat([df, z], axis=1)
    df['value_total'] = total_value(z, punt=punt)
    df['value_points_league'] = points_league_value(df)
    return df
