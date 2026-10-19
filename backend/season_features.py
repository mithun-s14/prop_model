"""
Season-level feature engineering for preseason fantasy projections.

Predicts a player's NEXT season per-game averages from their PREVIOUS season(s).
This is a different problem from features.py, which predicts a single game from
within-season rolling windows: on draft day every rolling window is empty, so
the in-season model cannot produce a preseason number at all.

Design notes (see markdowns/season_projection_model.md):
  - Ratio stats (FG%, FT%, 3P%) are computed as sum(makes)/sum(attempts), NEVER
    as the mean of per-game percentages. Averaging per-game percentages weights
    an 0-for-2 night the same as a 12-for-20 night; measured on this dataset it
    differs from the true season rate by 0.0415 on average and up to 0.436, when
    the entire year-over-year spread of FG% is only 0.071.
  - FG% and FT% are therefore never modelled directly. FGM/FGA and FTM/FTA are
    projected as separate targets and the rate is derived. FGA/g persists at
    r=0.885 across seasons while FG% persists at only 0.747, so projecting the
    volume and letting the rate follow is both more accurate and gives the
    volume term that fantasy value calculations need.
  - Eligibility is asymmetric: >= 20 GP in the feature season (enough history to
    characterise a player) but only >= 15 GP in the target season. A higher
    target bar looks like it buys label quality, but it actually induces
    survivorship bias by dropping players who got hurt or lost a rotation spot.
    Noisy short-season labels are handled by `sample_weight` instead (except for
    the GP target itself, which is fit unweighted -- see train_season_model).
  - `team_changed` is derived from the game logs for training rows -- 34% of
    paired player-seasons changed teams -- so no manual input is needed to
    measure its effect. It is carried as a column, not fitted as a feature: it
    looks into the target season, which inference cannot (see CTX below).
"""
import os

import numpy as np
import pandas as pd

from features import nba_season

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
GAMELOG_PATH = os.path.join(DATA_DIR, 'historical_player_gamelogs.csv')
PLAYER_INFO_PATH = os.path.join(DATA_DIR, 'cached_player_info.csv')
POSITIONS_PATH = os.path.join(DATA_DIR, 'players_positions.csv')

# A partially-scraped season fails quietly rather than loudly: team-level
# quantities like `fga_share` are summed over whoever is present, so a season
# holding half its players inflates every remaining player's share instead of
# leaving a visible gap. Detected RELATIVE to the other seasons in the same file
# (a season with far fewer players than its neighbours), not against an absolute
# NBA-scale constant -- an absolute floor would reject any small test dataset.
# The one case this cannot catch is every season being partial by the same
# factor, which a scrape interrupted mid-run does not produce.
PARTIAL_SEASON_RATIO = 0.5

# Teammates counted toward the ROSTER block's shot volume: season FGA per TEAM
# game (~410 FGA). Training sees every departure in hindsight -- retirements,
# G-League, unsigned -- while inference sees only moves in the roster file. Those
# unrecorded departures are overwhelmingly low-volume players, so counting only
# rotation players closes most of the gap: the mean team-level shift is -5.15
# shots in training vs -0.73 at inference with no floor, -1.79 vs -0.34 at 5.0.
# The per-season demeaning in _attach_teammate_shots removes what remains.
ROTATION_SHOTS_PG = 5.0

# Season end-year that cached_player_info.csv describes. `experience` follows
# Basketball Reference's convention of counting PRIOR seasons ('R' = rookie):
# Dyson Daniels (rookie 2022-23) shows 3, Stephon Castle (rookie 2024-25) 1,
# Cooper Flagg 'R', which only fits 2025-26, matching cache_metadata.json.
# Cross-checked against the game logs: for 375 players whose first logged season
# is after 2019, 363 match 2026 - first_season exactly and 3 (0.8%) differ by
# more than one year.
PLAYER_INFO_SNAPSHOT_SEASON = 2026

MIN_GP_FEATURE = 20   # enough history to profile a player
MIN_GP_TARGET = 15    # deliberately looser -- a higher bar induces survivorship bias
FULL_SEASON_GP = 41   # sample_weight saturates here (half a season)

POSITIONS = ['PG', 'SG', 'SF', 'PF', 'C']

# Counting stats averaged per game. FGM/FGA/FTM/FTA are also summed for ratios.
_COUNTING = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'TOV', 'MIN',
             'FGM', 'FGA', 'FTM', 'FTA', 'FG3A']

# Stats that get both a `_prev` level and a `_per36_prev` rate
_PER36 = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'TOV', 'FGA']

# Targets the model fits directly. FG_PCT/FT_PCT are DERIVED from the
# component targets, never fitted -- see module docstring.
COUNTING_TARGETS = ['pts', 'reb', 'ast', 'stl', 'blk', 'fg3m', 'tov', 'min', 'gp']
COMPONENT_TARGETS = ['fgm', 'fga', 'ftm', 'fta']
ALL_TARGETS = COUNTING_TARGETS + COMPONENT_TARGETS

SEASON_FEATURE_BLOCKS = {
    'PRIOR': ['pts_prev', 'reb_prev', 'ast_prev', 'stl_prev', 'blk_prev',
              'fg3m_prev', 'tov_prev', 'min_prev', 'gp_prev',
              'fgm_prev', 'fga_prev', 'ftm_prev', 'fta_prev'],
    'RATIO': ['fg_pct_prev', 'ft_pct_prev', 'fg3_pct_prev', 'ts_pct_prev',
              'fg3a_rate_prev', 'fta_rate_prev'],
    'RATE': ['pts_per36_prev', 'reb_per36_prev', 'ast_per36_prev',
             'stl_per36_prev', 'blk_per36_prev', 'fg3m_per36_prev',
             'tov_per36_prev', 'fga_per36_prev'],
    'TREND': ['pts_2h_minus_1h', 'min_2h_minus_1h', 'pts_last20', 'min_last20'],
    'PRIOR2': ['pts_prev2', 'min_prev2', 'has_prev2', 'pts_delta_prev'],
    'AGE': ['age', 'age_sq', 'experience_years', 'age_missing'],
    # `team_changed` is deliberately NOT a fitted feature. For a training row it
    # compares team N to team N+1 -- information from the target season -- while
    # at inference it is unknown for anyone missing from the roster file. Fitting
    # it made the evaluation see who moved and the live projection not: zeroing
    # it at test time (what inference does) cost 0.028 PTS MAE, worse than
    # dropping it outright. It stays in the dataset as a column; roster_changes.py
    # uses it as an interval signal, which is what the measurements support.
    'CTX': ['fga_share_prev', 'pts_std_prev', 'n_teams_prev']
           + [f'pos_{p}' for p in POSITIONS],
    # Change in teammates' shot volume between this season's roster and next
    # season's, both valued at feature-season volume -- a star arriving cuts
    # everyone else's shots. Uses next season's ROSTER, which is known preseason
    # (the roster file at inference), never next season's stats. See
    # _attach_teammate_shots.
    'ROSTER': ['tm_shots_delta', 'tm_shots_delta_x_share'],
    # Availability: games missed two years running and minutes load. Injury
    # research mostly measures per-game odds, so the season-scale gain is small;
    # select_blocks drops the block wherever it does not help. On all 7
    # transitions it selects AVAIL for every target except BLK, FTM and FTA;
    # held out, GP MAE 12.315 -> 12.274 and FGA 1.620 -> 1.616 improve while
    # MIN 3.622 -> 3.634 and REB 0.810 -> 0.812 get slightly worse.
    'AVAIL': ['gp_prev2', 'load_min_prev', 'age_x_min_prev', 'missed_20_both'],
}

# missed_20_both fires when both prior seasons had at most this many games
MISSED_20_GP = 62

SEASON_FEATURE_COLUMNS = [c for block in SEASON_FEATURE_BLOCKS.values() for c in block]

_ID_COLUMNS = ['player', 'player_id', 'feature_season', 'target_season',
               'team_prev', 'position']

REQUIRED_COLUMNS = {'PLAYER_NAME', 'GAME_DATE', 'team', 'PTS', 'MIN',
                    'FGM', 'FGA', 'FTM', 'FTA', 'FG3M', 'FG3A',
                    'REB', 'AST', 'STL', 'BLK', 'TOV'}


def _safe_div(numerator, denominator, fill=0.0):
    """Element-wise divide yielding `fill` where the denominator is 0 or missing."""
    numerator = pd.Series(numerator).reset_index(drop=True)
    denominator = pd.Series(denominator).reset_index(drop=True).replace(0, np.nan)
    return (numerator / denominator).fillna(fill).values


def load_positions(path=POSITIONS_PATH):
    """
    player name -> position, whitespace-stripped.

    The CSV carries ' PF' and ' PG' with leading spaces, which would otherwise
    produce 7 distinct values for 5 positions.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return {}
    return {str(r.Player).strip(): str(r.Position).strip() for r in df.itertuples()}


def load_player_info(path=PLAYER_INFO_PATH):
    """player_id -> {'birth_date', 'experience'} for the AGE block."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=['player_id', 'birth_date', 'experience'])
    keep = [c for c in ['player_id', 'player_name', 'birth_date', 'experience'] if c in df.columns]
    return df[keep].copy()


def _parse_experience(value):
    """'R' (rookie) -> 0; numeric strings -> int; anything else -> NaN."""
    text = str(value).strip().upper()
    if text == 'R':
        return 0.0
    try:
        return float(text)
    except ValueError:
        return np.nan


def aggregate_player_seasons(gamelog_df, positions=None):
    """
    Collapse a game log into one row per (player, season).

    Ratio stats are computed as sum(makes)/sum(attempts) -- never the mean of
    per-game percentages. `team` is the player's PRIMARY team for the season
    (the one they played the most games for); `n_teams` records mid-season
    moves separately.
    """
    missing = REQUIRED_COLUMNS - set(gamelog_df.columns)
    if missing:
        raise ValueError(f"gamelog_df is missing required columns: {sorted(missing)}")

    df = gamelog_df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='mixed', errors='coerce')
    df = df.dropna(subset=['GAME_DATE'])
    if df.empty:
        return pd.DataFrame(columns=['player', 'season'])

    df['PLAYER_NAME'] = df['PLAYER_NAME'].astype(str).str.strip()
    df['team'] = df['team'].astype(str).str.strip()
    df['season'] = nba_season(df['GAME_DATE'])

    # Player's share of team shot attempts, for the usage proxy. Computed per
    # game then averaged, so it is not distorted by games the player missed.
    team_fga = df.groupby(['game_id', 'team'])['FGA'].transform('sum') if 'game_id' in df else None
    df['fga_share'] = _safe_div(df['FGA'], team_fga) if team_fga is not None else 0.0

    df = df.sort_values(['PLAYER_NAME', 'season', 'GAME_DATE'])
    grouped = df.groupby(['PLAYER_NAME', 'season'], sort=False)

    agg = grouped.agg(
        gp=('PTS', 'size'),
        pts_std=('PTS', 'std'),
        fga_share=('fga_share', 'mean'),
        n_teams=('team', 'nunique'),
        **{f'{c.lower()}_mean': (c, 'mean') for c in _COUNTING},
        **{f'{c.lower()}_sum': (c, 'sum') for c in ['FGM', 'FGA', 'FTM', 'FTA', 'FG3M', 'FG3A', 'PTS']},
    ).reset_index().rename(columns={'PLAYER_NAME': 'player'})

    # Primary team = most games played for in that season
    primary = (
        df.groupby(['PLAYER_NAME', 'season', 'team']).size().rename('n').reset_index()
        .sort_values('n').groupby(['PLAYER_NAME', 'season']).tail(1)
        .rename(columns={'PLAYER_NAME': 'player'})[['player', 'season', 'team']]
    )
    agg = agg.merge(primary, on=['player', 'season'], how='left')

    # Player_ID is stable across name spellings; carry the modal one.
    if 'Player_ID' in df.columns:
        pid = (
            df.groupby(['PLAYER_NAME', 'season'])['Player_ID']
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
            .rename('player_id').reset_index().rename(columns={'PLAYER_NAME': 'player'})
        )
        agg = agg.merge(pid, on=['player', 'season'], how='left')
    else:
        agg['player_id'] = None

    # --- ratio stats: sum(makes)/sum(attempts), NOT mean of per-game pct ---
    agg['fg_pct'] = _safe_div(agg['fgm_sum'], agg['fga_sum'], fill=np.nan)
    agg['ft_pct'] = _safe_div(agg['ftm_sum'], agg['fta_sum'], fill=np.nan)
    agg['fg3_pct'] = _safe_div(agg['fg3m_sum'], agg['fg3a_sum'], fill=np.nan)
    # True shooting: PTS / (2 * (FGA + 0.44*FTA))
    agg['ts_pct'] = _safe_div(agg['pts_sum'], 2 * (agg['fga_sum'] + 0.44 * agg['fta_sum']), fill=np.nan)
    agg['fg3a_rate'] = _safe_div(agg['fg3a_sum'], agg['fga_sum'])
    agg['fta_rate'] = _safe_div(agg['fta_sum'], agg['fga_sum'])

    # --- per-36 rates: separates role (minutes) from ability ---
    for col in _PER36:
        agg[f'{col.lower()}_per36'] = _safe_div(agg[f'{col.lower()}_mean'] * 36.0, agg['min_mean'])

    # --- within-season trend: 2nd half vs 1st half, and the final 20 games ---
    half = grouped.cumcount() < grouped['PTS'].transform('size') / 2
    df['_is_first_half'] = half.values
    halves = (
        df.groupby(['PLAYER_NAME', 'season', '_is_first_half'])[['PTS', 'MIN']].mean()
        .unstack('_is_first_half')
    )
    trend = pd.DataFrame(index=halves.index)
    for stat in ['PTS', 'MIN']:
        first = halves[(stat, True)] if (stat, True) in halves else np.nan
        second = halves[(stat, False)] if (stat, False) in halves else np.nan
        trend[f'{stat.lower()}_2h_minus_1h'] = (second - first).fillna(0.0)
    trend = trend.reset_index().rename(columns={'PLAYER_NAME': 'player'})
    agg = agg.merge(trend, on=['player', 'season'], how='left')

    last20 = (
        df.groupby(['PLAYER_NAME', 'season']).tail(20)
        .groupby(['PLAYER_NAME', 'season'])[['PTS', 'MIN']].mean()
        .rename(columns={'PTS': 'pts_last20', 'MIN': 'min_last20'})
        .reset_index().rename(columns={'PLAYER_NAME': 'player'})
    )
    agg = agg.merge(last20, on=['player', 'season'], how='left')

    agg['pts_std'] = agg['pts_std'].fillna(0.0)
    if positions:
        clean = {str(k).strip(): str(v).strip() for k, v in positions.items()}
        agg['position'] = agg['player'].map(clean).fillna('UNK')
    else:
        agg['position'] = 'UNK'

    return agg


def _attach_age(rows, player_info, season_col='target_season'):
    """
    Age at Feb 1 of the target season, plus experience as of the target season.

    `experience` in the player info file is a single snapshot taken in
    PLAYER_INFO_SNAPSHOT_SEASON, so it is shifted by the season gap per row
    (a 2020 row gets six fewer years than the snapshot, a 2027 row one more).
    Assigning the raw snapshot to every row would be off by up to six years on
    the oldest transitions. Rows that land below 0 are clipped and counted
    (1 row on the full log).

    Measured on all 7 transitions (leave-one-transition-out): ranking held or
    improved (top-150 859 -> 862 hits, bust/breakout 570 -> 574 of 744) while
    MAE rose slightly (PTS 2.1917 -> 2.2072, MIN 3.5765 -> 3.6221), consistent
    across folds. Kept as a correctness fix: the snapshot gave every row of a
    player the same value, which with age encodes career length -- the old MAE
    was likely flattered by that.

    `cached_player_info.csv` covers ~87% of game-log player IDs, so missingness
    is flagged and imputed to the league median rather than silently zero-filled
    -- an age of 0 would be read by the model as an extreme value, not as absent.
    """
    rows = rows.copy()
    if player_info is None or player_info.empty or 'birth_date' not in player_info.columns:
        rows['age'] = np.nan
        rows['experience_years'] = np.nan
    else:
        info = player_info.copy()
        info['birth_dt'] = pd.to_datetime(info['birth_date'], format='mixed', errors='coerce')
        info['exp'] = info['experience'].map(_parse_experience) if 'experience' in info else np.nan

        merged = rows.merge(
            info[['player_id', 'birth_dt', 'exp']].dropna(subset=['player_id']),
            on='player_id', how='left',
        )
        # Fall back to name matching for rows whose player_id did not resolve
        if 'player_name' in info.columns:
            by_name = info.copy()
            by_name['key'] = by_name['player_name'].astype(str).str.strip()
            lookup = by_name.drop_duplicates('key').set_index('key')
            need = merged['birth_dt'].isna()
            merged.loc[need, 'birth_dt'] = merged.loc[need, 'player'].map(lookup['birth_dt'])
            merged.loc[need, 'exp'] = merged.loc[need, 'player'].map(lookup['exp'])

        feb1 = pd.to_datetime(merged[season_col].astype(int).astype(str) + '-02-01')
        rows['age'] = (feb1 - merged['birth_dt']).dt.days / 365.25
        # The snapshot is one season; shift it to each row's own season
        exp = (merged['exp'] - (PLAYER_INFO_SNAPSHOT_SEASON - merged[season_col].astype(int))).to_numpy()
        n_negative = int((exp < 0).sum())
        if n_negative:
            print(f"  [season_features] {n_negative} rows predate the player's NBA career "
                  f"by snapshot experience -- likely a name/ID mismatch; clipped to 0")
        rows['experience_years'] = np.clip(exp, 0, None)

    rows['age_missing'] = rows['age'].isna().astype(int)
    median_age = rows['age'].median()
    rows['age'] = rows['age'].fillna(median_age if pd.notna(median_age) else 26.0)
    rows['age_sq'] = rows['age'] ** 2
    median_exp = rows['experience_years'].median()
    rows['experience_years'] = rows['experience_years'].fillna(
        median_exp if pd.notna(median_exp) else 3.0
    )
    return rows


def _shots_per_team_game(seasons):
    """Season FGA over the season's game count -- a player's share of a team night."""
    games = seasons.groupby('season')['gp'].transform('max')
    return seasons['fga_sum'] / games.replace(0, np.nan)


def _attach_teammate_shots(rows, seasons, next_team):
    """
    ROSTER block: how much rotation shot volume a player's teammates add or lose.

    `next_team` maps (player, feature_season) -> primary team next season, NaN for
    a player who will not play. Both rosters are valued at FEATURE-season volume,
    so for a player who stays this is incoming minus departing teammates' shots;
    for a mover it is new team minus old team. Demeaned per target season, so the
    feature is relative to the league's typical turnover.
    """
    s = seasons.assign(shots=_shots_per_team_game(seasons).fillna(0.0))
    s = s[s['shots'] >= ROTATION_SHOTS_PG]
    s = s.assign(next_team=[next_team.get(k) for k in zip(s['player'], s['season'])])
    prev_sum = s.groupby(['season', 'team'])['shots'].sum()
    next_sum = s.dropna(subset=['next_team']).groupby(['season', 'next_team'])['shots'].sum()
    own = s.set_index(['player', 'season'])['shots']

    keys = list(zip(rows['player'], rows['feature_season']))
    own_shots = own.reindex(keys).fillna(0.0).to_numpy()
    row_next = [next_team.get(k) for k in keys]
    prev = prev_sum.reindex(list(zip(rows['feature_season'], rows['team_prev']))).fillna(0.0).to_numpy()
    nxt = next_sum.reindex(list(zip(rows['feature_season'], row_next))).fillna(0.0).to_numpy()
    # A player in `s` is in both sums (primary team = team_prev; next team is row_next)
    delta = (nxt - own_shots) - (prev - own_shots)

    rows['tm_shots_delta'] = delta - pd.Series(delta, index=rows.index).groupby(
        rows['target_season']).transform('mean').to_numpy()
    rows['tm_shots_delta_x_share'] = rows['tm_shots_delta'] * rows['fga_share']
    return rows


def _feature_frame(seasons, feature_season_rows, player_info, target_season, next_team):
    """
    Shared feature construction for both training and inference.

    Training and serving MUST go through this one path: at this sample size a
    train/serve skew would be undetectable and would silently poison every
    projection.
    """
    rows = feature_season_rows.copy()
    rows = rows.rename(columns={'season': 'feature_season', 'team': 'team_prev'})
    rows['target_season'] = target_season if target_season is not None else rows['feature_season'] + 1

    # --- PRIOR: previous-season levels ---
    for stat in ['pts', 'reb', 'ast', 'stl', 'blk', 'fg3m', 'tov', 'min', 'fgm', 'fga', 'ftm', 'fta']:
        rows[f'{stat}_prev'] = rows[f'{stat}_mean']
    rows['gp_prev'] = rows['gp']

    # --- RATIO / RATE ---
    for src, dst in [('fg_pct', 'fg_pct_prev'), ('ft_pct', 'ft_pct_prev'),
                     ('fg3_pct', 'fg3_pct_prev'), ('ts_pct', 'ts_pct_prev'),
                     ('fg3a_rate', 'fg3a_rate_prev'), ('fta_rate', 'fta_rate_prev')]:
        rows[dst] = rows[src]
    for col in _PER36:
        rows[f'{col.lower()}_per36_prev'] = rows[f'{col.lower()}_per36']

    # --- TREND ---
    rows['pts_2h_minus_1h'] = rows['pts_2h_minus_1h']
    rows['min_2h_minus_1h'] = rows['min_2h_minus_1h']

    # --- PRIOR2: two seasons back ---
    prev2 = seasons[['player', 'season', 'pts_mean', 'min_mean', 'gp']].rename(
        columns={'season': 'feature_season', 'pts_mean': 'pts_prev2', 'min_mean': 'min_prev2',
                 'gp': 'gp_prev2'}
    )
    prev2['feature_season'] = prev2['feature_season'] + 1  # align to the row's feature season
    rows = rows.merge(prev2, on=['player', 'feature_season'], how='left')
    rows['has_prev2'] = rows['pts_prev2'].notna().astype(int)
    rows['pts_delta_prev'] = (rows['pts_prev'] - rows['pts_prev2']).fillna(0.0)
    rows['pts_prev2'] = rows['pts_prev2'].fillna(0.0)
    rows['min_prev2'] = rows['min_prev2'].fillna(0.0)
    # No season two back: assume last season's availability, not 0 games
    rows['gp_prev2'] = rows['gp_prev2'].fillna(rows['gp_prev'])

    # --- CTX ---
    rows['fga_share_prev'] = rows['fga_share']
    rows['pts_std_prev'] = rows['pts_std']
    rows['n_teams_prev'] = rows['n_teams']
    for pos in POSITIONS:
        rows[f'pos_{pos}'] = (rows['position'] == pos).astype(int)

    # --- ROSTER ---
    rows = _attach_teammate_shots(rows, seasons, next_team)

    # --- AGE ---
    rows = _attach_age(rows, player_info, season_col='target_season')

    # --- AVAIL (needs age) ---
    rows['load_min_prev'] = rows['min_prev'] * rows['gp_prev']
    rows['age_x_min_prev'] = (rows['age'] - 27) * rows['min_prev']
    rows['missed_20_both'] = ((rows['gp_prev'] <= MISSED_20_GP) & (rows['gp_prev2'] <= MISSED_20_GP)
                              & (rows['has_prev2'] == 1)).astype(int)

    # Ratio stats can be undefined (a season with zero FTA exists in the real
    # data). Impute to the pool median with a flag rather than leaving NaN --
    # a 0.0 FT% would mis-value the player catastrophically.
    for col in ['fg_pct_prev', 'ft_pct_prev', 'fg3_pct_prev', 'ts_pct_prev']:
        rows[f'{col}_missing'] = rows[col].isna().astype(int)
        median = rows[col].median()
        rows[col] = rows[col].fillna(median if pd.notna(median) else 0.0)

    return rows


def complete_seasons(seasons, ratio=PARTIAL_SEASON_RATIO, warn=True):
    """
    Season end-years that are not obviously partially scraped.

    A season is treated as partial when its player count falls below `ratio`
    times the median across seasons. See PARTIAL_SEASON_RATIO.
    """
    counts = seasons.groupby('season')['player'].nunique()
    if len(counts) < 2:
        return set(counts.index)
    threshold = counts.median() * ratio
    good = set(counts[counts >= threshold].index)
    if warn:
        for season, n in counts.items():
            if season not in good:
                print(f"  [season_features] skipping season {season}: {n} players vs "
                      f"median {counts.median():.0f} -- looks partially scraped")
    return good


def build_season_dataset(gamelog_df, player_info=None, positions=None,
                         min_gp_feature=MIN_GP_FEATURE, min_gp_target=MIN_GP_TARGET,
                         require_complete_seasons=True):
    """
    One row per (player, season-pair): features from season N, targets from N+1.

    `team_changed` is derived here from the logs, so no manual roster file is
    needed to train or validate it. Returns SEASON_FEATURE_COLUMNS plus id
    columns, `target_*` for every fitted target, `baseline_*` (carry-forward),
    and `sample_weight`.
    """
    seasons = aggregate_player_seasons(gamelog_df, positions=positions)
    empty = pd.DataFrame(columns=_ID_COLUMNS + SEASON_FEATURE_COLUMNS
                         + [f'target_{t}' for t in ALL_TARGETS] + ['sample_weight'])
    if seasons.empty:
        return empty
    if require_complete_seasons:
        seasons = seasons[seasons['season'].isin(complete_seasons(seasons))]
    if seasons.empty or seasons['season'].nunique() < 2:
        return empty

    if player_info is None:
        player_info = load_player_info()

    feature_rows = seasons[seasons['gp'] >= min_gp_feature].copy()
    target_rows = seasons[seasons['gp'] >= min_gp_target].copy()
    if feature_rows.empty or target_rows.empty:
        return empty

    # Join season N features to season N+1 targets
    targets = target_rows.copy()
    targets['feature_season'] = targets['season'] - 1
    target_cols = {f'{t}_mean': f'target_{t}' for t in ALL_TARGETS if t != 'gp'}
    targets = targets.rename(columns=target_cols)
    targets['target_gp'] = targets['gp']
    keep_targets = ['player', 'feature_season', 'season', 'team'] + list(target_cols.values()) + ['target_gp']
    targets = targets[keep_targets].rename(columns={'season': 'target_season', 'team': 'team_next'})

    # Next season's primary team, from the logs -- training's view of the roster
    next_team = {(p, season - 1): team
                 for p, season, team in zip(seasons['player'], seasons['season'], seasons['team'])}
    rows = _feature_frame(seasons, feature_rows, player_info, target_season=None,
                          next_team=next_team)
    rows = rows.drop(columns=['target_season'])
    rows = rows.merge(targets, on=['player', 'feature_season'], how='inner')
    if rows.empty:
        return empty

    # team_changed is observable for training rows: primary team N vs N+1
    rows['team_changed'] = (rows['team_prev'] != rows['team_next']).astype(int)

    # Carry-forward baseline per target, and GP-based sample weight
    for target in ALL_TARGETS:
        rows[f'baseline_{target}'] = rows[f'{target}_prev']
    rows['sample_weight'] = (rows['target_gp'].clip(upper=FULL_SEASON_GP) / FULL_SEASON_GP)

    return _finalize(rows)


def build_inference_features(gamelog_df, feature_season, player_info=None,
                             positions=None, min_gp_feature=MIN_GP_FEATURE,
                             next_teams=None):
    """
    Features for a season with no target yet (2026 -> project 2027).

    `next_teams` maps player name -> next season's team (None/NaN = will not
    play), normally roster_changes.next_teams(). Anyone absent keeps their
    feature-season team.

    Goes through the same `_feature_frame` path as training. `team_changed` is
    0 here because the offseason has not been observed; roster_changes.py sets
    it at projection time. It is not a model input, so this cannot skew a fit.
    """
    seasons = aggregate_player_seasons(gamelog_df, positions=positions)
    empty = pd.DataFrame(columns=_ID_COLUMNS + SEASON_FEATURE_COLUMNS)
    if seasons.empty:
        return empty

    feature_rows = seasons[(seasons['season'] == feature_season)
                           & (seasons['gp'] >= min_gp_feature)].copy()
    if feature_rows.empty:
        return empty

    if player_info is None:
        player_info = load_player_info()

    overrides = next_teams or {}
    season_rows = seasons[seasons['season'] == feature_season]
    next_team = {(p, feature_season): overrides.get(p, team)
                 for p, team in zip(season_rows['player'], season_rows['team'])}
    rows = _feature_frame(seasons, feature_rows, player_info,
                          target_season=feature_season + 1, next_team=next_team)
    rows['team_changed'] = 0
    for target in ALL_TARGETS:
        rows[f'baseline_{target}'] = rows[f'{target}_prev']
    return _finalize(rows, with_targets=False)


def _finalize(rows, with_targets=True):
    """Select the stable column set, clean non-finite values, and order rows."""
    keep = _ID_COLUMNS + SEASON_FEATURE_COLUMNS + ['team_changed']
    keep += [c for c in rows.columns if c.startswith('baseline_')]
    # Imputation flags for the ratio stats are carried for inspection and testing
    # but deliberately kept OUT of SEASON_FEATURE_COLUMNS. They fire on ~1 row in
    # 700, and standardizing a near-constant column turns its single nonzero
    # entry into a huge value -- effectively a memorizable per-row indicator.
    # `age_missing` is a feature by contrast because it fires on ~10% of rows.
    keep += [c for c in rows.columns if c.endswith('_pct_prev_missing')]
    if with_targets:
        keep += [f'target_{t}' for t in ALL_TARGETS] + ['sample_weight', 'team_next']
    keep = [c for c in keep if c in rows.columns]
    result = rows[keep].copy()

    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sort_cols = [c for c in ['target_season', 'player'] if c in result.columns]
    return result.sort_values(sort_cols).reset_index(drop=True)
