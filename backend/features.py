"""
Point-in-time feature engineering for NBA points prediction.

Every feature here is built with an explicit `.shift(1)` so that a row's
features are derived ONLY from games strictly before that game. This makes
the resulting dataset leakage-free by construction, unlike the current-season
snapshots (usage rate, opponent defense from the Excel file) used by
model.py, which mix in information from after the game being predicted.

Design notes on what changed vs. the original 18-feature set:
  - `spread` / `total` were constants in the backtest (no historical odds are
    scraped), so they were pure noise columns. Dropped.
  - `usage_rate` was a current-season snapshot: constant per player and
    mildly leaky. Replaced by `fga_share_r5`, a point-in-time usage proxy
    (player's share of team field goal attempts over prior games).
  - Opponent defense came from a current-season Excel snapshot. Replaced by
    `opp_pts_allowed_pit` / `opp_pts_allowed_pos_pit`, computed from the game
    logs themselves using only prior games.
  - `pts_season` (expanding mean of prior points) is added deliberately: it
    IS the naive baseline, so the model gets the baseline as an input and
    only has to learn corrections on top of it.
"""
import numpy as np
import pandas as pd

# Feature columns produced by build_feature_dataset, in a stable order.
FEATURE_COLUMNS = [
    # Scoring history at several horizons (pts_season == the naive baseline)
    'pts_r3', 'pts_r5', 'pts_r10', 'pts_season',
    # Volatility and form
    'pts_std_r5', 'momentum_3_vs_season',
    # Workload
    'min_r3', 'min_r5', 'min_season', 'min_trend',
    # Volume / shot profile
    'fga_r5', 'fg3a_r5', 'fta_r5', 'fga_share_r5',
    'fg3a_rate_r5', 'fta_rate_r5',
    # Efficiency
    'pts_per_min_r5', 'pts_per_fga_r5', 'fga_per_min_r5',
    # Secondary box score
    'reb_r5', 'ast_r5', 'tov_r5', 'stl_r5', 'blk_r5',
    # Schedule / context
    'is_home', 'rest_days', 'is_b2b', 'games_played',
    # Point-in-time opponent defense
    'opp_pts_allowed_pit', 'opp_pts_allowed_pos_pit',
]

REQUIRED_COLUMNS = {'PLAYER_NAME', 'GAME_DATE', 'game_id', 'team', 'PTS'}

# All per-player history is grouped by (player, season): rolling and expanding
# windows must reset at a season boundary. Without this, a player's form from a
# previous season would leak into the current season's "season average" -- which
# is also the naive baseline, so it would corrupt the very thing we benchmark
# against. Matters only once the game logs span more than one season.
_PLAYER_KEYS = ['PLAYER_NAME', 'season']

# Box score columns rolled into 5-game averages
_ROLL_COLS = ['PTS', 'MIN', 'FGA', 'FG3A', 'FTA', 'REB', 'AST', 'TOV', 'STL', 'BLK']


def _shift_roll(group_series, window, stat='mean'):
    """Rolling stat over strictly-prior games (shift(1) before rolling)."""
    shifted = group_series.shift(1)
    roller = shifted.rolling(window, min_periods=1)
    return getattr(roller, stat)()


def nba_season(dates):
    """
    NBA season end-year for each date. A season spans October to June, so games
    from October onward belong to the following calendar year's season
    (2025-10-22 -> 2026, matching Basketball Reference's season numbering).
    """
    dates = pd.to_datetime(dates)
    return dates.dt.year + (dates.dt.month >= 10).astype(int)


def _safe_div(numerator, denominator):
    """Element-wise divide, yielding 0.0 where the denominator is 0 or missing."""
    denominator = pd.Series(denominator).replace(0, np.nan)
    return (pd.Series(numerator) / denominator).fillna(0.0).values


def prepare_gamelog(gamelog_df, positions=None):
    """
    Normalize the raw gamelog: parse dates, derive home/away and opponent from
    the Basketball-Reference game_id (whose last 3 chars are the home team),
    and attach each player's position.

    `positions` maps player name -> position; players missing from it are
    labelled 'UNK' (they still get team-level defense features).
    """
    missing = REQUIRED_COLUMNS - set(gamelog_df.columns)
    if missing:
        raise ValueError(f"gamelog_df is missing required columns: {missing}")

    df = gamelog_df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='mixed', errors='coerce')
    df = df.dropna(subset=['GAME_DATE'])
    df['team'] = df['team'].astype(str).str.strip()

    df['season'] = nba_season(df['GAME_DATE'])
    df['home_team'] = df['game_id'].astype(str).str[-3:]
    df['is_home'] = (df['team'] == df['home_team']).astype(int)

    # Opponent = the other team sharing this game_id (games with != 2 teams are dropped)
    teams_per_game = df.groupby('game_id')['team'].unique()
    valid_games = teams_per_game[teams_per_game.apply(len) == 2]
    opponent_lookup = {
        game_id: {teams[0]: teams[1], teams[1]: teams[0]}
        for game_id, teams in valid_games.items()
    }
    df = df[df['game_id'].isin(opponent_lookup)].copy()
    df['opponent'] = [opponent_lookup[g][t] for g, t in zip(df['game_id'], df['team'])]

    if positions:
        clean_positions = {str(k).strip(): str(v).strip() for k, v in positions.items()}
        df['position'] = df['PLAYER_NAME'].astype(str).str.strip().map(clean_positions).fillna('UNK')
    else:
        df['position'] = 'UNK'

    return df.sort_values(['GAME_DATE', 'game_id']).reset_index(drop=True)


def _add_opponent_defense_features(df):
    """
    Point-in-time opponent defense, computed from the game logs themselves.

    For a defending team, "points allowed" in a game is the total scored by
    the players it faced. Expanding means are shifted so a row only sees the
    opponent's games before the current one.
    """
    # Team level: total points a defending team allowed, per game. Expanding
    # means reset each season -- a team's defense two seasons ago says little
    # about its current roster.
    allowed = (
        df.groupby(['opponent', 'season', 'game_id', 'GAME_DATE'], as_index=False)['PTS']
        .sum()
        .rename(columns={'opponent': 'def_team', 'PTS': 'pts_allowed'})
        .sort_values('GAME_DATE')
    )
    allowed['opp_pts_allowed_pit'] = (
        allowed.groupby(['def_team', 'season'])['pts_allowed']
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    df = df.merge(
        allowed[['def_team', 'season', 'game_id', 'opp_pts_allowed_pit']],
        left_on=['opponent', 'season', 'game_id'],
        right_on=['def_team', 'season', 'game_id'], how='left',
    ).drop(columns='def_team')

    # Position level: points a defending team allowed to a given position, per game
    allowed_pos = (
        df.groupby(['opponent', 'season', 'position', 'game_id', 'GAME_DATE'], as_index=False)['PTS']
        .sum()
        .rename(columns={'opponent': 'def_team', 'PTS': 'pts_allowed_pos'})
        .sort_values('GAME_DATE')
    )
    allowed_pos['opp_pts_allowed_pos_pit'] = (
        allowed_pos.groupby(['def_team', 'season', 'position'])['pts_allowed_pos']
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    df = df.merge(
        allowed_pos[['def_team', 'season', 'position', 'game_id', 'opp_pts_allowed_pos_pit']],
        left_on=['opponent', 'season', 'position', 'game_id'],
        right_on=['def_team', 'season', 'position', 'game_id'], how='left',
    ).drop(columns='def_team')

    # A team's very first game has no prior defensive history; fall back to the
    # league-wide mean rather than dropping the row.
    for col in ['opp_pts_allowed_pit', 'opp_pts_allowed_pos_pit']:
        df[col] = df[col].fillna(df[col].mean())

    return df


def build_feature_dataset(gamelog_df, positions=None, min_prior_games=5):
    """
    Build a leakage-free, point-in-time feature matrix for points prediction.

    Returns a DataFrame with FEATURE_COLUMNS plus:
      player, team, opponent, game_date, actual_pts, baseline_pred, games_played

    Only rows where the player already has >= min_prior_games prior games are
    kept, so every feature is backed by a meaningful amount of history.
    `baseline_pred` is the naive season-average-to-date prediction, carried
    alongside so any evaluation can compare against it on identical rows.
    """
    df = prepare_gamelog(gamelog_df, positions=positions)
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS + ['player', 'actual_pts', 'baseline_pred'])

    # Player's share of team shot attempts in each game (usage proxy input)
    team_fga = df.groupby(['game_id', 'team'])['FGA'].transform('sum')
    df['fga_share'] = _safe_div(df['FGA'], team_fga)

    df = _add_opponent_defense_features(df)
    df = df.sort_values(_PLAYER_KEYS + ['GAME_DATE']).reset_index(drop=True)
    grouped = df.groupby(_PLAYER_KEYS, sort=False)

    # Multi-horizon rolling means over strictly-prior games
    df['pts_r3'] = grouped['PTS'].transform(lambda s: _shift_roll(s, 3))
    df['pts_r5'] = grouped['PTS'].transform(lambda s: _shift_roll(s, 5))
    df['pts_r10'] = grouped['PTS'].transform(lambda s: _shift_roll(s, 10))
    df['pts_std_r5'] = grouped['PTS'].transform(lambda s: _shift_roll(s, 5, 'std')).fillna(0.0)
    df['min_r3'] = grouped['MIN'].transform(lambda s: _shift_roll(s, 3))

    # Produces pts_r5, min_r5, fga_r5, fg3a_r5, fta_r5, reb_r5, ast_r5, tov_r5, stl_r5, blk_r5
    for col in _ROLL_COLS:
        df[f'{col.lower()}_r5'] = grouped[col].transform(lambda s: _shift_roll(s, 5))

    df['fga_share_r5'] = grouped['fga_share'].transform(lambda s: _shift_roll(s, 5))

    # Season-to-date (expanding) means -- pts_season is the naive baseline itself
    df['pts_season'] = grouped['PTS'].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df['min_season'] = grouped['MIN'].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    # Form / trend: recent vs. season-long
    df['momentum_3_vs_season'] = df['pts_r3'] - df['pts_season']
    df['min_trend'] = df['min_r3'] - df['min_season']

    # Efficiency and shot-profile ratios
    df['pts_per_min_r5'] = _safe_div(df['pts_r5'], df['min_r5'])
    df['pts_per_fga_r5'] = _safe_div(df['pts_r5'], df['fga_r5'])
    df['fga_per_min_r5'] = _safe_div(df['fga_r5'], df['min_r5'])
    df['fg3a_rate_r5'] = _safe_div(df['fg3a_r5'], df['fga_r5'])
    df['fta_rate_r5'] = _safe_div(df['fta_r5'], df['fga_r5'])

    # Schedule context
    df['rest_days'] = (
        grouped['GAME_DATE'].transform(lambda s: s.diff().dt.days).fillna(7).clip(upper=7)
    )
    df['is_b2b'] = (df['rest_days'] <= 1).astype(int)
    df['games_played'] = grouped.cumcount()

    df['actual_pts'] = df['PTS'].astype(float)
    df['baseline_pred'] = df['pts_season']
    df = df.rename(columns={'PLAYER_NAME': 'player', 'GAME_DATE': 'game_date'})

    df = df[df['games_played'] >= min_prior_games].copy()

    keep = FEATURE_COLUMNS + ['player', 'season', 'team', 'opponent', 'game_date',
                              'actual_pts', 'baseline_pred']
    result = df[keep].reset_index(drop=True)
    return result.replace([np.inf, -np.inf], 0.0).fillna(0.0)
