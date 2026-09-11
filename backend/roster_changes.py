"""
Offseason roster changes: the one manual input the projection model needs.

The model cannot see the offseason. Historical team changes ARE observable --
28.5% of paired player-seasons changed teams, derived straight from the game
logs' `team` column -- so `team_changed` is trained and validated with no manual
input. Only the inference season needs a hand-maintained file, because it has
not been played yet.

What a team change actually does (2,502 paired player-seasons, 7 transitions):

    group     n      dMIN mean/sd   dPTS mean/sd   carry-forward PTS MAE
    stayed    1644   +0.09 / 5.05   +0.24 / 3.06   2.388
    changed    858   -1.17 / 5.69   -0.73 / 3.21   2.610

Changing teams barely moves the AVERAGE outcome (small against its spread), and
once the model's own features are accounted for it barely widens it either:
against model residuals only minutes, games and rebounds get measurably less
predictable (TEAM_CHANGE_WIDENING). The default effect of this file is a modest
interval widening on those categories. It shifts the point estimate only where
the user supplies a role hint -- the knowledge the model lacks and they have.

Consequence: an incomplete file is fine. Every filled row improves one player's
interval; skipped rows fall back to model-only behaviour. There is no cliff.
"""
import difflib
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
ROSTER_CHANGES_PATH = os.path.join(DATA_DIR, 'roster_changes_2027.csv')

# Basketball Reference team codes -- NOT the broadcast abbreviations. Three
# differ from what a user would naturally type, so they are mapped with a
# warning rather than rejected.
VALID_TEAMS = {
    'ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
    'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
    'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS',
}
FREE_AGENT = 'FA'
COMMON_MISTAKES = {'BKN': 'BRK', 'CHA': 'CHO', 'PHX': 'PHO'}

CHANGE_TYPES = {'trade', 'free_agency', 're_signed', 'waived', 'retired',
                'overseas', 'draft'}
STATUSES = {'active', 'out_for_season', 'retired', 'overseas'}
DROP_STATUSES = {'out_for_season', 'retired', 'overseas'}

# Raw minutes multipliers by role hint. HAND-SET PRIORS, not fitted values:
# there is no labelled history of role changes to fit them against, and the
# measured average effect of a move is near zero, so they are shrunk toward 1.0
# rather than applied at face value.
ROLE_MULTIPLIERS = {
    'much_larger': 1.20, 'larger': 1.10, 'same': 1.00,
    'smaller': 0.90, 'much_smaller': 0.80, 'unknown': 1.00,
}
ROLE_SHRINK = 0.5

# Interval widening for a team change, PER CATEGORY. Measured over 2,502 paired
# player-seasons (7 transitions) as the ratio of out-of-fold MODEL residual 80%
# band widths, movers vs. stayers. Only categories whose bootstrap 90% CI
# excludes 1.0 widen; the rest are 1.0. Floored at 1.0: FTM/FTA measured
# narrower (~0.90) but a team change is never a reason to claim MORE certainty.
#
# This replaces an earlier single 1.16 taken from carry-forward MAE. Most of that
# gap was WHO moves (fringe, older, volatile players), which the model's own
# features already capture -- against model residuals PTS widens by 0.97x, i.e.
# not at all. What remains is minutes and games, where a role change lands.
TEAM_CHANGE_WIDENING = {'min': 1.13, 'gp': 1.16, 'reb': 1.09}

# Share of that widening that remains once the user supplies more information.
# Interpolations, not measurements -- there is no labelled history of role hints.
WIDENING_RETAINED = {
    'unchanged': 0.0,
    'min_override': 0.3,
    'role_known': 0.6,
    'role_unknown': 1.0,
}

REQUIRED_COLUMNS = ['player_name', 'new_team', 'change_type']
OPTIONAL_COLUMNS = ['player_id', 'role_change', 'min_override', 'status',
                    'note', 'source', 'updated']

# Counting stats scaled through per-36 rates when minutes change, so a role
# change moves every category coherently rather than points alone.
_SCALED_STATS = ['pts', 'reb', 'ast', 'stl', 'blk', 'fg3m', 'tov',
                 'fgm', 'fga', 'ftm', 'fta']


class ValidationReport:
    """Collected problems with a roster-changes file."""

    def __init__(self):
        self.unmatched = []       # (name, [suggestions])
        self.bad_teams = []       # (name, code)
        self.corrected_teams = []  # (name, typed, corrected)
        self.bad_change_types = []
        self.bad_roles = []
        self.duplicates = []
        self.missing_rows = []    # players who changed teams in the logs but have no row

    @property
    def ok(self):
        return not (self.unmatched or self.bad_teams or self.bad_change_types
                    or self.bad_roles or self.duplicates)

    def __str__(self):
        lines = []
        for name, suggestions in self.unmatched:
            hint = f" -- did you mean {', '.join(suggestions)}?" if suggestions else ""
            lines.append(f"  UNMATCHED  {name}{hint}")
        for name, code in self.bad_teams:
            lines.append(f"  BAD TEAM   {name}: '{code}' is not a valid team code")
        for name, typed, fixed in self.corrected_teams:
            lines.append(f"  corrected  {name}: '{typed}' -> '{fixed}'")
        for name, value in self.bad_change_types:
            lines.append(f"  BAD TYPE   {name}: '{value}' not in {sorted(CHANGE_TYPES)}")
        for name, value in self.bad_roles:
            lines.append(f"  BAD ROLE   {name}: '{value}' not in {sorted(ROLE_MULTIPLIERS)}")
        for name in self.duplicates:
            lines.append(f"  DUPLICATE  {name}: appears more than once")
        if self.missing_rows:
            lines.append(f"  note: {len(self.missing_rows)} players have no row "
                         f"(fine -- they fall back to model-only behaviour)")
        return "\n".join(lines) if lines else "  no problems found"


def _known_players(gamelog_df):
    """name -> player_id for everyone in the game logs."""
    if gamelog_df is None or gamelog_df.empty:
        return {}
    names = gamelog_df['PLAYER_NAME'].astype(str).str.strip()
    if 'Player_ID' in gamelog_df.columns:
        return dict(zip(names, gamelog_df['Player_ID']))
    return {name: None for name in names.unique()}


def normalize_team(code, report=None, player_name=''):
    """Map a typed team code to a Basketball Reference code, or None if invalid."""
    text = str(code).strip().upper()
    if text in VALID_TEAMS or text == FREE_AGENT:
        return text
    if text in COMMON_MISTAKES:
        fixed = COMMON_MISTAKES[text]
        if report is not None:
            report.corrected_teams.append((player_name, text, fixed))
        return fixed
    if report is not None:
        report.bad_teams.append((player_name, text))
    return None


def load_roster_changes(path=ROSTER_CHANGES_PATH, gamelog_df=None, warn=True):
    """
    Read the roster-changes file, normalizing teams and resolving player ids.

    A missing file is a WARNING, not an error: the average effect of a move is
    near zero, so an unmaintained file costs interval width, not correctness.
    """
    if not os.path.exists(path):
        if warn:
            print(f"  [roster_changes] no file at {path} -- projecting without "
                  f"offseason adjustments")
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"roster changes file is missing required columns: {missing}")

    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    df['player_name'] = df['player_name'].astype(str).str.strip()
    df['role_change'] = df['role_change'].fillna('unknown').astype(str).str.strip().str.lower()
    df['status'] = df['status'].fillna('active').astype(str).str.strip().str.lower()
    df['change_type'] = df['change_type'].astype(str).str.strip().str.lower()
    df['min_override'] = pd.to_numeric(df['min_override'], errors='coerce')

    report = ValidationReport()
    df['new_team'] = [normalize_team(t, report, n)
                      for t, n in zip(df['new_team'], df['player_name'])]
    if warn and report.corrected_teams:
        print(str(report))

    if gamelog_df is not None:
        df = resolve_player_ids(df, gamelog_df)
    return df


def resolve_player_ids(changes_df, gamelog_df):
    """Fill `player_id` by exact name match against the game logs."""
    known = _known_players(gamelog_df)
    df = changes_df.copy()
    df['player_id'] = [
        known.get(name, existing if pd.notna(existing) else None)
        for name, existing in zip(df['player_name'], df.get('player_id', [None] * len(df)))
    ]
    return df


def validate(changes_df, gamelog_df=None):
    """
    Check a roster-changes file before it reaches the model.

    Nothing is applied until this passes, so a typo can never silently become a
    missing projection.
    """
    report = ValidationReport()
    if changes_df is None or changes_df.empty:
        return report

    known = _known_players(gamelog_df)
    known_names = list(known)

    seen = set()
    for row in changes_df.itertuples():
        name = str(row.player_name).strip()
        if name in seen:
            report.duplicates.append(name)
        seen.add(name)

        change_type = str(getattr(row, 'change_type', '') or '').strip().lower()
        # A drafted rookie has no NBA game log yet, so absence is expected there
        if known_names and name not in known and change_type != 'draft':
            suggestions = difflib.get_close_matches(name, known_names, n=3, cutoff=0.75)
            report.unmatched.append((name, suggestions))

        team = str(getattr(row, 'new_team', '') or '').strip().upper()
        if team and team not in VALID_TEAMS and team != FREE_AGENT:
            if team in COMMON_MISTAKES:
                report.corrected_teams.append((name, team, COMMON_MISTAKES[team]))
            else:
                report.bad_teams.append((name, team))

        if change_type and change_type not in CHANGE_TYPES:
            report.bad_change_types.append((name, change_type))

        role = str(getattr(row, 'role_change', 'unknown') or 'unknown').strip().lower()
        if role and role not in ROLE_MULTIPLIERS:
            report.bad_roles.append((name, role))

    # Players who changed teams in the logs but have no row are fine -- noted only
    if gamelog_df is not None and 'team' in gamelog_df.columns:
        report.missing_rows = [n for n in known_names if n not in seen]

    return report


def effective_multiplier(role_change):
    """
    Shrink the hand-set role prior toward 1.0.

    These multipliers are unfitted priors and the measured average effect of a
    move is near zero, so applying them at face value would overstate what is
    actually known. `1 + SHRINK * (raw - 1)`.
    """
    raw = ROLE_MULTIPLIERS.get(str(role_change).strip().lower(), 1.0)
    return 1.0 + ROLE_SHRINK * (raw - 1.0)


def apply_to_projections(projections, changes_df, warn=True):
    """
    Apply offseason changes to a projection frame.

    Minutes are scaled by the shrunk role multiplier (or set outright by
    `min_override`), and every counting stat is carried along through its
    per-36 rate so a role change moves all categories coherently rather than
    points alone. Intervals widen per TEAM_CHANGE_WIDENING. Players who will not
    play are dropped rather than projected as 0.0, which would sort to the
    bottom and read as a real prediction -- `excluded_players` gives each one a
    reason code.

    `draft` rows naming a player with no projection are appended as rookies:
    0.0 in every stat, flagged NO_PRIOR_SEASON so fantasy_value keeps them out
    of the z-score pool and gives them NaN value (design doc section 10.8).
    """
    df = projections.copy().reset_index(drop=True)
    if changes_df is None or len(changes_df) == 0:
        return df
    if 'role_change' not in df.columns:
        df['role_change'] = ''

    changes = changes_df.drop_duplicates('player_name', keep='last')
    by_name = changes.set_index('player_name')

    for i, row in df.iterrows():
        name = row['player']
        if name not in by_name.index:
            continue
        change = by_name.loc[name]

        status = str(change.get('status', 'active') or 'active').strip().lower()
        if status in DROP_STATUSES:
            df.at[i, 'flags'] = _add_flag(row.get('flags', ''), status.upper())
            continue

        new_team = change.get('new_team')
        if isinstance(new_team, str) and new_team:
            df.at[i, 'team_prev'] = new_team
        df.at[i, 'team_changed'] = 1

        old_minutes = float(row.get('pred_min', 0.0) or 0.0)
        override = change.get('min_override')
        role = str(change.get('role_change', 'unknown') or 'unknown').strip().lower()
        df.at[i, 'role_change'] = role

        if pd.notna(override) and float(override) > 0:
            new_minutes = float(override)
            interval_case = 'min_override'
            df.at[i, 'flags'] = _add_flag(row.get('flags', ''), 'MIN_OVERRIDE')
        else:
            new_minutes = old_minutes * effective_multiplier(role)
            interval_case = 'role_unknown' if role == 'unknown' else 'role_known'

        if old_minutes > 0 and new_minutes != old_minutes:
            scale = new_minutes / old_minutes
            df.at[i, 'pred_min'] = new_minutes
            for stat in _SCALED_STATS:
                column = f'pred_{stat}'
                if column in df.columns:
                    df.at[i, column] = float(row[column]) * scale

        _widen_interval(df, i, interval_case)
        df.at[i, 'flags'] = _add_flag(df.at[i, 'flags'], 'ROSTER_CHANGE')

    dropped = df['flags'].astype(str).str.contains('|'.join(s.upper() for s in DROP_STATUSES))
    if warn and dropped.any():
        print(f"  [roster_changes] dropped {int(dropped.sum())} players who will not play "
              f"(see excluded_players for reason codes)")
    df = df[~dropped].reset_index(drop=True)

    rookies = _rookie_rows(df, changes)
    if not rookies.empty:
        df = pd.concat([df, rookies], ignore_index=True)
    return df


def _rookie_rows(board, changes):
    """Zero-stat NO_PRIOR_SEASON rows for active `draft` entries not already projected."""
    known = set(board['player'])
    drafted = changes[(changes['change_type'].astype(str).str.lower() == 'draft')
                      & ~changes['player_name'].isin(known)]
    if 'status' in drafted.columns:
        drafted = drafted[~drafted['status'].astype(str).str.lower().isin(DROP_STATUSES)]
    if drafted.empty:
        return pd.DataFrame(columns=board.columns)

    rows = pd.DataFrame(index=range(len(drafted)), columns=board.columns)
    for column in board.columns:
        if column.startswith('pred_') or (column.endswith('_prev') and column != 'team_prev'):
            rows[column] = 0.0
    rows['player'] = drafted['player_name'].to_numpy()
    rows['player_id'] = drafted['player_id'].to_numpy() if 'player_id' in drafted else None
    rows['team_prev'] = drafted['new_team'].to_numpy()
    rows['position'] = 'UNK'
    rows['n_prior_seasons'] = 0
    rows['team_changed'] = 0
    rows['role_change'] = drafted['role_change'].fillna('unknown').to_numpy() \
        if 'role_change' in drafted else 'unknown'
    rows['flags'] = 'NO_PRIOR_SEASON'
    return rows


def excluded_players(changes_df):
    """
    Players dropped from the board, each with a reason code.

    The board omits them rather than projecting 0.0; this is the record of why,
    so an app can say "out for season" instead of silently missing a star.
    """
    columns = ['player', 'player_id', 'reason']
    if changes_df is None or len(changes_df) == 0 or 'status' not in changes_df.columns:
        return pd.DataFrame(columns=columns)
    status = changes_df['status'].fillna('active').astype(str).str.strip().str.lower()
    out = changes_df[status.isin(DROP_STATUSES)].drop_duplicates('player_name', keep='last')
    return pd.DataFrame({
        'player': out['player_name'].to_numpy(),
        'player_id': out['player_id'].to_numpy() if 'player_id' in out else None,
        'reason': status[out.index].str.upper().to_numpy(),
    }, columns=columns)


def _add_flag(existing, flag):
    text = str(existing or '').strip()
    if not text:
        return flag
    return text if flag in text else f"{text};{flag}"


def interval_multiplier(stat, case):
    """Interval widening for one category: 1 + retained share of its measured excess."""
    excess = TEAM_CHANGE_WIDENING.get(stat, 1.0) - 1.0
    return 1.0 + WIDENING_RETAINED[case] * excess


def _widen_interval(df, i, case):
    """Scale a row's intervals about their point estimates, per category."""
    for column in df.columns:
        if not (column.startswith('pred_') and column.endswith('_low')):
            continue
        base = column[:-4]
        high_col = f'{base}_high'
        if base not in df.columns or high_col not in df.columns:
            continue
        multiplier = interval_multiplier(base[len('pred_'):], case)
        if multiplier == 1.0:
            continue
        point = float(df.at[i, base])
        df.at[i, column] = point + (float(df.at[i, column]) - point) * multiplier
        df.at[i, high_col] = point + (float(df.at[i, high_col]) - point) * multiplier


def main():
    """`python -m backend.roster_changes --validate`"""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', action='store_true', help='check the file and exit')
    parser.add_argument('--path', default=ROSTER_CHANGES_PATH)
    parser.add_argument('--gamelog', default=os.path.join(DATA_DIR, 'historical_player_gamelogs.csv'))
    args = parser.parse_args()

    gamelog_df = pd.read_csv(args.gamelog) if os.path.exists(args.gamelog) else None
    changes = load_roster_changes(args.path, gamelog_df=gamelog_df)
    if changes.empty:
        print("no roster changes to validate")
        return 0

    changes = resolve_player_ids(changes, gamelog_df) if gamelog_df is not None else changes
    report = validate(changes, gamelog_df)
    print(f"roster changes: {len(changes)} rows")
    print(report)
    if report.ok:
        changes.to_csv(args.path, index=False)
        print("\nOK -- player_id filled in and written back")
        return 0
    print("\nFAILED -- fix the problems above; nothing was written")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
