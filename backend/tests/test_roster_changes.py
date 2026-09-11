"""
Tests for roster_changes.py -- the manual offseason input.

Guards the decisions in design doc section 10: BBRef team codes (not broadcast
abbreviations), role priors shrunk rather than applied at face value, coherent
scaling across categories, and players who will not play dropped rather than
projected as 0.0.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roster_changes import (  # noqa: E402
    COMMON_MISTAKES,
    ROLE_MULTIPLIERS,
    ROLE_SHRINK,
    TEAM_CHANGE_WIDENING,
    WIDENING_RETAINED,
    apply_to_projections,
    effective_multiplier,
    excluded_players,
    interval_multiplier,
    load_roster_changes,
    normalize_team,
    resolve_player_ids,
    validate,
)


@pytest.fixture
def projections():
    """A minimal projection frame shaped like fit_and_project's output."""
    players = ['Alpha Adams', 'Beta Brown', 'Gamma Green', 'Delta Davis']
    frame = pd.DataFrame({
        'player': players,
        'player_id': ['alphaad', 'betabro', 'gammagr', 'deltada'],
        'team_prev': ['LAL', 'BOS', 'MIA', 'NYK'],
        'position': ['PG', 'SG', 'C', 'PF'],
        'pred_min': [30.0, 24.0, 32.0, 20.0],
        'pred_pts': [20.0, 12.0, 18.0, 8.0],
        'pred_reb': [5.0, 3.0, 10.0, 6.0],
        'pred_ast': [6.0, 2.0, 3.0, 1.0],
        'pred_stl': [1.0, 0.8, 0.5, 0.4],
        'pred_blk': [0.4, 0.2, 1.8, 0.9],
        'pred_fg3m': [2.0, 1.5, 0.2, 0.5],
        'pred_tov': [2.5, 1.2, 2.0, 0.8],
        'pred_fgm': [7.0, 4.5, 7.0, 3.0],
        'pred_fga': [15.0, 10.0, 12.0, 6.5],
        'pred_ftm': [4.0, 2.0, 4.0, 1.5],
        'pred_fta': [5.0, 2.4, 6.0, 2.0],
        'team_changed': [0, 0, 0, 0],
        'flags': ['', '', '', ''],
    })
    for column in ['pred_pts', 'pred_min', 'pred_reb']:
        frame[f'{column}_low'] = frame[column] - 3.0
        frame[f'{column}_high'] = frame[column] + 3.0
    frame['pts_prev'] = frame['pred_pts']
    frame['gp_prev'] = 60.0
    frame['n_prior_seasons'] = 2
    return frame


@pytest.fixture
def gamelog():
    return pd.DataFrame({
        'PLAYER_NAME': ['Alpha Adams', 'Beta Brown', 'Gamma Green', 'Delta Davis'],
        'Player_ID': ['alphaad', 'betabro', 'gammagr', 'deltada'],
        'team': ['LAL', 'BOS', 'MIA', 'NYK'],
    })


def write_changes(tmp_path, rows):
    path = tmp_path / 'roster_changes_2027.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Team codes
# ---------------------------------------------------------------------------

def test_valid_bbref_team_codes_pass():
    for code in ['LAL', 'BRK', 'CHO', 'PHO', 'FA']:
        assert normalize_team(code) == code


def test_broadcast_abbreviations_are_corrected():
    """
    The data uses BBRef codes. BKN/CHA/PHX are what a user naturally types and
    are mapped with a warning rather than rejected.
    """
    for typed, expected in COMMON_MISTAKES.items():
        assert normalize_team(typed) == expected


def test_unknown_team_code_is_rejected_not_silently_dropped():
    from roster_changes import ValidationReport
    report = ValidationReport()
    assert normalize_team('XYZ', report, 'Alpha Adams') is None
    assert report.bad_teams == [('Alpha Adams', 'XYZ')]
    assert not report.ok


def test_validate_reports_bad_team(gamelog, tmp_path):
    path = write_changes(tmp_path, [
        {'player_name': 'Alpha Adams', 'new_team': 'XYZ', 'change_type': 'trade'}])
    changes = pd.read_csv(path)
    report = validate(changes, gamelog)
    assert not report.ok
    assert any(code == 'XYZ' for _, code in report.bad_teams)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_unmatched_name_is_reported_with_a_suggestion(gamelog, tmp_path):
    path = write_changes(tmp_path, [
        {'player_name': 'Alpha Adamss', 'new_team': 'BOS', 'change_type': 'trade'}])
    report = validate(pd.read_csv(path), gamelog)
    assert not report.ok
    name, suggestions = report.unmatched[0]
    assert name == 'Alpha Adamss'
    assert 'Alpha Adams' in suggestions


def test_duplicate_rows_are_reported_not_last_wins(gamelog, tmp_path):
    path = write_changes(tmp_path, [
        {'player_name': 'Alpha Adams', 'new_team': 'BOS', 'change_type': 'trade'},
        {'player_name': 'Alpha Adams', 'new_team': 'MIA', 'change_type': 'trade'}])
    report = validate(pd.read_csv(path), gamelog)
    assert 'Alpha Adams' in report.duplicates
    assert not report.ok


def test_invalid_change_type_and_role_are_reported(gamelog, tmp_path):
    path = write_changes(tmp_path, [
        {'player_name': 'Alpha Adams', 'new_team': 'BOS',
         'change_type': 'abducted', 'role_change': 'gigantic'}])
    report = validate(pd.read_csv(path), gamelog)
    assert report.bad_change_types and report.bad_roles
    assert not report.ok


def test_clean_file_validates(gamelog, tmp_path):
    path = write_changes(tmp_path, [
        {'player_name': 'Alpha Adams', 'new_team': 'BOS',
         'change_type': 'trade', 'role_change': 'larger'}])
    assert validate(pd.read_csv(path), gamelog).ok


def test_resolve_player_ids_fills_from_gamelog(gamelog, tmp_path):
    path = write_changes(tmp_path, [
        {'player_name': 'Alpha Adams', 'new_team': 'BOS', 'change_type': 'trade'}])
    resolved = resolve_player_ids(load_roster_changes(path), gamelog)
    assert resolved['player_id'].iat[0] == 'alphaad'


# ---------------------------------------------------------------------------
# Role multipliers
# ---------------------------------------------------------------------------

def test_role_multipliers_are_shrunk_toward_one():
    """
    The multipliers are unfitted priors and the measured average effect of a
    move is near zero, so applying them at face value would overstate what is
    known. Applied value must be 1 + SHRINK*(raw - 1).
    """
    for role, raw in ROLE_MULTIPLIERS.items():
        assert effective_multiplier(role) == pytest.approx(1 + ROLE_SHRINK * (raw - 1))
    assert effective_multiplier('much_larger') < ROLE_MULTIPLIERS['much_larger']
    assert effective_multiplier('same') == pytest.approx(1.0)


def test_unknown_role_leaves_point_estimate_unchanged(projections, gamelog, tmp_path):
    path = write_changes(tmp_path, [
        {'player_name': 'Alpha Adams', 'new_team': 'BOS',
         'change_type': 'trade', 'role_change': 'unknown'}])
    out = apply_to_projections(projections, load_roster_changes(path, gamelog))
    row = out[out['player'] == 'Alpha Adams'].iloc[0]
    assert row['pred_pts'] == pytest.approx(20.0)
    assert row['pred_min'] == pytest.approx(30.0)


def test_larger_role_raises_every_counting_category(projections, gamelog, tmp_path):
    """A role change must move all categories coherently, not points alone."""
    path = write_changes(tmp_path, [
        {'player_name': 'Alpha Adams', 'new_team': 'BOS',
         'change_type': 'trade', 'role_change': 'much_larger'}])
    out = apply_to_projections(projections, load_roster_changes(path, gamelog))
    before = projections[projections['player'] == 'Alpha Adams'].iloc[0]
    after = out[out['player'] == 'Alpha Adams'].iloc[0]

    scale = effective_multiplier('much_larger')
    assert after['pred_min'] == pytest.approx(before['pred_min'] * scale)
    for stat in ['pts', 'reb', 'ast', 'stl', 'blk', 'fg3m', 'tov', 'fgm', 'fga', 'ftm', 'fta']:
        assert after[f'pred_{stat}'] == pytest.approx(before[f'pred_{stat}'] * scale), stat


def test_role_change_preserves_per36_rates(projections, gamelog, tmp_path):
    path = write_changes(tmp_path, [
        {'player_name': 'Alpha Adams', 'new_team': 'BOS',
         'change_type': 'trade', 'role_change': 'smaller'}])
    out = apply_to_projections(projections, load_roster_changes(path, gamelog))
    before = projections[projections['player'] == 'Alpha Adams'].iloc[0]
    after = out[out['player'] == 'Alpha Adams'].iloc[0]
    assert (after['pred_pts'] / after['pred_min']) == pytest.approx(
        before['pred_pts'] / before['pred_min'])


def test_min_override_wins_over_role_change(projections, gamelog, tmp_path):
    path = write_changes(tmp_path, [
        {'player_name': 'Alpha Adams', 'new_team': 'BOS', 'change_type': 'trade',
         'role_change': 'much_larger', 'min_override': 15.0}])
    out = apply_to_projections(projections, load_roster_changes(path, gamelog))
    row = out[out['player'] == 'Alpha Adams'].iloc[0]
    assert row['pred_min'] == pytest.approx(15.0), "override used exactly, no multiplier"
    assert row['pred_pts'] == pytest.approx(20.0 * 15.0 / 30.0)


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------

def _width(projections, gamelog, tmp_path, stat, **change):
    row = {'player_name': 'Alpha Adams', 'new_team': 'BOS', 'change_type': 'trade'}
    row.update(change)
    path = write_changes(tmp_path, [row])
    out = apply_to_projections(projections, load_roster_changes(path, gamelog))
    after = out[out['player'] == 'Alpha Adams'].iloc[0]
    return after[f'pred_{stat}_high'] - after[f'pred_{stat}_low']


def test_unknown_role_widens_minutes_interval_by_the_measured_ratio(projections, gamelog, tmp_path):
    """
    Minutes are where a team change measurably adds uncertainty against model
    residuals (1.13x). An unknown role takes the full widening.
    """
    width = _width(projections, gamelog, tmp_path, 'min', role_change='unknown')
    assert width == pytest.approx(6.0 * TEAM_CHANGE_WIDENING['min'])


def test_known_role_and_override_recover_part_of_the_widening(projections, gamelog, tmp_path):
    unknown = _width(projections, gamelog, tmp_path, 'min', role_change='unknown')
    known = _width(projections, gamelog, tmp_path, 'min', role_change='same')
    override = _width(projections, gamelog, tmp_path, 'min', role_change='same', min_override=30.0)
    assert unknown > known > override > 6.0
    assert known == pytest.approx(6.0 * interval_multiplier('min', 'role_known'))
    assert override == pytest.approx(6.0 * interval_multiplier('min', 'min_override'))


def test_points_interval_is_not_widened_by_a_team_change(projections, gamelog, tmp_path):
    """
    Regression guard on the re-derivation. Against model residuals a mover's PTS
    band measured 0.97x a stayer's -- the old blanket 1.16 came from
    carry-forward MAE, where WHO moves (not the move) made movers look noisier.
    """
    assert 'pts' not in TEAM_CHANGE_WIDENING
    width = _width(projections, gamelog, tmp_path, 'pts', role_change='unknown')
    assert width == pytest.approx(6.0)


def test_widening_is_never_below_one():
    """A team change must never make a projection look MORE certain."""
    assert all(v >= 1.0 for v in TEAM_CHANGE_WIDENING.values())
    for case in WIDENING_RETAINED:
        for stat in ['pts', 'min', 'gp', 'reb', 'fta']:
            assert interval_multiplier(stat, case) >= 1.0


def test_players_without_a_row_are_untouched(projections, gamelog, tmp_path):
    path = write_changes(tmp_path, [
        {'player_name': 'Alpha Adams', 'new_team': 'BOS', 'change_type': 'trade'}])
    out = apply_to_projections(projections, load_roster_changes(path, gamelog))
    before = projections[projections['player'] == 'Gamma Green'].iloc[0]
    after = out[out['player'] == 'Gamma Green'].iloc[0]
    assert after['pred_pts'] == pytest.approx(before['pred_pts'])
    assert after['pred_pts_high'] - after['pred_pts_low'] == pytest.approx(6.0)
    assert after['team_changed'] == 0


# ---------------------------------------------------------------------------
# Status handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('status', ['out_for_season', 'retired', 'overseas'])
def test_players_who_will_not_play_are_dropped_not_zeroed(projections, gamelog,
                                                          tmp_path, status):
    """
    A projected 0.0 sorts to the bottom of the board and reads as a real
    prediction. Dropping with a reason code is the honest representation.
    """
    path = write_changes(tmp_path, [
        {'player_name': 'Beta Brown', 'new_team': 'BOS',
         'change_type': 'waived', 'status': status}])
    out = apply_to_projections(projections, load_roster_changes(path, gamelog))
    assert 'Beta Brown' not in set(out['player'])
    assert not (out['pred_pts'] == 0.0).any()
    assert len(out) == len(projections) - 1


# ---------------------------------------------------------------------------
# Missing / empty file
# ---------------------------------------------------------------------------

def test_missing_file_warns_and_returns_empty(tmp_path, capsys):
    changes = load_roster_changes(tmp_path / 'does_not_exist.csv')
    assert changes.empty
    assert 'no file at' in capsys.readouterr().out


def test_missing_file_leaves_projections_unadjusted(projections, tmp_path):
    out = apply_to_projections(projections, load_roster_changes(tmp_path / 'nope.csv'))
    pd.testing.assert_frame_equal(out, projections.reset_index(drop=True))


def test_missing_required_column_raises(tmp_path):
    path = tmp_path / 'bad.csv'
    pd.DataFrame({'player_name': ['Alpha Adams']}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='new_team'):
        load_roster_changes(path)


def test_partial_file_is_valid(projections, gamelog, tmp_path):
    """
    An incomplete file must degrade gracefully -- every filled row helps one
    player, skipped rows fall back to model-only behaviour, no cliff.
    """
    path = write_changes(tmp_path, [
        {'player_name': 'Alpha Adams', 'new_team': 'BOS', 'change_type': 'trade'}])
    out = apply_to_projections(projections, load_roster_changes(path, gamelog))
    assert len(out) == len(projections)
    assert out['pred_pts'].notna().all()


# ---------------------------------------------------------------------------
# Rookies, exclusions, role_change column
# ---------------------------------------------------------------------------

def test_drafted_rookie_is_emitted_as_zero_row_flagged_no_prior_season(projections, gamelog, tmp_path):
    """Section 10.8: rookies appear on the board, at 0.0, flagged so value is NaN."""
    path = write_changes(tmp_path, [
        {'player_name': 'Rookie Ray', 'new_team': 'SAS', 'change_type': 'draft'}])
    out = apply_to_projections(projections, load_roster_changes(path, gamelog))
    assert len(out) == len(projections) + 1
    rookie = out[out['player'] == 'Rookie Ray'].iloc[0]
    assert rookie['flags'] == 'NO_PRIOR_SEASON'
    assert rookie['n_prior_seasons'] == 0
    assert rookie['team_prev'] == 'SAS'
    for column in [c for c in out.columns if c.startswith('pred_')] + ['pts_prev', 'gp_prev']:
        assert rookie[column] == 0.0, column


def test_rookie_value_is_nan_and_real_players_are_unaffected(projections, gamelog, tmp_path):
    """End to end: the zero row must not reach the z-score pool."""
    from train_season_model import attach_fantasy_value, derive_ratio_projections
    base = derive_ratio_projections(projections)
    path = write_changes(tmp_path, [
        {'player_name': 'Rookie Ray', 'new_team': 'SAS', 'change_type': 'draft'}])
    with_rookie = derive_ratio_projections(
        apply_to_projections(projections, load_roster_changes(path, gamelog)))
    # derive_ratio_projections gives the rookie the league rate; §10.8 wants 0.0
    with_rookie.loc[with_rookie['flags'] == 'NO_PRIOR_SEASON', ['pred_fg_pct', 'pred_ft_pct']] = 0.0

    before = attach_fantasy_value(base, pool_size=None)
    after = attach_fantasy_value(with_rookie, pool_size=None)
    assert np.isnan(after.loc[after['player'] == 'Rookie Ray', 'value_total']).all()
    real = after[after['player'] != 'Rookie Ray'].reset_index(drop=True)
    # Float summation order differs by ~1e-16 with an extra row; the
    # bit-identical guarantee is asserted directly in test_fantasy_value.py.
    # Without the pool exclusion the shift would be O(1), not O(1e-16).
    np.testing.assert_allclose(real['value_total'].to_numpy(),
                               before['value_total'].to_numpy(), rtol=0, atol=1e-12)


def test_draft_row_for_an_already_projected_player_adds_no_duplicate(projections, gamelog, tmp_path):
    path = write_changes(tmp_path, [
        {'player_name': 'Alpha Adams', 'new_team': 'BOS', 'change_type': 'draft'}])
    out = apply_to_projections(projections, load_roster_changes(path, gamelog))
    assert out['player'].is_unique
    assert len(out) == len(projections)


def test_injured_rookie_is_not_emitted(projections, gamelog, tmp_path):
    path = write_changes(tmp_path, [
        {'player_name': 'Rookie Ray', 'new_team': 'SAS', 'change_type': 'draft',
         'status': 'out_for_season'}])
    out = apply_to_projections(projections, load_roster_changes(path, gamelog))
    assert 'Rookie Ray' not in set(out['player'])


def test_validate_does_not_flag_a_drafted_rookie_as_unmatched(gamelog, tmp_path):
    """Rookies have no game log yet; that absence is expected, not a typo."""
    path = write_changes(tmp_path, [
        {'player_name': 'Rookie Ray', 'new_team': 'SAS', 'change_type': 'draft'}])
    report = validate(pd.read_csv(path), gamelog)
    assert report.ok
    assert not report.unmatched


def test_validate_still_flags_unmatched_veterans(gamelog, tmp_path):
    path = write_changes(tmp_path, [
        {'player_name': 'Nobody Known', 'new_team': 'SAS', 'change_type': 'trade'}])
    assert not validate(pd.read_csv(path), gamelog).ok


def test_excluded_players_carry_a_reason_code(gamelog, tmp_path):
    """Section 13.5: every player dropped from the board has a recorded reason."""
    path = write_changes(tmp_path, [
        {'player_name': 'Beta Brown', 'new_team': 'BOS', 'change_type': 'waived',
         'status': 'out_for_season'},
        {'player_name': 'Gamma Green', 'new_team': 'FA', 'change_type': 'retired',
         'status': 'retired'},
        {'player_name': 'Alpha Adams', 'new_team': 'BOS', 'change_type': 'trade'}])
    excluded = excluded_players(load_roster_changes(path, gamelog))
    assert dict(zip(excluded['player'], excluded['reason'])) == {
        'Beta Brown': 'OUT_FOR_SEASON', 'Gamma Green': 'RETIRED'}


def test_excluded_players_of_an_empty_file_is_empty(tmp_path):
    excluded = excluded_players(load_roster_changes(tmp_path / 'nope.csv', warn=False))
    assert excluded.empty
    assert list(excluded.columns) == ['player', 'player_id', 'reason']


def test_role_change_is_recorded_on_the_board(projections, gamelog, tmp_path):
    path = write_changes(tmp_path, [
        {'player_name': 'Alpha Adams', 'new_team': 'BOS', 'change_type': 'trade',
         'role_change': 'larger'}])
    out = apply_to_projections(projections, load_roster_changes(path, gamelog))
    assert out.loc[out['player'] == 'Alpha Adams', 'role_change'].iat[0] == 'larger'
    assert out.loc[out['player'] == 'Gamma Green', 'role_change'].iat[0] == ''


# ---------------------------------------------------------------------------
# The real 2026-27 file (backend/data/roster_changes_2027.csv)
# ---------------------------------------------------------------------------
#
# These guard the shipped data file itself, not the loader. It is hand-editable
# by design, so a typo here would otherwise surface as a silently missing
# projection rather than an error.

@pytest.fixture(scope='module')
def real_roster_file():
    from roster_changes import ROSTER_CHANGES_PATH
    if not os.path.exists(ROSTER_CHANGES_PATH):
        pytest.skip('roster_changes_2027.csv not present')
    frame = pd.read_csv(ROSTER_CHANGES_PATH)
    if frame.empty:
        pytest.skip('roster_changes_2027.csv is still an empty template')
    return frame


@pytest.fixture(scope='module')
def real_gamelog():
    from season_features import GAMELOG_PATH
    if not os.path.exists(GAMELOG_PATH):
        pytest.skip('game logs not present')
    return pd.read_csv(GAMELOG_PATH)


def test_real_roster_file_has_the_documented_schema(real_roster_file):
    from roster_changes import OPTIONAL_COLUMNS, REQUIRED_COLUMNS

    assert list(real_roster_file.columns) == [
        'player_name', 'player_id', 'new_team', 'change_type', 'role_change',
        'min_override', 'status', 'note', 'source', 'updated']
    for column in REQUIRED_COLUMNS:
        assert real_roster_file[column].notna().all(), column
    assert set(OPTIONAL_COLUMNS) <= set(real_roster_file.columns)


def test_real_roster_file_uses_valid_codes_and_types(real_roster_file):
    from roster_changes import (CHANGE_TYPES, FREE_AGENT, ROLE_MULTIPLIERS,
                                STATUSES, VALID_TEAMS)

    assert set(real_roster_file['new_team']) <= VALID_TEAMS | {FREE_AGENT}
    assert set(real_roster_file['change_type']) <= CHANGE_TYPES
    assert set(real_roster_file['status'].dropna()) <= STATUSES
    assert set(real_roster_file['role_change'].dropna()) <= set(ROLE_MULTIPLIERS)


def test_real_roster_file_has_no_duplicate_players(real_roster_file):
    duplicated = real_roster_file['player_name'][
        real_roster_file['player_name'].duplicated()]
    assert list(duplicated) == []


def test_real_roster_file_validates_against_the_game_logs(real_roster_file,
                                                          real_gamelog):
    """Every name resolves, so no row can silently fail to reach a projection."""
    from roster_changes import ROSTER_CHANGES_PATH, load_roster_changes, validate

    changes = load_roster_changes(ROSTER_CHANGES_PATH, gamelog_df=real_gamelog)
    report = validate(changes, real_gamelog)
    assert report.ok, str(report)
    assert changes['player_id'].notna().all()


def test_real_roster_file_only_lists_players_who_changed_something(
        real_roster_file, real_gamelog):
    """
    Section 10.3: absence means 'no change'. A row whose new_team equals the
    player's 2025-26 team does nothing but widen that player's interval for no
    reason, so it is a defect rather than a harmless extra.
    """
    from roster_changes import DROP_STATUSES
    from season_features import aggregate_player_seasons

    seasons = aggregate_player_seasons(real_gamelog)
    latest = seasons[seasons['season'] == seasons['season'].max()]
    prior_team = dict(zip(latest['player'], latest['team']))

    stayers = []
    for row in real_roster_file.itertuples():
        if str(row.status) in DROP_STATUSES or row.change_type in DROP_STATUSES:
            continue                      # a drop is a change even without a move
        if str(row.role_change) not in ('unknown', 'nan', ''):
            continue                      # an explicit role hint is a change
        previous = prior_team.get(row.player_name)
        if previous is not None and previous == row.new_team:
            stayers.append(row.player_name)
    assert stayers == []
