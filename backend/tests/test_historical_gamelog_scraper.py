"""
Tests for backend/historical_gamelog_scraper.py.

All tests are offline: HTML is built inline rather than fetched, so the suite
never touches Basketball Reference.
"""
import sys
import os
import json
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from historical_gamelog_scraper import (
    SCHEMA,
    _load_checkpoint,
    _save_checkpoint,
    build_game_id,
    format_game_date,
    merge_into_cache,
    parse_gamelog_html,
    parse_minutes,
)


def make_row_html(date='2025-10-22', team='MIA', opp='ORL', location='@', mp='34:12', pts='15'):
    """One <tr> in Basketball Reference's player game log shape."""
    return f"""
    <tr>
      <th data-stat="ranker">1</th>
      <td data-stat="date">{date}</td>
      <td data-stat="team_name_abbr">{team}</td>
      <td data-stat="game_location">{location}</td>
      <td data-stat="opp_name_abbr">{opp}</td>
      <td data-stat="mp">{mp}</td>
      <td data-stat="fg">4</td><td data-stat="fga">13</td><td data-stat="fg_pct">.308</td>
      <td data-stat="fg3">1</td><td data-stat="fg3a">5</td><td data-stat="fg3_pct">.200</td>
      <td data-stat="ft">6</td><td data-stat="fta">7</td><td data-stat="ft_pct">.857</td>
      <td data-stat="orb">3</td><td data-stat="drb">9</td><td data-stat="trb">12</td>
      <td data-stat="ast">2</td><td data-stat="stl">0</td><td data-stat="blk">0</td>
      <td data-stat="tov">4</td><td data-stat="pf">3</td><td data-stat="pts">{pts}</td>
    </tr>"""


def make_table_html(rows_html, table_id='player_game_log_reg'):
    return f'<html><body><table id="{table_id}"><tbody>{rows_html}</tbody></table></body></html>'


# Field parsing

class TestParseMinutes:
    def test_converts_mm_ss_to_decimal(self):
        assert parse_minutes('34:12') == 34.2
        assert parse_minutes('28:24') == 28.4

    def test_zero_minutes(self):
        assert parse_minutes('0:00') == 0.0

    def test_returns_none_for_unparseable(self):
        for value in ['', None, 'Did Not Play', 'Inactive', '34']:
            assert parse_minutes(value) is None


class TestFormatGameDate:
    def test_matches_existing_cache_format(self):
        assert format_game_date('2026-02-19') == 'Thu, Feb 19, 2026'

    def test_single_digit_day_is_not_zero_padded(self):
        # The existing cache stores 'Sun, Mar 1, 2026', not 'Mar 01'
        assert format_game_date('2026-03-01') == 'Sun, Mar 1, 2026'

    def test_roundtrips_through_pandas(self):
        formatted = format_game_date('2025-10-22')
        assert pd.to_datetime(formatted, format='mixed') == pd.Timestamp('2025-10-22')


class TestBuildGameId:
    def test_home_game_uses_players_team(self):
        assert build_game_id('2026-03-10', 'MIA', is_home=True, opponent='ORL') == '202603100MIA'

    def test_away_game_uses_opponent(self):
        assert build_game_id('2025-10-22', 'MIA', is_home=False, opponent='ORL') == '202510220ORL'

    def test_suffix_is_readable_as_home_team(self):
        """features.py recovers the home team from the last 3 chars; keep that true."""
        game_id = build_game_id('2026-01-05', 'LAL', is_home=False, opponent='BOS')
        assert game_id[-3:] == 'BOS'


# Game log parsing

class TestParseGamelogHtml:
    def test_parses_a_row_into_schema(self):
        html = make_table_html(make_row_html())
        rows = parse_gamelog_html(html, 'adebaba01', 'Bam Adebayo')
        assert len(rows) == 1
        assert set(rows[0]) == set(SCHEMA)

    def test_maps_stats_correctly(self):
        html = make_table_html(make_row_html(pts='15'))
        row = parse_gamelog_html(html, 'adebaba01', 'Bam Adebayo')[0]
        assert row['PLAYER_NAME'] == 'Bam Adebayo'
        assert row['Player_ID'] == 'adebaba01'
        assert row['PTS'] == 15
        assert row['FGM'] == 4 and row['FGA'] == 13
        assert row['REB'] == 12  # trb -> REB
        assert row['MIN'] == 34.2
        assert row['GAME_DATE'] == 'Wed, Oct 22, 2025'

    def test_away_game_id_uses_opponent(self):
        html = make_table_html(make_row_html(location='@', team='MIA', opp='ORL'))
        assert parse_gamelog_html(html, 'x', 'X')[0]['game_id'] == '202510220ORL'

    def test_home_game_id_uses_own_team(self):
        html = make_table_html(make_row_html(location='', team='MIA', opp='ORL'))
        assert parse_gamelog_html(html, 'x', 'X')[0]['game_id'] == '202510220MIA'

    def test_skips_dnp_rows(self):
        """Rows with no minutes are DNP/inactive and must not become 0-point games."""
        html = make_table_html(make_row_html() + make_row_html(date='2025-10-24', mp='Did Not Play'))
        rows = parse_gamelog_html(html, 'x', 'X')
        assert len(rows) == 1

    def test_skips_header_separator_rows(self):
        html = make_table_html('<tr class="thead"><th>Rk</th></tr>' + make_row_html())
        assert len(parse_gamelog_html(html, 'x', 'X')) == 1

    def test_skips_rows_missing_team_or_date(self):
        html = make_table_html(make_row_html(team='', opp=''))
        assert parse_gamelog_html(html, 'x', 'X') == []

    def test_missing_table_returns_empty(self):
        assert parse_gamelog_html('<html><body>no table</body></html>', 'x', 'X') == []

    def test_integer_columns_are_ints(self):
        row = parse_gamelog_html(make_table_html(make_row_html()), 'x', 'X')[0]
        for column in ['PTS', 'FGM', 'FGA', 'REB', 'AST']:
            assert isinstance(row[column], int)


# Checkpointing

class TestCheckpoint:
    def test_missing_checkpoint_is_empty(self, tmp_path):
        assert _load_checkpoint(str(tmp_path / 'nope.json')) == set()

    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / 'ckpt.json')
        _save_checkpoint(path, {'2026:abc', '2026:def'})
        assert _load_checkpoint(path) == {'2026:abc', '2026:def'}

    def test_checkpoint_records_timestamp(self, tmp_path):
        path = str(tmp_path / 'ckpt.json')
        _save_checkpoint(path, {'2026:abc'})
        with open(path) as handle:
            assert 'updated' in json.load(handle)


# Merging into the cache

def make_frame(player_id, game_id, pts, name='Player A'):
    row = {col: 0 for col in SCHEMA}
    row.update({'PLAYER_NAME': name, 'Player_ID': player_id, 'game_id': game_id,
                'PTS': pts, 'GAME_DATE': 'Thu, Feb 19, 2026', 'team': 'MIA', 'MIN': 30.0})
    return pd.DataFrame([row], columns=SCHEMA)


class TestMergeIntoCache:
    def test_combines_disjoint_rows(self, tmp_path):
        cache = tmp_path / 'cache.csv'
        backfill = tmp_path / 'back.csv'
        make_frame('p1', '202602190MIA', 10).to_csv(cache, index=False)
        make_frame('p2', '202602190MIA', 20).to_csv(backfill, index=False)

        merged = merge_into_cache(str(backfill), str(cache), str(tmp_path / 'out.csv'))
        assert len(merged) == 2

    def test_deduplicates_on_player_and_game(self, tmp_path):
        cache = tmp_path / 'cache.csv'
        backfill = tmp_path / 'back.csv'
        make_frame('p1', '202602190MIA', 10).to_csv(cache, index=False)
        make_frame('p1', '202602190MIA', 99).to_csv(backfill, index=False)

        merged = merge_into_cache(str(backfill), str(cache), str(tmp_path / 'out.csv'))
        assert len(merged) == 1
        # Existing cached value wins over the backfilled duplicate
        assert merged.iloc[0]['PTS'] == 10

    def test_preserves_schema_order(self, tmp_path):
        cache = tmp_path / 'cache.csv'
        backfill = tmp_path / 'back.csv'
        make_frame('p1', 'g1', 10).to_csv(cache, index=False)
        make_frame('p2', 'g2', 20).to_csv(backfill, index=False)

        merged = merge_into_cache(str(backfill), str(cache), str(tmp_path / 'out.csv'))
        assert list(merged.columns) == SCHEMA

    def test_works_with_no_existing_cache(self, tmp_path):
        backfill = tmp_path / 'back.csv'
        make_frame('p1', 'g1', 10).to_csv(backfill, index=False)
        merged = merge_into_cache(str(backfill), str(tmp_path / 'missing.csv'), str(tmp_path / 'out.csv'))
        assert len(merged) == 1

    def test_missing_backfill_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            merge_into_cache(str(tmp_path / 'nope.csv'), str(tmp_path / 'cache.csv'))
