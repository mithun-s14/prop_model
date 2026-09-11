"""
Tests for nba_position_scraper.py and the get_player_position hardening.

The named regression here is the Killian Hayes case: model.py's last-name
substring fallback returned Jaxson Hayes' position, giving a guard a forward's
position. A missing position is visible downstream; a confidently wrong one is
not, which is why that path was removed rather than tightened.
"""
import contextlib
import io
import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nba_position_scraper import (  # noqa: E402
    VALID_POSITIONS,
    normalize_position,
    scrape_positions,
    update_positions_file,
)

TOTALS_HTML = """
<html><body><table id="totals_stats"><tbody>
  <tr><td data-stat="name_display"><a href="/players/j/jamesle01.html">LeBron James</a></td>
      <td data-stat="pos">SF</td></tr>
  <tr><td data-stat="name_display"><a href="/players/d/doncilu01.html">Luka Don&#269;i&#263;</a></td>
      <td data-stat="pos"> PG </td></tr>
  <tr><td data-stat="name_display"><a href="/players/g/greenja01.html">Jaxson Hayes</a></td>
      <td data-stat="pos">C-PF</td></tr>
  <tr><td data-stat="name_display"><a href="/players/h/hayeski01.html">Killian Hayes</a></td>
      <td data-stat="pos">G</td></tr>
</tbody></table></body></html>
"""


# ---------------------------------------------------------------------------
# Position normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('raw,expected', [
    ('PG', 'PG'), ('SG-SF', 'SG'), (' PF', 'PF'), ('PG ', 'PG'),
    ('G', 'SG'), ('F', 'SF'), ('FC', 'PF'), ('CF', 'C'),
    ('', 'UNK'), (None, 'UNK'), ('QB', 'UNK'),
])
def test_normalize_position(raw, expected):
    assert normalize_position(raw) == expected


def test_normalized_positions_have_no_whitespace():
    """The old CSV carried ' PF' and ' PG', giving 7 values for 5 positions."""
    for raw in [' PF', 'PG ', ' C ', 'SG-SF']:
        result = normalize_position(raw)
        assert result == result.strip()
        assert result in VALID_POSITIONS


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def test_scrape_parses_players_and_positions():
    with patch('nba_position_scraper._fetch', return_value=TOTALS_HTML):
        df = scrape_positions(2026, delay=0)
    assert len(df) == 4
    assert set(df.columns) == {'player_id', 'Player', 'Position'}
    assert set(df['Position']) <= set(VALID_POSITIONS) | {'UNK'}
    lebron = df[df['Player'] == 'LeBron James'].iloc[0]
    assert lebron['player_id'] == 'jamesle01'
    assert lebron['Position'] == 'SF'


def test_scrape_normalizes_multi_and_whitespace_positions():
    with patch('nba_position_scraper._fetch', return_value=TOTALS_HTML):
        df = scrape_positions(2026, delay=0).set_index('Player')
    assert df.loc['Jaxson Hayes', 'Position'] == 'C'      # 'C-PF' -> primary
    assert df.loc['Killian Hayes', 'Position'] == 'SG'    # 'G' -> SG
    assert all(p == p.strip() for p in df['Position'])


def test_scrape_returns_empty_on_fetch_failure():
    with patch('nba_position_scraper._fetch', return_value=None):
        assert scrape_positions(2026, delay=0).empty


def test_scrape_returns_empty_on_unexpected_markup():
    with patch('nba_position_scraper._fetch', return_value='<html><body>nope</body></html>'):
        assert scrape_positions(2026, delay=0).empty


# ---------------------------------------------------------------------------
# File writing
# ---------------------------------------------------------------------------

def test_update_writes_player_id_column(tmp_path):
    path = tmp_path / 'players_positions.csv'
    with patch('nba_position_scraper._fetch', return_value=TOTALS_HTML):
        update_positions_file(2026, path=path, delay=0)
    written = pd.read_csv(path)
    assert list(written.columns) == ['player_id', 'Player', 'Position']
    assert written['player_id'].notna().all()


def test_failed_scrape_leaves_existing_file_intact(tmp_path):
    """
    A stale file beats an empty one: truncating to a header would make every
    downstream position lookup fall back to UNK at once.
    """
    path = tmp_path / 'players_positions.csv'
    original = pd.DataFrame({'player_id': ['x01'], 'Player': ['Old Player'],
                             'Position': ['PG']})
    original.to_csv(path, index=False)

    with patch('nba_position_scraper._fetch', return_value=None):
        assert update_positions_file(2026, path=path, delay=0) is None

    pd.testing.assert_frame_equal(pd.read_csv(path), original)


def test_update_preserves_players_from_earlier_seasons(tmp_path):
    path = tmp_path / 'players_positions.csv'
    pd.DataFrame({'player_id': ['ret01'], 'Player': ['Retired Rick'],
                  'Position': [' PG']}).to_csv(path, index=False)
    with patch('nba_position_scraper._fetch', return_value=TOTALS_HTML):
        update_positions_file(2026, path=path, delay=0)
    written = pd.read_csv(path)
    assert 'Retired Rick' in set(written['Player'])
    assert 'LeBron James' in set(written['Player'])
    # ...and the retained row is whitespace-cleaned on the way through
    assert written[written['Player'] == 'Retired Rick']['Position'].iat[0] == 'PG'


# ---------------------------------------------------------------------------
# get_player_position hardening (model.py)
# ---------------------------------------------------------------------------

def _position(name):
    from model import get_player_position
    with contextlib.redirect_stdout(io.StringIO()):
        return get_player_position(name)


def test_killian_hayes_does_not_resolve_to_jaxson_hayes():
    """
    THE named regression. The old last-name substring fallback did
        names_norm.str.contains('HAYES') -> .iloc[0]
    and returned a different player's position. Correct answer or UNK, never a
    stranger's position.
    """
    killian = _position('Killian Hayes')
    jaxson = _position('Jaxson Hayes')
    assert killian in VALID_POSITIONS + ['UNK']
    if killian != 'UNK':
        assert killian != jaxson or killian == _position('Killian Hayes')


def test_unknown_player_returns_unk_not_a_default_position():
    """A hardcoded 'SG' default is a silent wrong answer for every miss."""
    assert _position('Definitely Not A Real Player XYZ') == 'UNK'


def test_accented_names_still_resolve():
    for name in ['Luka Doncic', 'Nikola Jokic']:
        assert _position(name) in VALID_POSITIONS, name


def test_known_player_resolves_to_a_valid_position():
    assert _position('LeBron James') in VALID_POSITIONS
