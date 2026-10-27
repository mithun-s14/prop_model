"""
Tests for covers_odds_scraper.py (standalone; the model no longer uses
spread/total). HTML fixtures mirror the covers.com odds page
markup (tr.oddsGameRow / td[data-book] / .home-cell a).
"""
import os
import sys
from unittest.mock import patch

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import covers_odds_scraper  # noqa: E402
from covers_odds_scraper import get_team_lines, parse_game_lines  # noqa: E402


def _cell(book, line):
    if line is None:  # book has no line posted
        return f'<td data-book="{book}"><div class="td-cell odds-book-image"></div></td>'
    return (f'<td data-book="{book}"><div class="td-cell away-cell"><a> x <span>-110</span></a></div>'
            f'<div class="td-cell home-cell"><a> {line} <span class="American">-110</span>'
            f'<span class="Decimal">1.91</span></a></div></td>')


def _row(away, home, lines):
    return (f'<tr class="oddsGameRow"><td class="left-cell basketball">'
            f'<div class="td-cell away-cell"><a><strong>{away}</strong></a></div>'
            f'<div class="td-cell home-cell"><a><strong>{home}</strong></a></div></td>'
            + ''.join(_cell(f'Book{i}', l) for i, l in enumerate(lines)) + '</tr>')


def _page(spread_rows, total_rows):
    return (f'<table id="moneyline-table"><tbody>{_row("MIN", "CLE", ["+140"])}</tbody></table>'
            f'<table id="spread-table"><tbody>{"".join(spread_rows)}</tbody></table>'
            f'<table id="total-table"><tbody>{"".join(total_rows)}</tbody></table>')


PAGE = _page(
    [_row('MIN', 'CLE', ['-3.5', '-3.5', '-4.5']),
     _row('IND', 'MIL', ['+4.5', None, '+4.5']),
     _row('GS', 'DEN', ['PK', 'PK'])],
    [_row('MIN', 'CLE', ['u 232.5', 'u 233.5', 'u 232.5']),
     _row('IND', 'MIL', ['u 239.5'])],
)


class TestParseGameLines:
    def test_consensus_is_median_across_books(self):
        games = parse_game_lines(PAGE)
        assert games[('MIN', 'CLE')] == {'home_spread': -3.5, 'total': 232.5}

    def test_positive_spread_and_missing_book(self):
        assert parse_game_lines(PAGE)[('IND', 'MIL')] == {'home_spread': 4.5, 'total': 239.5}

    def test_covers_abbreviations_normalized_to_bbref(self):
        games = parse_game_lines(PAGE)
        assert ('GSW', 'DEN') in games
        assert ('GS', 'DEN') not in games

    def test_pick_em_is_zero_and_missing_total_absent(self):
        assert parse_game_lines(PAGE)[('GSW', 'DEN')] == {'home_spread': 0.0}

    def test_moneyline_table_ignored(self):
        html = _page([], [])
        assert parse_game_lines(html) == {}

    def test_page_without_tables(self):
        assert parse_game_lines('<html><body>blocked</body></html>') == {}

    def test_first_listing_of_a_matchup_wins(self):
        html = _page([_row('MIN', 'CLE', ['-3.5']), _row('MIN', 'CLE', ['-9.5'])], [])
        assert parse_game_lines(html)[('MIN', 'CLE')]['home_spread'] == -3.5

    def test_all_values_are_floats(self):
        for lines in parse_game_lines(PAGE).values():
            assert all(isinstance(v, float) for v in lines.values())


class TestGetTeamLines:
    games = parse_game_lines(PAGE)

    def test_home_team_keeps_home_spread(self):
        assert get_team_lines('CLE', 'MIN', self.games) == (-3.5, 232.5)

    def test_away_team_spread_is_negated(self):
        assert get_team_lines('MIN', 'CLE', self.games) == (3.5, 232.5)

    def test_unknown_game_returns_none(self):
        assert get_team_lines('LAL', 'BOS', self.games) == (None, None)

    def test_missing_total_returns_none_total(self):
        assert get_team_lines('DEN', 'GSW', self.games) == (0.0, None)

    def test_network_failure_returns_none(self):
        with patch.object(covers_odds_scraper, 'fetch_game_lines',
                          side_effect=requests.ConnectionError('down')):
            assert get_team_lines('CLE', 'MIN') == (None, None)

    def test_fetches_when_no_games_given(self):
        with patch.object(covers_odds_scraper, 'fetch_game_lines', return_value=self.games) as f:
            assert get_team_lines('MIL', 'IND') == (4.5, 239.5)
        f.assert_called_once()


class TestFetchGameLines:
    def test_http_error_raises(self):
        resp = requests.Response()
        resp.status_code = 403
        with patch.object(covers_odds_scraper.requests, 'get', return_value=resp):
            with pytest.raises(requests.HTTPError):
                covers_odds_scraper.fetch_game_lines()

    def test_parses_response_body(self):
        resp = requests.Response()
        resp.status_code = 200
        resp._content = PAGE.encode()
        with patch.object(covers_odds_scraper.requests, 'get', return_value=resp):
            assert covers_odds_scraper.fetch_game_lines()[('MIN', 'CLE')]['total'] == 232.5


class TestGameContextIgnoresOdds:
    """Spread/total were dropped from the model; the scraper is kept standalone."""

    def test_context_has_no_odds_and_never_fetches(self):
        import model
        with patch.object(model, 'get_player_info', return_value={'id': 'x', 'team': 'MIN'}), \
             patch.object(model, 'get_player_team_id', return_value=1), \
             patch.object(model, 'get_team_schedule',
                          return_value={'HOME_TEAM_ID': 2, 'VISITOR_TEAM_ID': 1}), \
             patch.object(model.pd, 'read_csv',
                          return_value=model.pd.DataFrame({'id': [2], 'abbreviation': ['CLE']})), \
             patch.object(covers_odds_scraper, 'fetch_game_lines') as fetch:
            c = model.get_tonights_game_context('Anthony Edwards')
        fetch.assert_not_called()
        assert c == {'opponent': 'CLE', 'is_home': 0}


@pytest.mark.parametrize('req_file', ['requirements.txt', os.path.join('backend', 'requirements.txt')])
def test_scraper_deps_in_every_requirements_file(req_file):
    """Regression: bs4 was only in backend/requirements.txt, so the root
    (Gradio app) deploy crashed on `import bs4` at startup."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(root, req_file)) as fh:
        names = {line.split('==')[0].split('[')[0].strip().lower() for line in fh
                 if line.strip() and not line.startswith('#')}
    assert {'requests', 'beautifulsoup4'} <= names
