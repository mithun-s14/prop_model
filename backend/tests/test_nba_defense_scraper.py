"""
Tests for nba_defense_scraper.py.

Regression: the scraper took the 4th table on the page by index. The site
dropped one of its 5-row summary tables, leaving 3, and the scrape returned
nothing. It now picks the largest table regardless of position.
"""
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapling.parser import Adaptor  # noqa: E402

import nba_defense_scraper  # noqa: E402
from nba_defense_scraper import clean_position_data, scrape_hashtag_basketball_table4  # noqa: E402

HEADER = ("<tr>" + "".join(f"<th>Sort: {c}</th>" for c in
          ["Position", "Team", "PTS", "FG%", "FT%", "3PM", "REB", "AST", "STL", "BLK", "TO"]) + "</tr>")


def _row(pos, team, pts):
    return (f"<tr><td>{pos}</td><td>{team} 1</td><td>{pts} 20</td><td>43.3 12</td><td>80.6 98</td>"
            "<td>3.1 111</td><td>7.8 77</td><td>3.6 7</td><td>1.1 5</td><td>0.6 33</td><td>2.1 61</td></tr>")


def _table(rows):
    return f"<table>{HEADER}{''.join(rows)}</table>"


SUMMARY = _table([_row("PG", "OKC", 21.6)])
FULL = _table([_row(p, t, 20.0 + i) for i, p in enumerate(["PG", "SG", "SF", "PF", "C"])
               for t in ["BOS", "NY"]])


def _scrape(html):
    with patch.object(nba_defense_scraper.DynamicFetcher, "fetch", return_value=Adaptor(html)):
        return scrape_hashtag_basketball_table4()


@pytest.mark.parametrize("n_summaries", [0, 1, 2, 3])
def test_finds_full_table_regardless_of_position(n_summaries):
    data = _scrape(f"<html><body>{SUMMARY * n_summaries}{FULL}</body></html>")
    assert sorted(data) == ["C", "PF", "PG", "SF", "SG"]
    assert list(data["PG"]["Team"]) == ["BOS", "NY"]
    assert list(data["PG"]["PTS"]) == [20.0, 20.0]
    assert list(data["C"]["PTS"]) == [24.0, 24.0]


def test_no_tables_returns_empty():
    assert _scrape("<html><body><p>nothing</p></body></html>") == {}


def test_fetch_error_returns_empty():
    with patch.object(nba_defense_scraper.DynamicFetcher, "fetch", side_effect=RuntimeError("boom")):
        assert scrape_hashtag_basketball_table4() == {}


def test_clean_position_data_strips_ranks():
    import pandas as pd
    df = pd.DataFrame({"Team": ["BOS 1"], "PTS": ["21.4 20"], "FG%": ["43.3 12"], "BLK": ["bad 3"]})
    out = clean_position_data(df)
    assert out["Team"][0] == "BOS"
    assert out["PTS"][0] == 21.4
    assert out["FG%"][0] == "43.3"
    assert pd.isna(out["BLK"][0])
