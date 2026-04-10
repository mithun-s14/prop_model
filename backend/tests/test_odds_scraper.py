"""
Tests for odds_scraper.py.
All network/subprocess calls are mocked.
"""
import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from odds_scraper import (
    get_today_str,
    run_oddsharvester,
    parse_odds_output,
    _flatten_match,
    save_odds,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_MATCH = {
    "home_team": "Los Angeles Lakers",
    "away_team": "Boston Celtics",
    "date": "2026-04-09T19:30:00",
    "markets": {
        "spreads": [
            {
                "home_handicap": -5.5,
                "home_odds": -110,
                "away_handicap": 5.5,
                "away_odds": -110,
            }
        ],
        "totals": [
            {
                "total": 224.5,
                "over_odds": -110,
                "under_odds": -110,
            }
        ],
    },
}

SAMPLE_MATCH_NO_ODDS = {
    "home_team": "Golden State Warriors",
    "away_team": "Phoenix Suns",
    "date": "2026-04-09T21:00:00",
    "markets": {},
}


# ---------------------------------------------------------------------------
# get_today_str
# ---------------------------------------------------------------------------

class TestGetTodayStr:
    def test_returns_eight_digit_string(self):
        result = get_today_str()
        assert len(result) == 8
        assert result.isdigit()

    def test_format_is_yyyymmdd(self):
        from datetime import datetime
        expected = datetime.now().strftime("%Y%m%d")
        assert get_today_str() == expected


# ---------------------------------------------------------------------------
# run_oddsharvester
# ---------------------------------------------------------------------------

class TestRunOddsHarvester:
    @patch("odds_scraper.subprocess.run")
    def test_returns_true_on_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        assert run_oddsharvester("20260409", "/tmp/out.json") is True

    @patch("odds_scraper.subprocess.run")
    def test_returns_false_on_nonzero_exit(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        assert run_oddsharvester("20260409", "/tmp/out.json") is False

    @patch("odds_scraper.subprocess.run", side_effect=FileNotFoundError)
    def test_returns_false_when_not_installed(self, mock_run):
        assert run_oddsharvester("20260409", "/tmp/out.json") is False

    @patch("odds_scraper.subprocess.run")
    def test_passes_correct_date(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        run_oddsharvester("20260409", "/tmp/out.json")
        cmd = mock_run.call_args[0][0]
        assert "20260409" in cmd

    @patch("odds_scraper.subprocess.run")
    def test_passes_spreads_and_totals_markets(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        run_oddsharvester("20260409", "/tmp/out.json")
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "spreads" in cmd_str
        assert "totals" in cmd_str

    @patch("odds_scraper.subprocess.run")
    def test_passes_headless_flag(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        run_oddsharvester("20260409", "/tmp/out.json")
        cmd = mock_run.call_args[0][0]
        assert "--headless" in cmd


# ---------------------------------------------------------------------------
# _flatten_match
# ---------------------------------------------------------------------------

class TestFlattenMatch:
    def test_extracts_home_and_away_team(self):
        result = _flatten_match(SAMPLE_MATCH)
        assert result["home_team"] == "Los Angeles Lakers"
        assert result["away_team"] == "Boston Celtics"

    def test_extracts_spread_line(self):
        result = _flatten_match(SAMPLE_MATCH)
        assert result["home_spread"] == -5.5
        assert result["away_spread"] == 5.5

    def test_extracts_spread_odds(self):
        result = _flatten_match(SAMPLE_MATCH)
        assert result["home_spread_odds"] == -110
        assert result["away_spread_odds"] == -110

    def test_extracts_total_line(self):
        result = _flatten_match(SAMPLE_MATCH)
        assert result["total_line"] == 224.5

    def test_extracts_over_under_odds(self):
        result = _flatten_match(SAMPLE_MATCH)
        assert result["over_odds"] == -110
        assert result["under_odds"] == -110

    def test_missing_markets_yields_none_odds(self):
        result = _flatten_match(SAMPLE_MATCH_NO_ODDS)
        assert result["home_spread"] is None
        assert result["total_line"] is None

    def test_returns_none_when_teams_missing(self):
        assert _flatten_match({}) is None
        assert _flatten_match({"home_team": "Lakers"}) is None

    def test_sport_is_nba(self):
        result = _flatten_match(SAMPLE_MATCH)
        assert result["sport"] == "NBA"


# ---------------------------------------------------------------------------
# parse_odds_output
# ---------------------------------------------------------------------------

class TestParseOddsOutput:
    def test_parses_list_format(self, tmp_path):
        path = tmp_path / "odds.json"
        path.write_text(json.dumps([SAMPLE_MATCH]))
        result = parse_odds_output(str(path))
        assert len(result) == 1
        assert result[0]["home_team"] == "Los Angeles Lakers"

    def test_parses_dict_wrapper_format(self, tmp_path):
        path = tmp_path / "odds.json"
        path.write_text(json.dumps({"data": [SAMPLE_MATCH]}))
        result = parse_odds_output(str(path))
        assert len(result) == 1

    def test_returns_empty_list_when_file_missing(self):
        result = parse_odds_output("/nonexistent/path.json")
        assert result == []

    def test_returns_empty_list_for_empty_json_array(self, tmp_path):
        path = tmp_path / "odds.json"
        path.write_text("[]")
        result = parse_odds_output(str(path))
        assert result == []

    def test_skips_match_without_teams(self, tmp_path):
        path = tmp_path / "odds.json"
        path.write_text(json.dumps([{}, SAMPLE_MATCH]))
        result = parse_odds_output(str(path))
        assert len(result) == 1

    def test_handles_multiple_games(self, tmp_path):
        path = tmp_path / "odds.json"
        match2 = {**SAMPLE_MATCH, "home_team": "Miami Heat", "away_team": "Chicago Bulls"}
        path.write_text(json.dumps([SAMPLE_MATCH, match2]))
        result = parse_odds_output(str(path))
        assert len(result) == 2


# ---------------------------------------------------------------------------
# save_odds
# ---------------------------------------------------------------------------

class TestSaveOdds:
    def test_creates_json_file(self, tmp_path):
        save_odds([_flatten_match(SAMPLE_MATCH)], str(tmp_path))
        assert (tmp_path / "nba_odds_today.json").exists()

    def test_creates_csv_file(self, tmp_path):
        save_odds([_flatten_match(SAMPLE_MATCH)], str(tmp_path))
        assert (tmp_path / "nba_odds_today.csv").exists()

    def test_json_contains_correct_data(self, tmp_path):
        save_odds([_flatten_match(SAMPLE_MATCH)], str(tmp_path))
        with open(tmp_path / "nba_odds_today.json") as f:
            data = json.load(f)
        assert data[0]["home_team"] == "Los Angeles Lakers"
        assert data[0]["total_line"] == 224.5

    def test_empty_records_writes_empty_files(self, tmp_path):
        save_odds([], str(tmp_path))
        with open(tmp_path / "nba_odds_today.json") as f:
            assert json.load(f) == []
        csv_content = (tmp_path / "nba_odds_today.csv").read_text().strip()
        assert csv_content == ""

    def test_csv_has_expected_columns(self, tmp_path):
        save_odds([_flatten_match(SAMPLE_MATCH)], str(tmp_path))
        import pandas as pd
        df = pd.read_csv(tmp_path / "nba_odds_today.csv")
        for col in ("home_team", "away_team", "home_spread", "total_line", "over_odds"):
            assert col in df.columns
