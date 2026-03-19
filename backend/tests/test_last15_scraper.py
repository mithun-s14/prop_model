"""
Tests for fantasypros_scraper.py.
Validates scraping, caching, and player feature lookup without network calls.
"""
import sys
import os
import pytest
import pandas as pd
from io import StringIO
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from last15_scraper import (
    scrape_fantasypros_last15,
    load_fantasypros_last15,
    get_player_last15_features,
    _clean_player_name,
)


# ---------------------------------------------------------------------------
# _clean_player_name
# ---------------------------------------------------------------------------

class TestCleanPlayerName:
    def test_strips_team_and_position_suffix(self):
        assert _clean_player_name('Luka Doncic (LAL - PG,SG)') == 'Luka Doncic'

    def test_strips_single_position(self):
        assert _clean_player_name('LeBron James (LAL - SF)') == 'LeBron James'

    def test_removes_dtd_status(self):
        assert _clean_player_name('Jaylen Brown DTD') == 'Jaylen Brown'

    def test_removes_out_status(self):
        assert _clean_player_name('Kawhi Leonard OUT') == 'Kawhi Leonard'

    def test_removes_g_league_status(self):
        assert _clean_player_name('Some Player G-League') == 'Some Player'

    def test_removes_two_way_status(self):
        assert _clean_player_name('Some Player TWO-WAY') == 'Some Player'

    def test_removes_fa_status(self):
        assert _clean_player_name('Some Player FA') == 'Some Player'

    def test_strips_surrounding_quotes(self):
        assert _clean_player_name('"Jayson Tatum"') == 'Jayson Tatum'

    def test_name_without_suffix_unchanged(self):
        assert _clean_player_name('Stephen Curry') == 'Stephen Curry'

    def test_non_string_passthrough(self):
        assert _clean_player_name(None) is None


# ---------------------------------------------------------------------------
# scrape_fantasypros_last15
# ---------------------------------------------------------------------------

class TestScrapeFantasyProsLast15:
    def _mock_page(self, df):
        mock = MagicMock()
        mock.html_content = df.to_html(index=False)
        return mock

    @patch('last15_scraper.StealthyFetcher.fetch')
    def test_returns_dataframe(self, mock_fetch, sample_fantasypros_df):
        mock_fetch.return_value = self._mock_page(sample_fantasypros_df)
        result = scrape_fantasypros_last15()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    @patch('last15_scraper.StealthyFetcher.fetch')
    def test_returns_correct_row_count(self, mock_fetch, sample_fantasypros_df):
        mock_fetch.return_value = self._mock_page(sample_fantasypros_df)
        result = scrape_fantasypros_last15()
        assert len(result) == len(sample_fantasypros_df)

    @patch('last15_scraper.StealthyFetcher.fetch')
    def test_drops_rank_column(self, mock_fetch, sample_fantasypros_df):
        df_with_rank = sample_fantasypros_df.copy()
        df_with_rank.insert(0, 'RK', range(1, len(df_with_rank) + 1))
        mock_fetch.return_value = self._mock_page(df_with_rank)
        result = scrape_fantasypros_last15()
        assert 'RK' not in result.columns

    @patch('last15_scraper.StealthyFetcher.fetch')
    def test_deduplicates_players(self, mock_fetch, sample_fantasypros_df):
        df_dupes = pd.concat([sample_fantasypros_df, sample_fantasypros_df], ignore_index=True)
        mock_fetch.return_value = self._mock_page(df_dupes)
        result = scrape_fantasypros_last15()
        assert result['Player'].nunique() == sample_fantasypros_df['Player'].nunique()

    @patch('last15_scraper.StealthyFetcher.fetch')
    def test_stat_columns_are_numeric(self, mock_fetch, sample_fantasypros_df):
        mock_fetch.return_value = self._mock_page(sample_fantasypros_df)
        result = scrape_fantasypros_last15()
        for col in result.columns:
            if col not in ('Player', 'Team', 'POS'):
                assert pd.api.types.is_numeric_dtype(result[col]), f"{col} is not numeric"

    @patch('last15_scraper.pd.read_html', side_effect=ValueError("No tables found"))
    @patch('last15_scraper.StealthyFetcher.fetch')
    def test_no_table_returns_empty_df(self, mock_fetch, mock_read_html, sample_fantasypros_df):
        mock_fetch.return_value = self._mock_page(sample_fantasypros_df)
        result = scrape_fantasypros_last15()
        assert result.empty

    @patch('last15_scraper.StealthyFetcher.fetch')
    def test_drops_mid_table_header_rows(self, mock_fetch, sample_fantasypros_df):
        """FantasyPros sometimes injects a repeated header row mid-table."""
        header_row = pd.DataFrame([{col: col for col in sample_fantasypros_df.columns}])
        df_with_header = pd.concat([sample_fantasypros_df, header_row], ignore_index=True)
        mock_fetch.return_value = self._mock_page(df_with_header)
        result = scrape_fantasypros_last15()
        assert not (result['Player'] == 'Player').any()

    @patch('last15_scraper.StealthyFetcher.fetch')
    def test_cleans_player_names(self, mock_fetch, sample_fantasypros_df):
        """Team/position suffix and status tags should be stripped from names."""
        dirty_df = sample_fantasypros_df.copy()
        dirty_df['Player'] = [
            'LeBron James (LAL - SF)',
            'Stephen Curry DTD',
            '"Kevin Durant"',
            'Jayson Tatum (BOS - SF,PF)',
        ]
        mock_fetch.return_value = self._mock_page(dirty_df)
        result = scrape_fantasypros_last15()
        assert list(result['Player']) == ['LeBron James', 'Stephen Curry', 'Kevin Durant', 'Jayson Tatum']


# ---------------------------------------------------------------------------
# load_fantasypros_last15
# ---------------------------------------------------------------------------

class TestLoadFantasyProsLast15:
    @patch('last15_scraper.scrape_fantasypros_last15')
    @patch('last15_scraper.os.path.exists', return_value=False)
    def test_scrapes_when_no_cache(self, mock_exists, mock_scrape, sample_fantasypros_df):
        mock_scrape.return_value = sample_fantasypros_df
        with patch('last15_scraper.pd.DataFrame.to_csv'):
            result = load_fantasypros_last15()
        mock_scrape.assert_called_once()
        assert not result.empty

    @patch('last15_scraper.scrape_fantasypros_last15')
    @patch('last15_scraper.pd.read_csv')
    @patch('last15_scraper.os.path.getmtime')
    @patch('last15_scraper.os.path.exists', return_value=True)
    def test_uses_cache_when_fresh(self, mock_exists, mock_mtime, mock_csv, mock_scrape, sample_fantasypros_df):
        mock_mtime.return_value = datetime.now().timestamp()
        mock_csv.return_value = sample_fantasypros_df
        result = load_fantasypros_last15()
        mock_scrape.assert_not_called()
        assert not result.empty

    @patch('last15_scraper.scrape_fantasypros_last15')
    @patch('last15_scraper.pd.read_csv')
    @patch('last15_scraper.os.path.getmtime')
    @patch('last15_scraper.os.path.exists', return_value=True)
    def test_re_scrapes_when_stale(self, mock_exists, mock_mtime, mock_csv, mock_scrape, sample_fantasypros_df):
        mock_mtime.return_value = (datetime.now() - timedelta(hours=48)).timestamp()
        mock_scrape.return_value = sample_fantasypros_df
        with patch('last15_scraper.pd.DataFrame.to_csv'):
            load_fantasypros_last15()
        mock_scrape.assert_called_once()

    @patch('last15_scraper.scrape_fantasypros_last15')
    @patch('last15_scraper.pd.read_csv')
    @patch('last15_scraper.os.path.getmtime')
    @patch('last15_scraper.os.path.exists', return_value=True)
    def test_force_refresh_bypasses_fresh_cache(self, mock_exists, mock_mtime, mock_csv, mock_scrape, sample_fantasypros_df):
        mock_mtime.return_value = datetime.now().timestamp()
        mock_scrape.return_value = sample_fantasypros_df
        with patch('last15_scraper.pd.DataFrame.to_csv'):
            load_fantasypros_last15(force_refresh=True)
        mock_scrape.assert_called_once()


# ---------------------------------------------------------------------------
# get_player_last15_features
# ---------------------------------------------------------------------------

class TestGetPlayerLast15Features:
    @patch('last15_scraper.load_fantasypros_last15')
    def test_exact_name_match(self, mock_load, sample_fantasypros_df):
        mock_load.return_value = sample_fantasypros_df
        result = get_player_last15_features('LeBron James')
        assert result['pts_roll_avg'] == 27.3

    @patch('last15_scraper.load_fantasypros_last15')
    def test_partial_name_match(self, mock_load, sample_fantasypros_df):
        mock_load.return_value = sample_fantasypros_df
        result = get_player_last15_features('LeBron')
        assert result['pts_roll_avg'] == 27.3

    @patch('last15_scraper.load_fantasypros_last15')
    def test_last_name_match(self, mock_load, sample_fantasypros_df):
        mock_load.return_value = sample_fantasypros_df
        result = get_player_last15_features('Curry')
        assert result['pts_roll_avg'] == 29.8

    @patch('last15_scraper.load_fantasypros_last15')
    def test_player_not_found_returns_empty_dict(self, mock_load, sample_fantasypros_df):
        mock_load.return_value = sample_fantasypros_df
        result = get_player_last15_features('Fake Player')
        assert result == {}

    @patch('last15_scraper.load_fantasypros_last15')
    def test_returns_all_expected_keys(self, mock_load, sample_fantasypros_df):
        mock_load.return_value = sample_fantasypros_df
        result = get_player_last15_features('LeBron James')
        expected = {
            'pts_roll_avg', 'reb_roll_avg', 'ast_roll_avg', 'min_roll_avg',
            'fga_roll_avg', 'fg3a_roll_avg', 'fta_roll_avg',
            'stl_roll_avg', 'blk_roll_avg', 'tov_roll_avg',
        }
        assert set(result.keys()) == expected

    @patch('last15_scraper.load_fantasypros_last15')
    def test_all_values_are_floats(self, mock_load, sample_fantasypros_df):
        mock_load.return_value = sample_fantasypros_df
        result = get_player_last15_features('Kevin Durant')
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not float: {type(val)}"

    @patch('last15_scraper.load_fantasypros_last15')
    def test_maps_correct_columns(self, mock_load, sample_fantasypros_df):
        mock_load.return_value = sample_fantasypros_df
        result = get_player_last15_features('Jayson Tatum')
        row = sample_fantasypros_df[sample_fantasypros_df['Player'] == 'Jayson Tatum'].iloc[0]
        assert result['pts_roll_avg']  == float(row['PTS'])
        assert result['reb_roll_avg']  == float(row['REB'])
        assert result['ast_roll_avg']  == float(row['AST'])
        assert result['min_roll_avg']  == float(row['MIN'])
        assert result['fga_roll_avg']  == float(row['2PM'])   # proxy (FGA not in data)
        assert result['fg3a_roll_avg'] == float(row['3PM'])   # proxy (3PA not in data)
        assert result['fta_roll_avg']  == float(row['FTM'])   # proxy (FTA not in data)
        assert result['stl_roll_avg']  == float(row['STL'])
        assert result['blk_roll_avg']  == float(row['BLK'])
        assert result['tov_roll_avg']  == float(row['TO'])

    @patch('last15_scraper.load_fantasypros_last15')
    def test_empty_dataframe_returns_empty_dict(self, mock_load):
        mock_load.return_value = pd.DataFrame()
        result = get_player_last15_features('LeBron James')
        assert result == {}
