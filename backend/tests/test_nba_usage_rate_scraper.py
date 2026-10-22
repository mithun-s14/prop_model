"""
Tests for nba_usage_rate_scraper.py.
Validates parsing of the Basketball Reference advanced table without network calls.
"""
import sys
import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nba_usage_rate_scraper import scrape_nba_usage_rates, save_usage_data


def _advanced_html(rows):
    df = pd.DataFrame(rows, columns=['Player', 'Tm', 'USG%'])
    return df.to_html(index=False, table_id='advanced')


def _mock_page(rows):
    mock = MagicMock()
    mock.html_content = _advanced_html(rows)
    return mock


ROWS = [
    ['Luka Doncic', 'LAL', '36.1'],
    ['Player', 'Tm', 'USG%'],          # BBR mid-table header row
    ['Jayson Tatum', 'BOS', '30.2'],
    ['Jayson Tatum', 'TOT', '29.0'],   # duplicate after trade, keep first
    ['Bench Guy', 'NYK', ''],          # no usage -> dropped
]


class TestScrapeNbaUsageRates:
    @patch('nba_usage_rate_scraper.StealthyFetcher.fetch')
    def test_waits_for_table_not_network_idle(self, mock_fetch):
        mock_fetch.return_value = _mock_page(ROWS)
        scrape_nba_usage_rates()
        kwargs = mock_fetch.call_args.kwargs
        assert kwargs['wait_selector'] == 'table#advanced'
        assert not kwargs.get('network_idle')

    @patch('nba_usage_rate_scraper.StealthyFetcher.fetch')
    def test_returns_expected_columns(self, mock_fetch):
        mock_fetch.return_value = _mock_page(ROWS)
        result = scrape_nba_usage_rates()
        assert list(result.columns) == ['PLAYER', 'Team', 'USGPCT']

    @patch('nba_usage_rate_scraper.StealthyFetcher.fetch')
    def test_drops_header_rows_duplicates_and_missing_usage(self, mock_fetch):
        mock_fetch.return_value = _mock_page(ROWS)
        result = scrape_nba_usage_rates()
        assert result['PLAYER'].tolist() == ['Luka Doncic', 'Jayson Tatum']
        assert result.loc[result['PLAYER'] == 'Jayson Tatum', 'Team'].item() == 'BOS'

    @patch('nba_usage_rate_scraper.StealthyFetcher.fetch')
    def test_usage_is_numeric(self, mock_fetch):
        mock_fetch.return_value = _mock_page(ROWS)
        result = scrape_nba_usage_rates()
        assert pd.api.types.is_numeric_dtype(result['USGPCT'])
        assert result['USGPCT'].between(0, 100).all()

    @patch('nba_usage_rate_scraper.StealthyFetcher.fetch')
    def test_missing_table_raises(self, mock_fetch):
        # e.g. a Cloudflare challenge page served instead of the stats page
        mock = MagicMock()
        mock.html_content = '<html><body>Just a moment...</body></html>'
        mock_fetch.return_value = mock
        # ValueError from lxml, or ImportError if pandas falls back to html5lib
        with pytest.raises((ValueError, ImportError)):
            scrape_nba_usage_rates()


class TestSaveUsageData:
    def test_empty_df_not_saved(self):
        assert save_usage_data(pd.DataFrame()) is None

    def test_none_not_saved(self):
        assert save_usage_data(None) is None
