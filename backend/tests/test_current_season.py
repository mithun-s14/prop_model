"""
Every live scraper must target the same, current NBA season. These defaults
were hand-edited when 2025-26 rolled to 2026-27; a scraper left behind would
silently keep serving last season's rosters or usage rates.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import historical_gamelog_scraper  # noqa: E402
import nba_data_scraper  # noqa: E402
import nba_usage_rate_scraper  # noqa: E402
from features import nba_season  # noqa: E402

import pandas as pd  # noqa: E402

SEASON = nba_data_scraper.CURRENT_SEASON


def _default(fn, name):
    return inspect.signature(fn).parameters[name].default


def test_current_season_is_2026_27():
    assert SEASON == 2027
    # Opening night 2026-10-20 belongs to the 2027 season in BBR numbering
    assert nba_season(pd.Series(pd.to_datetime(['2026-10-20']))).tolist() == [SEASON]


def test_roster_and_schedule_defaults_match():
    assert _default(nba_data_scraper.scrape_team_roster, 'season') == SEASON
    assert _default(nba_data_scraper.scrape_active_players_from_rosters, 'season') == SEASON
    # scrape_schedule takes the START year and requests NBA_{season + 1}
    assert _default(nba_data_scraper.scrape_schedule, 'season') + 1 == SEASON


def test_usage_rate_url_targets_current_season():
    assert f'NBA_{SEASON}_advanced' in nba_usage_rate_scraper._BBR_URL


def test_last_season_not_targeted():
    assert 'NBA_2026_' not in nba_usage_rate_scraper._BBR_URL
    assert _default(nba_data_scraper.scrape_team_roster, 'season') != 2026


def test_position_and_gamelog_cli_defaults():
    src = inspect.getsource(historical_gamelog_scraper)
    assert 'default=[2027]' in src
    import nba_position_scraper
    assert 'default=2027' in inspect.getsource(nba_position_scraper)
