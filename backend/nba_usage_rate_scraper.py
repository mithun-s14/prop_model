import pandas as pd
import json
import os
from urllib.parse import urlencode
from scrapling.fetchers import Fetcher


# NBA Stats API requires these headers to accept requests from non-browser clients
_NBA_HEADERS = {
    'Host': 'stats.nba.com',
    'Referer': 'https://www.nba.com/',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Origin': 'https://www.nba.com',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
}

_API_PARAMS = {
    'College': '', 'Conference': '', 'Country': '', 'DateFrom': '', 'DateTo': '',
    'Division': '', 'DraftPick': '', 'DraftYear': '', 'GameScope': '',
    'GameSegment': '', 'Height': '', 'ISTRound': '', 'LastNGames': 0,
    'LeagueID': '00', 'Location': '', 'MeasureType': 'Usage', 'Month': 0,
    'OpponentTeamID': 0, 'Outcome': '', 'PORound': 0, 'PaceAdjust': 'N',
    'PerMode': 'PerGame', 'Period': 0, 'PlayerExperience': '',
    'PlayerPosition': '', 'PlusMinus': 'N', 'Rank': 'N',
    'Season': '2025-26', 'SeasonSegment': '', 'SeasonType': 'Regular Season',
    'ShotClockRange': '', 'StarterBench': '', 'TeamID': 0,
    'VsConference': '', 'VsDivision': '', 'Weight': '',
}


def scrape_nba_usage_rates():
    """
    Fetch NBA player usage rates from the NBA Stats JSON API.
    Replaces the previous Selenium-based multi-page scraper with a single API call.
    Returns a DataFrame of all players with usage statistics.
    """
    url = "https://stats.nba.com/stats/leaguedashplayerstats?" + urlencode(_API_PARAMS)

    print("Fetching NBA usage stats from NBA Stats API...")
    page = Fetcher.get(url, stealthy_headers=True, headers=_NBA_HEADERS, timeout=30)

    data = json.loads(page.text)
    result_set = data['resultSets'][0]
    headers = result_set['headers']
    rows = result_set['rowSet']

    df = pd.DataFrame(rows, columns=headers)
    print(f"Successfully fetched data for {len(df)} players")
    return df


def clean_usage_data(df):
    """
    Clean and process the usage data.
    Renames API columns to match the expected output format and converts numeric types.
    """
    if df is None or df.empty:
        return df

    cleaned_df = df.copy()

    # Rename API column names to match the previous Selenium scraper output
    cleaned_df = cleaned_df.rename(columns={
        'PLAYER_NAME': 'PLAYER',
        'TEAM_ABBREVIATION': 'TEAM',
    })

    # Drop columns not needed downstream
    drop_cols = ['PLAYER_ID', 'NICKNAME', 'TEAM_ID', 'CFID', 'CFPARAMS']
    cleaned_df = cleaned_df.drop(columns=[c for c in drop_cols if c in cleaned_df.columns])

    # Convert numeric columns
    numeric_columns = [
        'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'USG_PCT',
        'PCT_FGM', 'PCT_FGA', 'PCT_FG3M', 'PCT_FG3A',
        'PCT_FTM', 'PCT_FTA', 'PCT_OREB', 'PCT_DREB', 'PCT_REB',
        'PCT_AST', 'PCT_TOV', 'PCT_STL', 'PCT_BLK', 'PCT_BLKA',
        'PCT_PF', 'PCT_PFD', 'PCT_PTS',
    ]
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

    print(f"Data cleaning completed. {len(cleaned_df.columns)} columns, {len(cleaned_df)} players")
    return cleaned_df


def save_usage_data(usage_df):
    """
    Clean and save usage data to CSV.
    """
    if usage_df is None or usage_df.empty:
        print("No data to save")
        return None

    cleaned_df = clean_usage_data(usage_df)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'nba_usage_rates_latest.csv')
    cleaned_df.to_csv(csv_path, index=False)
    print(f"Saved to '{csv_path}'")

    return 'nba_usage_rates_latest.csv'


def cache_usage_data():
    """
    Main function: fetch and cache NBA usage rate data.
    """
    print("Starting NBA usage rate scraping...")

    usage_data = scrape_nba_usage_rates()

    if usage_data is not None:
        print(f"\nData summary:")
        print(f"Columns: {list(usage_data.columns)}")
        print(f"Shape: {usage_data.shape}")

        csv_file = save_usage_data(usage_data)
        print(f"\nScraping completed successfully! CSV: {csv_file}")

        if not usage_data.empty:
            key_cols = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'USG_PCT']
            available_cols = [col for col in key_cols if col in usage_data.columns]
            if available_cols:
                print("\nSample of players and usage rates:")
                for _, row in usage_data[available_cols].head(10).iterrows():
                    print(f"  {row.get('PLAYER_NAME', 'N/A')} ({row.get('TEAM_ABBREVIATION', 'N/A')}): {row.get('USG_PCT', 'N/A')}%")

        return usage_data
    else:
        print("Scraping failed - no data retrieved")
        return None


if __name__ == "__main__":
    cache_usage_data()
