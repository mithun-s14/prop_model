import pandas as pd
import os
from nba_api.stats.endpoints import leaguedashplayerstats


def scrape_nba_usage_rates():
    """
    Fetch NBA player usage rates from the NBA Stats API via nba_api.
    Returns a DataFrame of all players with usage statistics.
    """
    print("Fetching NBA usage stats from NBA Stats API...")
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        measure_type_detailed_defense='Usage',
        per_mode_simple='PerGame',
        season='2025-26',
        season_type_all_star='Regular Season',
        timeout=60,
    )
    df = stats.get_data_frames()[0]
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
