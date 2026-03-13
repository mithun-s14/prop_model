import pandas as pd
import requests
import os


_BBR_URL = "https://www.basketball-reference.com/leagues/NBA_2026_advanced.html"

_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}


def scrape_nba_usage_rates():
    """
    Fetch NBA player usage rates from Basketball Reference advanced stats.
    Returns a DataFrame with PLAYER, TEAM, and USGPCT columns.
    """
    print("Fetching NBA usage stats from Basketball Reference...")
    response = requests.get(_BBR_URL, headers=_HEADERS, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(response.text, attrs={'id': 'advanced'})
    df = tables[0]

    # Drop duplicate header rows that BBR injects mid-table
    df = df[df['Player'] != 'Player'].reset_index(drop=True)

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(str(c) for c in col).strip() for col in df.columns]

    df = df.rename(columns={'Player': 'PLAYER', 'Tm': 'TEAM', 'USG%': 'USGPCT'})
    df['USGPCT'] = pd.to_numeric(df['USGPCT'], errors='coerce')

    # Keep one row per player (first occurrence = most recent team after trades)
    df = df.drop_duplicates(subset='PLAYER', keep='first')

    df = df[['PLAYER', 'TEAM', 'USGPCT']].dropna(subset=['USGPCT'])
    print(f"Successfully fetched data for {len(df)} players")
    return df


def save_usage_data(usage_df):
    if usage_df is None or usage_df.empty:
        print("No data to save")
        return None

    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'nba_usage_rates_latest.csv')
    usage_df.to_csv(csv_path, index=False)
    print(f"Saved to '{csv_path}'")
    return 'nba_usage_rates_latest.csv'


def cache_usage_data():
    print("Starting NBA usage rate scraping...")
    usage_data = scrape_nba_usage_rates()

    if usage_data is not None:
        print(f"\nData summary: {usage_data.shape[0]} players, columns: {list(usage_data.columns)}")
        csv_file = save_usage_data(usage_data)
        print(f"\nScraping completed successfully! CSV: {csv_file}")

        print("\nSample of players and usage rates:")
        for _, row in usage_data.head(10).iterrows():
            print(f"  {row['PLAYER']} ({row['TEAM']}): {row['USGPCT']}%")

        return usage_data
    else:
        print("Scraping failed - no data retrieved")
        return None


if __name__ == "__main__":
    cache_usage_data()
