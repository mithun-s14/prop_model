from io import StringIO
import pandas as pd
import os
from scrapling.fetchers import DynamicFetcher


def scrape_hashtag_basketball_table4():
    """
    Scrape NBA defense vs position data from Hashtag Basketball.
    Uses Scrapling's DynamicFetcher (Playwright-based) to render the JS page,
    replacing the previous Selenium implementation.
    """
    positions_data = {}

    try:
        print("Loading Hashtag Basketball defense vs position page...")
        page = DynamicFetcher.fetch(
            "https://hashtagbasketball.com/nba-defense-vs-position",
            headless=True,
            network_idle=True,
        )

        print("Looking for all tables on the page...")
        all_tables = page.css('table')
        print(f"Found {len(all_tables)} tables on the page")

        # Table 4 (index 3) contains position-based defense data with ~150 rows
        if len(all_tables) < 4:
            print("'Position data' table not found!")
            return positions_data

        # .get() returns the outer HTML of the element (Scrapy-compatible API)
        table_html = all_tables[3].get()
        df_list = pd.read_html(StringIO(table_html))

        if not df_list:
            print("Could not parse 'Position data' table")
            return positions_data

        main_df = df_list[0]
        print(f"'Position data' Table shape: {main_df.shape}")
        print(f"Raw columns: {main_df.columns.tolist()}")

        # Clean column names - remove the "Sort: " prefix
        main_df.columns = [col.replace('Sort: ', '') for col in main_df.columns]
        print(f"Cleaned columns: {main_df.columns.tolist()}")

        print("\nFirst 3 rows of Table 4:")
        print(main_df.head(3).to_string(index=False))

        for position in ['PG', 'SG', 'SF', 'PF', 'C']:
            try:
                position_df = main_df[main_df['Position'] == position].copy()
                if not position_df.empty:
                    positions_data[position] = clean_position_data(position_df)
                else:
                    print(f"No data found for position: {position}")
            except Exception as e:
                print(f"Error processing {position}: {e}")

    except Exception as e:
        print(f"Error during scraping: {e}")
        import traceback
        traceback.print_exc()

    return positions_data


def clean_position_data(df):
    """
    Clean the position data by extracting main values and removing rank numbers.
    """
    cleaned_df = df.copy()

    columns_to_clean = ['PTS', 'FG%', 'FT%', '3PM', 'REB', 'AST', 'STL', 'BLK', 'TO']

    for col in columns_to_clean:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.split().str[0]
            if col not in ['FG%', 'FT%']:
                try:
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                except Exception:
                    pass

    if 'Team' in cleaned_df.columns:
        cleaned_df['Team'] = cleaned_df['Team'].astype(str).str.split().str[0]

    return cleaned_df


def export_to_excel(positions_data, filename='nba_defense_data.xlsx'):
    """
    Export data to Excel with separate sheets for each position.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filename)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for position, df in positions_data.items():
                if not df.empty:
                    df.reset_index(drop=True, inplace=True)
                    df.to_excel(writer, sheet_name=position, index=False)
        print(f"Data exported to {filepath}")
    except Exception as e:
        print(f"Error exporting to Excel: {e}")


def display_data_summary(positions_data):
    """
    Display a summary of the extracted data.
    """
    print("\nDATA SUMMARY:")
    print("=" * 80)

    for position, df in positions_data.items():
        print(f"\n{position} POSITION DATA ({len(df)} teams):")
        print("-" * 40)
        if not df.empty:
            display_cols = ['Team', 'PTS', 'FG%', 'FT%', '3PM', 'REB', 'AST', 'STL', 'BLK', 'TO']
            available_cols = [col for col in display_cols if col in df.columns]
            if available_cols:
                print(df[available_cols].head().to_string(index=False))


if __name__ == "__main__":
    defense_data = scrape_hashtag_basketball_table4()

    if defense_data:
        display_data_summary(defense_data)
        export_to_excel(defense_data)
    else:
        print("No data was scraped successfully")
