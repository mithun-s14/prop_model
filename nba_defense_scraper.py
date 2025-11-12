from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from io import StringIO
import pandas as pd
import time
import pickle

def scrape_hashtag_basketball_table4():
    """
    Scrape NBA defense vs position data from Hashtag Basketball - specifically targeting Position data
    """
    # Set up Chrome options
    chrome_options = Options()
    # chrome_options.add_argument("--headless=new")  # Uncomment for headless
    chrome_options.add_experimental_option(
        "prefs", {"profile.managed_default_content_settings.images": 2}
    )
    
    driver = webdriver.Chrome(options=chrome_options)
    positions_data = {}
    
    try:
        # Navigate to the target website
        print("Loading Hashtag Basketball defense vs position page...")
        driver.get("https://hashtagbasketball.com/nba-defense-vs-position")
        
        # Wait for the page to load completely
        time.sleep(5)
        
        # Find ALL tables on the page
        print("Looking for all tables on the page...")
        all_tables = driver.find_elements(By.TAG_NAME, "table")
        print(f"Found {len(all_tables)} tables on the page")
        
        # Specifically target Table 4 (index 3) which has 150 rows
        if len(all_tables) >= 4:
            target_table = all_tables[3]  # Table 4 is at index 3
        else:
            print("'Position data' not found!")
            return positions_data
        
        # Get the table HTML and parse with pandas
        html_content = target_table.get_attribute('outerHTML')
        df_list = pd.read_html(StringIO(html_content))
        
        if df_list:
            main_df = df_list[0]
            print(f"'Position data' Table shape: {main_df.shape}")
            print(f"Raw columns: {main_df.columns.tolist()}")
            
            # Clean up column names - remove the "Sort: " prefix
            main_df.columns = [col.replace('Sort: ', '') for col in main_df.columns]
            print(f"Cleaned columns: {main_df.columns.tolist()}")
            
            # Display first few rows to verify
            print("\nFirst 3 rows of Table 4:")
            print(main_df.head(3).to_string(index=False))
            
            # Process data by position
            positions = ['PG', 'SG', 'SF', 'PF', 'C']
            
            for position in positions:
                try:
                    # Filter rows for this position
                    position_df = main_df[main_df['Position'] == position].copy()
                    
                    if not position_df.empty:
                        # Clean the data - extract main values and remove rank numbers
                        clean_position_df = clean_position_data(position_df)
                        positions_data[position] = clean_position_df
                    else:
                        print(f"No data found for position: {position}")
                        
                except Exception as e:
                    print(f"Error processing {position}: {str(e)}")
            
        else:
            print("Could not parse 'Position data' table")
            
    except Exception as e:
        print(f"Error during scraping: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        driver.quit()
    
    return positions_data

def clean_position_data(df):
    """
    Clean the position data by extracting main values and removing rank numbers
    """
    cleaned_df = df.copy()
    
    # Columns to clean (all except Position and Team)
    columns_to_clean = ['PTS', 'FG%', 'FT%', '3PM', 'REB', 'AST', 'STL', 'BLK', 'TO']
    
    for col in columns_to_clean:
        if col in cleaned_df.columns:
            # Extract just the main value (before the space and rank number)
            # Handle both string and numeric types
            cleaned_df[col] = cleaned_df[col].astype(str).str.split().str[0]
            
            # Convert back to numeric where appropriate
            if col not in ['FG%', 'FT%']:  # Keep percentages as strings for now
                try:
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                except:
                    pass
    
    # Clean team names - remove rank numbers
    if 'Team' in cleaned_df.columns:
        cleaned_df['Team'] = cleaned_df['Team'].astype(str).str.split().str[0]
    
    return cleaned_df

def save_data(positions_data, filename='nba_defense_data.pkl'):
    """
    Save the scraped data to a pickle file
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(positions_data, f)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data: {str(e)}")

def export_to_excel(positions_data, filename='nba_defense_data.xlsx'):
    """
    Export data to Excel with separate sheets for each position
    """
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for position, df in positions_data.items():
                if not df.empty:
                    # Reset index for cleaner output
                    df.reset_index(drop=True, inplace=True)
                    df.to_excel(writer, sheet_name=position, index=False)
        print(f"Data exported to {filename}")
    except Exception as e:
        print(f"Error exporting to Excel: {str(e)}")

def display_data_summary(positions_data):
    """
    Display a summary of the extracted data
    """
    print("\nDATA SUMMARY:")
    print("=" * 80)
    
    for position, df in positions_data.items():
        print(f"\n{position} POSITION DATA ({len(df)} teams):")
        print("-" * 40)
        if not df.empty:
            # Display key columns
            display_cols = ['Team', 'PTS', 'FG%', 'FT%', '3PM', 'REB', 'AST', 'STL', 'BLK', 'TO']
            available_cols = [col for col in display_cols if col in df.columns]
            
            if available_cols:
                print(df[available_cols].head().to_string(index=False))

# Main execution
if __name__ == "__main__":
    print("Starting NBA Defense vs Position scraper...")
    
    # Scrape the data
    defense_data = scrape_hashtag_basketball_table4()
    
    if defense_data:
        # Display summary
        display_data_summary(defense_data)
        
        # Save the data
        save_data(defense_data)
        
        # Export to Excel
        export_to_excel(defense_data)
        
        print("\nScraping completed successfully!")
        
        # Final summary
        print(f"\nFinal Results:")
        total_rows = sum(len(df) for df in defense_data.values())
        print(f"   Total rows extracted: {total_rows}")
        for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
            if pos in defense_data:
                df = defense_data[pos]
                print(f"   {pos}: {len(df)} teams")
    else:
        print("No data was scraped successfully")