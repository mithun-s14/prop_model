import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import time

def scrape_nba_usage_rates():
    """
    Scrapes NBA player usage rates from all pages of the NBA stats table
    """
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Use webdriver-manager to handle ChromeDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    all_data = []  # Store data from all pages
    headers = []  # Store headers once
    
    try:
        # Navigate to the usage stats page
        print("Loading NBA usage stats page...")
        driver.get("https://www.nba.com/stats/players/usage")
        
        # Wait for the specific table to load
        print("Waiting for usage rates table to load...")
        wait = WebDriverWait(driver, 20)
        
        page_count = 0
        max_pages = 30  # Safety limit
        
        while page_count < max_pages:
            page_count += 1
            print(f"\n--- Processing page {page_count} ---")
            
            # Wait for the specific table with class Crom_table__p1iZz
            try:
                table = wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, "Crom_table__p1iZz"))
                )
            except TimeoutException:
                print("Table not found - may have reached the end")
                break
            
            print("Table found! Extracting data...")
            
            # Extract all rows from the table body
            rows = table.find_elements(By.TAG_NAME, "tr")
            print(f"Found {len(rows)} rows in the table")
            
            # Extract column headers only on first page
            if not headers:
                header_elements = table.find_elements(By.TAG_NAME, "th")
                for header in header_elements:
                    header_text = header.text.strip()
                    # If empty, try to get from field attribute
                    if not header_text:
                        field_name = header.get_attribute('field')
                        if field_name:
                            # Convert field name to readable format
                            if field_name == '__RANK':
                                header_text = 'RANK'
                            elif field_name == 'PLAYER_NAME':
                                header_text = 'PLAYER'
                            else:
                                header_text = field_name
                    headers.append(header_text)
                
                print(f"Table headers ({len(headers)}): {headers}")
            
            # Extract data from each row
            data = []
            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if cells:  # Skip header row
                    row_data = [cell.text.strip() for cell in cells]
                    # Check if we need to adjust for header count mismatch
                    if len(row_data) == len(headers):
                        data.append(row_data)
                    elif len(row_data) > len(headers):
                        # If more data columns than headers, truncate to match headers
                        data.append(row_data[:len(headers)])
                    else:
                        # If fewer data columns, pad with empty strings
                        padded_data = row_data + [''] * (len(headers) - len(row_data))
                        data.append(padded_data)
            
            print(f"Extracted {len(data)} data rows from page {page_count}")
            
            # Add data from this page to our collection
            all_data.extend(data)
            
            # Try to find and click the next page button with better waiting
            try:
                # Wait for the next button to be clickable
                next_button = wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[title="Next Page Button"]'))
                )
                
                # Check if we're on last page
                if next_button.get_attribute('disabled'):
                    print("Reached the last page!")
                    break
                
                # Click the next page button
                print("Clicking next page button...")
                driver.execute_script("arguments[0].click();", next_button)
                # Wait for the page to load
                print("Waiting for next page to load...")
                # Wait for the table to refresh
                time.sleep(2)
                # Wait for the table to be present and have data
                wait.until(lambda driver: len(driver.find_elements(By.CSS_SELECTOR, "table.Crom_table__p1iZz tbody tr")) > 0)
                
                print("Next page loaded successfully")
                
            except TimeoutException:
                print("Next page button not found or not clickable - assuming we're on the last page")
                break
            except NoSuchElementException:
                print("Next page button not found - assuming we're on the last page")
                break
            except Exception as e:
                print(f"Error navigating to next page: {e}")
                # Try one more time with a different approach
                try:
                    # Look for next page button
                    buttons = driver.find_elements(By.CSS_SELECTOR, "button")
                    next_btns = [btn for btn in buttons if "next" in btn.get_attribute('outerHTML').lower()]
                    if next_btns:
                        driver.execute_script("arguments[0].click();", next_btns[0])
                        time.sleep(3)
                        wait.until(lambda driver: len(driver.find_elements(By.CSS_SELECTOR, "table.Crom_table__p1iZz tbody tr")) > 0)
                    else:
                        break
                except:
                    break
        
        print(f"\nCompleted scraping {page_count} pages")
        print(f"Total players collected: {len(all_data)}")
        
        # Create DataFrame from all collected data
        if all_data:
            usage_df = pd.DataFrame(all_data, columns=headers)
            print(f"Successfully created DataFrame with {len(usage_df)} players")
            return usage_df
        else:
            print("No data collected from any page")
            return None
        
    except Exception as e:
        print(f"Error during scraping: {e}")
        return None
    finally:
        driver.quit()

def clean_usage_data(df):
    """
    Clean and process the usage data - keep only the main stat columns, not the rank columns
    """
    if df is None or df.empty:
        return df
    
    # Make a copy to avoid warnings
    cleaned_df = df.copy()
    
    # Remove empty columns
    cleaned_df = cleaned_df.loc[:, ~cleaned_df.columns.str.contains('^Unnamed')]
    
    # Filter out the _RANK columns - we only want the main stats
    main_columns = [col for col in cleaned_df.columns if not col.endswith('_RANK')]
    cleaned_df = cleaned_df[main_columns]
    
    # Clean column names
    cleaned_df.columns = [col.replace('%', 'PCT').replace(' ', '_').upper() for col in cleaned_df.columns]
    
    # Convert numeric columns - using raw string for regex
    numeric_columns = ['RANK', 'AGE', 'GP', 'W', 'L', 'MIN', 'USG_PCT', 'PCT_FGM', 'PCT_FGA', 
                      'PCT_FG3M', 'PCT_FG3A', 'PCT_FTM', 'PCT_FTA', 'PCT_OREB', 'PCT_DREB', 
                      'PCT_REB', 'PCT_AST', 'PCT_TOV', 'PCT_STL', 'PCT_BLK', 'PCT_BLKA', 
                      'PCT_PF', 'PCT_PFD', 'PCT_PTS']
    
    # Only use columns that actually exist in our dataframe
    available_numeric_cols = [col for col in numeric_columns if col in cleaned_df.columns]
    
    for col in available_numeric_cols:
        # Use raw string for regex to fix the escape sequence warning
        cleaned_df[col] = pd.to_numeric(
            cleaned_df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True), 
            errors='coerce'
        )
    
    print(f"Data cleaning completed. Kept {len(cleaned_df.columns)} main stat columns")
    return cleaned_df

def save_usage_data(usage_df):
    """
    Saves usage data to both CSV and pickle files
    """
    if usage_df is None or usage_df.empty:
        print("No data to save")
        return None, None
    
    # Clean the data first
    cleaned_df = clean_usage_data(usage_df)
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save files in the current directory (backend folder)
    csv_path = os.path.join(current_dir, 'nba_usage_rates_latest.csv')
    
    cleaned_df.to_csv(csv_path, index=False)
    print(f"Latest versions saved as '{csv_path}'")
    
    return 'nba_usage_rates_latest.csv', None

def cache_usage_data():
    """
    Main function to scrape and cache usage data from all pages
    """
    print("Starting NBA usage rate scraping...")
    
    # Scrape the data from all pages
    usage_data = scrape_nba_usage_rates()
    
    if usage_data is not None:
        print(f"Scraping successful! Found data for {len(usage_data)} players")
        print("\nData summary:")
        print(f"Columns: {list(usage_data.columns)}")
        print(f"Shape: {usage_data.shape}")
        
        # Save the data
        csv_file, pkl_file = save_usage_data(usage_data)
        
        print(f"\nScraping completed successfully!")
        print(f"CSV file: {csv_file}")
        
        # Display key columns for verification
        if not usage_data.empty:
            key_cols = ['PLAYER', 'TEAM', 'USG_PCT']
            available_cols = [col for col in key_cols if col in usage_data.columns]
            if available_cols:
                print(f"\nSample of players and usage rates:")
                sample_data = usage_data[available_cols].head(10)
                for _, row in sample_data.iterrows():
                    print(f"  {row['PLAYER']} ({row['TEAM']}): {row.get('USG_PCT', 'N/A')}%")
        
        return usage_data
    else:
        print("Scraping failed - no data retrieved")
        return None

# Run the scraper
if __name__ == "__main__":
    data = cache_usage_data()