"""
NBA Data Scraper - Fetches and caches NBA data from Basketball Reference
Runs daily via GitHub Actions - Uses direct web scraping
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
import re

# NBA team abbreviations for Basketball Reference
NBA_TEAMS = [
    'ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
    'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
    'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
]

NBA_TEAM_INFO = {
    'ATL': {'id': 1610612737, 'full_name': 'Atlanta Hawks'},
    'BOS': {'id': 1610612738, 'full_name': 'Boston Celtics'},
    'BRK': {'id': 1610612751, 'full_name': 'Brooklyn Nets'},
    'CHO': {'id': 1610612766, 'full_name': 'Charlotte Hornets'},
    'CHI': {'id': 1610612741, 'full_name': 'Chicago Bulls'},
    'CLE': {'id': 1610612739, 'full_name': 'Cleveland Cavaliers'},
    'DAL': {'id': 1610612742, 'full_name': 'Dallas Mavericks'},
    'DEN': {'id': 1610612743, 'full_name': 'Denver Nuggets'},
    'DET': {'id': 1610612765, 'full_name': 'Detroit Pistons'},
    'GSW': {'id': 1610612744, 'full_name': 'Golden State Warriors'},
    'HOU': {'id': 1610612745, 'full_name': 'Houston Rockets'},
    'IND': {'id': 1610612754, 'full_name': 'Indiana Pacers'},
    'LAC': {'id': 1610612746, 'full_name': 'LA Clippers'},
    'LAL': {'id': 1610612747, 'full_name': 'Los Angeles Lakers'},
    'MEM': {'id': 1610612763, 'full_name': 'Memphis Grizzlies'},
    'MIA': {'id': 1610612748, 'full_name': 'Miami Heat'},
    'MIL': {'id': 1610612749, 'full_name': 'Milwaukee Bucks'},
    'MIN': {'id': 1610612750, 'full_name': 'Minnesota Timberwolves'},
    'NOP': {'id': 1610612740, 'full_name': 'New Orleans Pelicans'},
    'NYK': {'id': 1610612752, 'full_name': 'New York Knicks'},
    'OKC': {'id': 1610612760, 'full_name': 'Oklahoma City Thunder'},
    'ORL': {'id': 1610612753, 'full_name': 'Orlando Magic'},
    'PHI': {'id': 1610612755, 'full_name': 'Philadelphia 76ers'},
    'PHO': {'id': 1610612756, 'full_name': 'Phoenix Suns'},
    'POR': {'id': 1610612757, 'full_name': 'Portland Trail Blazers'},
    'SAC': {'id': 1610612758, 'full_name': 'Sacramento Kings'},
    'SAS': {'id': 1610612759, 'full_name': 'San Antonio Spurs'},
    'TOR': {'id': 1610612761, 'full_name': 'Toronto Raptors'},
    'UTA': {'id': 1610612762, 'full_name': 'Utah Jazz'},
    'WAS': {'id': 1610612764, 'full_name': 'Washington Wizards'}
}

# Headers to mimic a browser request
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def safe_request(url, max_retries=3):
    """Make a safe HTTP request with retry logic and rate limiting"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            time.sleep(3.5)  # Rate limiting - Basketball Reference: 20 req/min
            return response
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
    return None

def scrape_team_roster(team_abbr, season=2026):
    """Scrape team roster from Basketball Reference"""
    url = f"https://www.basketball-reference.com/teams/{team_abbr}/{season}.html"
    print(f"  Fetching: {url}")

    try:
        response = safe_request(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the roster table
        roster_table = soup.find('table', {'id': 'roster'})
        if not roster_table:
            print(f"  No roster table found for {team_abbr}")
            return []

        players = []
        rows = roster_table.find('tbody').find_all('tr')

        for row in rows:
            # Skip header rows
            if row.get('class') and 'thead' in row.get('class'):
                continue

            cols = row.find_all(['th', 'td'])
            if len(cols) < 6:
                continue

            # Extract player data
            player_link = cols[1].find('a')
            if not player_link:
                continue

            player_name = player_link.text.strip()
            player_url = player_link.get('href', '')

            # Extract player ID from URL (/players/c/curryst01.html)
            player_id = None
            if player_url:
                match = re.search(r'/players/\w/(\w+)\.html', player_url)
                if match:
                    player_id = match.group(1)

            player_data = {
                'player_name': player_name,
                'player_id': player_id,
                'number': cols[0].text.strip() if len(cols) > 0 else '',
                'position': cols[2].text.strip() if len(cols) > 2 else '',
                'height': cols[3].text.strip() if len(cols) > 3 else '',
                'weight': cols[4].text.strip() if len(cols) > 4 else '',
                'birth_date': cols[5].text.strip() if len(cols) > 5 else '',
                'experience': cols[7].text.strip() if len(cols) > 7 else '',
                'college': cols[8].text.strip() if len(cols) > 8 else ''
            }
            players.append(player_data)

        return players

    except Exception as e:
        print(f"  Error scraping roster for {team_abbr}: {e}")
        return []

def scrape_schedule(season=2025):
    """Scrape NBA schedule from Basketball Reference
    Note: season parameter is the start year (2025 for 2025-26 season)
    """
    # Basketball Reference breaks schedule into months
    # We'll scrape the full season schedule
    months = ['october', 'november', 'december', 'january', 'february', 'march', 'april', 'may', 'june']

    all_games = []

    for month in months:
        url = f"https://www.basketball-reference.com/leagues/NBA_{season + 1}_games-{month}.html"
        print(f"  Fetching schedule for {month}...")

        try:
            response = safe_request(url)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the schedule table
            schedule_table = soup.find('table', {'id': 'schedule'})
            if not schedule_table:
                print(f"    No games found for {month}")
                continue

            rows = schedule_table.find('tbody').find_all('tr')

            for row in rows:
                # Skip header rows
                if row.get('class') and 'thead' in row.get('class'):
                    continue

                cols = row.find_all(['th', 'td'])
                if len(cols) < 7:
                    continue

                # Extract game data
                date_str = cols[0].text.strip()
                if not date_str:
                    continue

                # Parse teams
                visitor_team = cols[2].text.strip() if len(cols) > 2 else ''
                visitor_pts = cols[3].text.strip() if len(cols) > 3 else ''
                home_team = cols[4].text.strip() if len(cols) > 4 else ''
                home_pts = cols[5].text.strip() if len(cols) > 5 else ''

                # Box score link to get more info
                box_score_link = cols[6].find('a') if len(cols) > 6 else None
                game_id = None
                if box_score_link:
                    href = box_score_link.get('href', '')
                    match = re.search(r'/boxscores/(\w+)\.html', href)
                    if match:
                        game_id = match.group(1)

                game_data = {
                    'date': date_str,
                    'visitor_team': visitor_team,
                    'visitor_pts': visitor_pts,
                    'home_team': home_team,
                    'home_pts': home_pts,
                    'game_id': game_id
                }
                all_games.append(game_data)

        except Exception as e:
            print(f"    Error scraping {month}: {e}")
            continue

    return all_games

def scrape_player_gamelog(player_id, player_name, season=2026):
    """Scrape player game log from Basketball Reference
    player_id: Basketball Reference player ID (e.g., 'curryst01')
    """
    if not player_id:
        return None

    url = f"https://www.basketball-reference.com/players/{player_id[0]}/{player_id}/gamelog/{season}"

    try:
        response = safe_request(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the game log table (pgl_basic for basic stats)
        gamelog_table = soup.find('table', {'id': 'pgl_basic'})
        if not gamelog_table:
            return None

        games = []
        rows = gamelog_table.find('tbody').find_all('tr')

        for row in rows:
            # Skip header rows and rows without data
            if row.get('class') and 'thead' in row.get('class'):
                continue
            if row.find('th', {'data-stat': 'reason'}):  # Skip "Did Not Play" rows
                continue

            cols = row.find_all(['th', 'td'])
            if len(cols) < 10:
                continue

            # Extract game stats
            game_data = {
                'player_name': player_name,
                'date': row.find('td', {'data-stat': 'date_game'}).text.strip() if row.find('td', {'data-stat': 'date_game'}) else '',
                'age': row.find('td', {'data-stat': 'age'}).text.strip() if row.find('td', {'data-stat': 'age'}) else '',
                'team': row.find('td', {'data-stat': 'team_id'}).text.strip() if row.find('td', {'data-stat': 'team_id'}) else '',
                'opponent': row.find('td', {'data-stat': 'opp_id'}).text.strip() if row.find('td', {'data-stat': 'opp_id'}) else '',
                'game_result': row.find('td', {'data-stat': 'game_result'}).text.strip() if row.find('td', {'data-stat': 'game_result'}) else '',
                'mp': row.find('td', {'data-stat': 'mp'}).text.strip() if row.find('td', {'data-stat': 'mp'}) else '',
                'fg': row.find('td', {'data-stat': 'fg'}).text.strip() if row.find('td', {'data-stat': 'fg'}) else '',
                'fga': row.find('td', {'data-stat': 'fga'}).text.strip() if row.find('td', {'data-stat': 'fga'}) else '',
                'fg_pct': row.find('td', {'data-stat': 'fg_pct'}).text.strip() if row.find('td', {'data-stat': 'fg_pct'}) else '',
                'fg3': row.find('td', {'data-stat': 'fg3'}).text.strip() if row.find('td', {'data-stat': 'fg3'}) else '',
                'fg3a': row.find('td', {'data-stat': 'fg3a'}).text.strip() if row.find('td', {'data-stat': 'fg3a'}) else '',
                'fg3_pct': row.find('td', {'data-stat': 'fg3_pct'}).text.strip() if row.find('td', {'data-stat': 'fg3_pct'}) else '',
                'ft': row.find('td', {'data-stat': 'ft'}).text.strip() if row.find('td', {'data-stat': 'ft'}) else '',
                'fta': row.find('td', {'data-stat': 'fta'}).text.strip() if row.find('td', {'data-stat': 'fta'}) else '',
                'ft_pct': row.find('td', {'data-stat': 'ft_pct'}).text.strip() if row.find('td', {'data-stat': 'ft_pct'}) else '',
                'orb': row.find('td', {'data-stat': 'orb'}).text.strip() if row.find('td', {'data-stat': 'orb'}) else '',
                'drb': row.find('td', {'data-stat': 'drb'}).text.strip() if row.find('td', {'data-stat': 'drb'}) else '',
                'trb': row.find('td', {'data-stat': 'trb'}).text.strip() if row.find('td', {'data-stat': 'trb'}) else '',
                'ast': row.find('td', {'data-stat': 'ast'}).text.strip() if row.find('td', {'data-stat': 'ast'}) else '',
                'stl': row.find('td', {'data-stat': 'stl'}).text.strip() if row.find('td', {'data-stat': 'stl'}) else '',
                'blk': row.find('td', {'data-stat': 'blk'}).text.strip() if row.find('td', {'data-stat': 'blk'}) else '',
                'tov': row.find('td', {'data-stat': 'tov'}).text.strip() if row.find('td', {'data-stat': 'tov'}) else '',
                'pf': row.find('td', {'data-stat': 'pf'}).text.strip() if row.find('td', {'data-stat': 'pf'}) else '',
                'pts': row.find('td', {'data-stat': 'pts'}).text.strip() if row.find('td', {'data-stat': 'pts'}) else '',
                'game_score': row.find('td', {'data-stat': 'game_score'}).text.strip() if row.find('td', {'data-stat': 'game_score'}) else '',
            }
            games.append(game_data)

        return games

    except Exception as e:
        print(f"    Error scraping gamelog for {player_name}: {e}")
        return None

def scrape_all_teams(current_dir):
    """Cache all NBA teams"""
    print("Fetching all NBA teams...")

    all_teams = []
    for abbr in NBA_TEAMS:
        team_info = NBA_TEAM_INFO[abbr]
        all_teams.append({
            'id': team_info['id'],
            'full_name': team_info['full_name'],
            'abbreviation': abbr
        })

    df = pd.DataFrame(all_teams)
    csv_path = os.path.join(current_dir, 'cached_all_teams.csv')
    df.to_csv(csv_path, index=False)

    json_path = os.path.join(current_dir, 'cached_all_teams.json')
    with open(json_path, 'w') as f:
        json.dump(all_teams, f)

    print(f"Cached {len(all_teams)} teams")
    return all_teams

def scrape_active_players_from_rosters(current_dir, season=2026):
    """Get active players by scraping team rosters from Basketball Reference"""
    print("Scraping active players from team rosters...")

    active_players = []
    player_info_list = []

    for team_abbr in NBA_TEAMS:
        team_info = NBA_TEAM_INFO[team_abbr]
        team_id = team_info['id']
        team_name = team_info['full_name']

        print(f"  Fetching roster for {team_abbr}...")

        try:
            roster = scrape_team_roster(team_abbr, season)

            for player in roster:
                player_name = player['player_name']
                player_id = player['player_id']

                player_info = {
                    'player_id': player_id,
                    'player_name': player_name,
                    'team_id': team_id,
                    'team_abbreviation': team_abbr,
                    'team_name': team_name,
                    'position': player.get('position', ''),
                    'jersey': player.get('number', ''),
                    'height': player.get('height', ''),
                    'weight': player.get('weight', ''),
                    'birth_date': player.get('birth_date', ''),
                    'experience': player.get('experience', ''),
                    'college': player.get('college', '')
                }
                player_info_list.append(player_info)

                # Also create player dict for compatibility
                active_players.append({
                    'id': player_id,
                    'full_name': player_name,
                    'first_name': player_name.split()[0] if ' ' in player_name else '',
                    'last_name': player_name.split()[-1] if ' ' in player_name else player_name
                })

        except Exception as e:
            print(f"    Error: {e}")
            continue

    # Save player info
    df_info = pd.DataFrame(player_info_list)
    player_info_path = os.path.join(current_dir, 'cached_player_info.csv')
    df_info.to_csv(player_info_path, index=False)

    # Save active players list
    df_players = pd.DataFrame(active_players)
    players_csv_path = os.path.join(current_dir, 'cached_all_players.csv')
    df_players.to_csv(players_csv_path, index=False)

    players_json_path = os.path.join(current_dir, 'cached_all_players.json')
    with open(players_json_path, 'w') as f:
        json.dump(active_players, f)

    print(f"Cached {len(active_players)} active players from team rosters")

    return active_players, player_info_list

def scrape_player_gamelogs(current_dir, player_list, player_info_list, season=2026, last_n_days=30):
    """Cache recent game logs for all active players"""
    print(f"Scraping recent game logs (season {season})...")
    all_gamelogs = []

    # Calculate date range for filtering
    end_date = datetime.now()
    start_date = end_date - timedelta(days=last_n_days)

    for i, player in enumerate(player_list):
        try:
            player_name = player['full_name']
            player_id = player['id']

            print(f"  {i+1}/{len(player_list)}: {player_name} ({player_id})")

            # Get game logs from Basketball Reference
            games = scrape_player_gamelog(player_id, player_name, season)

            if games:
                # Convert to DataFrame and filter by date
                df = pd.DataFrame(games)

                # Parse dates and filter for recent games
                if 'date' in df.columns and not df.empty:
                    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df[df['date_parsed'] >= start_date]
                    df = df.drop('date_parsed', axis=1)

                if not df.empty:
                    all_gamelogs.append(df)

        except Exception as e:
            print(f"    Error: {e}")
            continue

    if all_gamelogs:
        combined_df = pd.concat(all_gamelogs, ignore_index=True)
        gamelogs_path = os.path.join(current_dir, 'cached_player_gamelogs.csv')
        combined_df.to_csv(gamelogs_path, index=False)
        print(f"Cached game logs for {len(all_gamelogs)} players")

    return all_gamelogs

def scrape_todays_games(current_dir, season=2025):
    """Cache today's NBA schedule"""
    print("Scraping today's games...")
    today = datetime.now().strftime("%Y-%m-%d")

    csv_path = os.path.join(current_dir, 'cached_todays_games.csv')
    json_path = os.path.join(current_dir, 'cached_todays_games.json')

    try:
        # Get full season schedule
        all_games = scrape_schedule(season)

        if not all_games:
            print("No schedule data found")
            pd.DataFrame().to_csv(csv_path, index=False)
            with open(json_path, 'w') as f:
                json.dump([], f)
            return

        # Convert to DataFrame
        schedule_df = pd.DataFrame(all_games)

        print(f"Schedule columns: {list(schedule_df.columns)}")
        print(f"Schedule shape: {schedule_df.shape}")

        # Parse dates and filter for today
        schedule_df['date_parsed'] = pd.to_datetime(schedule_df['date'], errors='coerce')
        todays_games = schedule_df[schedule_df['date_parsed'].dt.strftime('%Y-%m-%d') == today]

        if not todays_games.empty:
            todays_games = todays_games.drop('date_parsed', axis=1)
            todays_games.to_csv(csv_path, index=False)

            games_list = todays_games.to_dict('records')
            with open(json_path, 'w') as f:
                json.dump(games_list, f, default=str)

            print(f"Cached {len(todays_games)} games for {today}")
        else:
            print("No games today")
            # Create empty files
            pd.DataFrame().to_csv(csv_path, index=False)
            with open(json_path, 'w') as f:
                json.dump([], f)

    except Exception as e:
        print(f"Error fetching today's games: {e}")
        import traceback
        traceback.print_exc()
        # Create empty files on error
        pd.DataFrame().to_csv(csv_path, index=False)
        with open(json_path, 'w') as f:
            json.dump([], f)

def main():
    """Main scraping workflow"""
    print("="*60)
    print("NBA DATA SCRAPER - Basketball Reference (Direct Scraping)")
    print("="*60)

    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Saving files to: {current_dir}")

    start_time = time.time()
    current_season = 2026  # Update this for the current season

    # 1. Scrape teams
    all_teams = scrape_all_teams(current_dir)

    # 2. Get active players from team rosters
    active_players, player_info_list = scrape_active_players_from_rosters(current_dir, season=current_season)

    # 3. Scrape game logs for active players (last 30 days)
    scrape_player_gamelogs(current_dir, active_players, player_info_list, season=current_season, last_n_days=30)

    # 4. Scrape today's games
    scrape_todays_games(current_dir, season=2025)  # 2025-26 season

    # 5. Create metadata file
    metadata = {
        'last_updated': datetime.now().isoformat(),
        'active_players': len(active_players),
        'total_teams': len(all_teams),
        'season': current_season,
        'source': 'Basketball Reference (Direct Scraping)'
    }

    metadata_path = os.path.join(current_dir, 'cache_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_time
    print("="*60)
    print(f"SCRAPING COMPLETE! Took {elapsed/60:.1f} minutes")
    print(f"Active players: {len(active_players)}")
    print("="*60)

if __name__ == "__main__":
    main()
