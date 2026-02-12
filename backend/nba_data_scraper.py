"""
NBA Data Scraper - Fetches and caches NBA data from Basketball Reference
Runs daily via GitHub Actions
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from basketball_reference_scraper.teams import get_roster
from basketball_reference_scraper.players import get_game_logs
from basketball_reference_scraper.seasons import get_schedule
import time

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

def safe_api_call(func, *args, **kwargs):
    """Wrapper for API calls with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            time.sleep(3.5)  # Rate limiting - Basketball Reference: 20 req/min = 3s between calls
            return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

def scrape_all_teams():
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
    df.to_csv('cached_all_teams.csv', index=False)

    with open('cached_all_teams.json', 'w') as f:
        json.dump(all_teams, f)

    print(f"Cached {len(all_teams)} teams")
    return all_teams

def scrape_active_players_from_rosters(season=2026):
    """Get active players by fetching team rosters from Basketball Reference"""
    print("Fetching active players from team rosters...")

    active_players = []
    player_info_list = []

    for team_abbr in NBA_TEAMS:
        team_info = NBA_TEAM_INFO[team_abbr]
        team_id = team_info['id']
        team_name = team_info['full_name']

        print(f"  Fetching roster for {team_abbr}...")

        try:
            roster_df = safe_api_call(get_roster, team_abbr, season)

            for _, row in roster_df.iterrows():
                # Basketball Reference doesn't provide player IDs easily
                # We'll use the player name as identifier
                player_name = row['PLAYER']

                player_info = {
                    'player_id': hash(player_name),  # Generate a hash as ID
                    'player_name': player_name,
                    'team_id': team_id,
                    'team_abbreviation': team_abbr,
                    'team_name': team_name,
                    'position': row.get('POS', ''),
                    'jersey': row.get('NUMBER', ''),
                    'height': row.get('HEIGHT', ''),
                    'weight': row.get('WEIGHT', '')
                }
                player_info_list.append(player_info)

                # Also create player dict for compatibility
                active_players.append({
                    'id': hash(player_name),
                    'full_name': player_name,
                    'first_name': player_name.split()[0] if ' ' in player_name else '',
                    'last_name': player_name.split()[-1] if ' ' in player_name else player_name
                })

        except Exception as e:
            print(f"    Error: {e}")
            continue

    # Save player info
    df_info = pd.DataFrame(player_info_list)
    df_info.to_csv('cached_player_info.csv', index=False)

    # Save active players list
    df_players = pd.DataFrame(active_players)
    df_players.to_csv('cached_all_players.csv', index=False)

    with open('cached_all_players.json', 'w') as f:
        json.dump(active_players, f)

    print(f"Cached {len(active_players)} active players from team rosters")

    return active_players, player_info_list

def scrape_player_gamelogs(player_list, season=2026, last_n_days=30):
    """Cache recent game logs for all active players"""
    print(f"Fetching recent game logs (last {last_n_days} days)...")
    all_gamelogs = []

    # Calculate date range for game logs
    end_date = datetime.now()
    start_date = end_date - timedelta(days=last_n_days)

    for i, player in enumerate(player_list):
        try:
            player_name = player['full_name']
            print(f"  {i+1}/{len(player_list)}: {player_name}")

            # Get game logs for the date range
            df = safe_api_call(
                get_game_logs,
                player_name,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                playoffs=False
            )

            if df is not None and not df.empty:
                df['PLAYER_NAME'] = player_name
                all_gamelogs.append(df)

        except Exception as e:
            print(f"    Error: {e}")
            continue

    if all_gamelogs:
        combined_df = pd.concat(all_gamelogs, ignore_index=True)
        combined_df.to_csv('cached_player_gamelogs.csv', index=False)
        print(f"Cached game logs for {len(all_gamelogs)} players")

    return all_gamelogs

def scrape_todays_games(season=2026):
    """Cache today's NBA schedule"""
    print("Fetching today's games...")
    today = datetime.now().strftime("%Y-%m-%d")

    try:
        # Get full season schedule
        schedule_df = safe_api_call(get_schedule, season, playoffs=False)

        # Filter for today's games
        schedule_df['DATE'] = pd.to_datetime(schedule_df['DATE'])
        todays_games = schedule_df[schedule_df['DATE'].dt.strftime('%Y-%m-%d') == today]

        if not todays_games.empty:
            todays_games.to_csv('cached_todays_games.csv', index=False)

            games_list = todays_games.to_dict('records')
            with open('cached_todays_games.json', 'w') as f:
                json.dump(games_list, f, default=str)

            print(f"Cached {len(todays_games)} games for {today}")
        else:
            print("No games today")
            # Create empty files
            pd.DataFrame().to_csv('cached_todays_games.csv', index=False)
            with open('cached_todays_games.json', 'w') as f:
                json.dump([], f)

    except Exception as e:
        print(f"Error fetching today's games: {e}")
        # Create empty files on error
        pd.DataFrame().to_csv('cached_todays_games.csv', index=False)
        with open('cached_todays_games.json', 'w') as f:
            json.dump([], f)

def main():
    """Main scraping workflow - ACTIVE PLAYERS ONLY"""
    print("="*60)
    print("NBA DATA SCRAPER - Basketball Reference")
    print("="*60)

    start_time = time.time()
    current_season = 2026  # Update this for the current season

    # 1. Scrape teams
    all_teams = scrape_all_teams()

    # 2. Get active players from team rosters
    active_players, player_info_list = scrape_active_players_from_rosters(season=current_season)

    # 3. Scrape game logs for active players (last 30 days)
    scrape_player_gamelogs(active_players, season=current_season, last_n_days=30)

    # 4. Scrape today's games
    scrape_todays_games(season=current_season)

    # 5. Create metadata file
    metadata = {
        'last_updated': datetime.now().isoformat(),
        'active_players': len(active_players),
        'total_teams': len(all_teams),
        'season': current_season,
        'source': 'Basketball Reference'
    }

    with open('cache_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_time
    print("="*60)
    print(f"SCRAPING COMPLETE! Took {elapsed/60:.1f} minutes")
    print(f"Active players: {len(active_players)}")
    print("="*60)

if __name__ == "__main__":
    main()
