"""
NBA Data Scraper - Fetches and caches all NBA API data needed for predictions
Runs daily via GitHub Actions
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import (
    commonplayerinfo, 
    playergamelog, 
    commonteamroster,
    scoreboardv2
)
import time

def safe_api_call(func, *args, **kwargs):
    """Wrapper for API calls with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            time.sleep(0.6)  # Rate limiting
            return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

def scrape_all_players():
    """Cache all NBA players"""
    print("Fetching all NBA players...")
    all_players = players.get_players()
    
    df = pd.DataFrame(all_players)
    df.to_csv('cached_all_players.csv', index=False)
    
    with open('cached_all_players.json', 'w') as f:
        json.dump(all_players, f)
    
    print(f"Cached {len(all_players)} players")
    return all_players

def scrape_all_teams():
    """Cache all NBA teams"""
    print("Fetching all NBA teams...")
    all_teams = teams.get_teams()
    
    df = pd.DataFrame(all_teams)
    df.to_csv('cached_all_teams.csv', index=False)
    
    with open('cached_all_teams.json', 'w') as f:
        json.dump(all_teams, f)
    
    print(f"Cached {len(all_teams)} teams")
    return all_teams

def scrape_active_players_from_rosters(season='2025-26'):
    """Get active players by fetching team rosters (much faster!)"""
    print("Fetching active players from team rosters...")
    
    all_teams = teams.get_teams()
    active_players = []
    player_info_list = []
    
    for team in all_teams:
        team_id = team['id']
        team_abbr = team['abbreviation']
        print(f"  Fetching roster for {team_abbr}...")
        
        try:
            roster = safe_api_call(
                commonteamroster.CommonTeamRoster,
                team_id=team_id,
                season=season
            )
            
            roster_df = roster.common_team_roster.get_data_frame()
            
            for _, row in roster_df.iterrows():
                player_info = {
                    'player_id': row['PLAYER_ID'],
                    'player_name': row['PLAYER'],
                    'team_id': team_id,
                    'team_abbreviation': team_abbr,
                    'team_name': team['full_name'],
                    'position': row['POSITION'],
                    'jersey': row['NUM'],
                    'height': row.get('HEIGHT', ''),
                    'weight': row.get('WEIGHT', '')
                }
                player_info_list.append(player_info)
                
                # Also create player dict for compatibility
                active_players.append({
                    'id': row['PLAYER_ID'],
                    'full_name': row['PLAYER'],
                    'first_name': row['PLAYER'].split()[0] if ' ' in row['PLAYER'] else '',
                    'last_name': row['PLAYER'].split()[-1] if ' ' in row['PLAYER'] else row['PLAYER']
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

def scrape_player_gamelogs(player_list, season='2025-26', last_n_games=10):
    """Cache recent game logs for all active players"""
    print(f"Fetching recent game logs (last {last_n_games} games)...")
    all_gamelogs = []
    
    for i, player in enumerate(player_list):
        try:
            player_id = player['id']
            player_name = player['full_name']
            print(f"  {i+1}/{len(player_list)}: {player_name}")
            
            gamelog = safe_api_call(
                playergamelog.PlayerGameLog,
                player_id=player_id,
                season=season,
                timeout=60
            )
            
            df = gamelog.get_data_frames()[0]
            
            if not df.empty:
                # Take only last N games
                df = df.head(last_n_games)
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

def scrape_todays_games():
    """Cache today's NBA schedule"""
    print("Fetching today's games...")
    today = datetime.now().strftime("%Y-%m-%d")
    
    try:
        board = safe_api_call(scoreboardv2.ScoreboardV2, game_date=today)
        df_games = board.game_header.get_data_frame()
        
        if not df_games.empty:
            df_games.to_csv('cached_todays_games.csv', index=False)
            
            games_list = df_games.to_dict('records')
            with open('cached_todays_games.json', 'w') as f:
                json.dump(games_list, f)
            
            print(f"Cached {len(df_games)} games for {today}")
        else:
            print("No games today")
            
    except Exception as e:
        print(f"Error fetching today's games: {e}")

def get_active_players(all_players, player_info_list):
    """Filter to only active players (have a team)"""
    active_ids = {p['player_id'] for p in player_info_list if p.get('team_id')}
    active_players = [p for p in all_players if p['id'] in active_ids]
    print(f"Found {len(active_players)} active players")
    return active_players

def main():
    """Main scraping workflow - ACTIVE PLAYERS ONLY"""
    print("="*60)
    print("NBA DATA SCRAPER - Active Players Only")
    print("="*60)
    
    start_time = time.time()
    
    # 1. Scrape teams
    all_teams = scrape_all_teams()
    
    # 2. Get active players from team rosters (MUCH FASTER!)
    active_players, player_info_list = scrape_active_players_from_rosters(season='2025-26')
    
    # 3. Scrape game logs for active players
    scrape_player_gamelogs(active_players, last_n_games=10)
    
    # 4. Scrape today's games
    scrape_todays_games()
    
    # 5. Create metadata file
    metadata = {
        'last_updated': datetime.now().isoformat(),
        'active_players': len(active_players),
        'total_teams': len(all_teams),
        'season': '2025-26'
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