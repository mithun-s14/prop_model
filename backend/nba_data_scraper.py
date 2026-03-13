"""
NBA Data Scraper - Fetches and caches NBA data from Basketball Reference
Runs daily via GitHub Actions - Uses Scrapling for HTTP requests
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import re
from scrapling.fetchers import Fetcher

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

def safe_request(url, max_retries=3):
    """Make a safe HTTP request with retry logic and rate limiting."""
    for attempt in range(max_retries):
        try:
            response = Fetcher.get(url, stealthy_headers=True, timeout=30)
            time.sleep(3.5)  # Rate limiting - Basketball Reference: 20 req/min
            return response
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    return None


def scrape_team_roster(team_abbr, season=2026):
    """Scrape team roster from Basketball Reference"""
    url = f"https://www.basketball-reference.com/teams/{team_abbr}/{season}.html"
    print(f"  Fetching: {url}")

    try:
        page = safe_request(url)

        roster_tables = page.css('table#roster')
        if not roster_tables:
            print(f"  No roster table found for {team_abbr}")
            return []

        roster_table = roster_tables[0]
        players = []

        for row in roster_table.css('tbody tr'):
            # Skip header rows
            if 'thead' in (row.attrib.get('class') or ''):
                continue

            cols = row.css('th, td')
            if len(cols) < 6:
                continue

            player_links = cols[1].css('a')
            if not player_links:
                continue

            player_link = player_links[0]
            player_name = (player_link.css('::text').get() or '').strip()
            player_href = player_link.attrib.get('href', '')

            player_id = None
            if player_href:
                match = re.search(r'/players/\w/(\w+)\.html', player_href)
                if match:
                    player_id = match.group(1)

            def col_text(idx):
                return (cols[idx].css('::text').get() or '').strip() if len(cols) > idx else ''

            player_data = {
                'player_name': player_name,
                'player_id': player_id,
                'number': col_text(0),
                'position': col_text(2),
                'height': col_text(3),
                'weight': col_text(4),
                'birth_date': col_text(5),
                'experience': col_text(7),
                'college': col_text(8),
            }
            players.append(player_data)

        return players

    except Exception as e:
        print(f"  Error scraping roster for {team_abbr}: {e}")
        return []


def scrape_schedule(season=2025):
    """Scrape NBA schedule from Basketball Reference.
    season parameter is the start year (2025 for 2025-26 season).
    """
    months = ['october', 'november', 'december', 'january', 'february', 'march', 'april', 'may', 'june']
    all_games = []

    for month in months:
        url = f"https://www.basketball-reference.com/leagues/NBA_{season + 1}_games-{month}.html"
        print(f"  Fetching schedule for {month}...")

        try:
            page = safe_request(url)

            schedule_tables = page.css('table#schedule')
            if not schedule_tables:
                print(f"    No games found for {month}")
                continue

            schedule_table = schedule_tables[0]

            for row in schedule_table.css('tbody tr'):
                if 'thead' in (row.attrib.get('class') or ''):
                    continue

                cols = row.css('th, td')
                if len(cols) < 7:
                    continue

                def col_text(idx):
                    return (cols[idx].css('::text').get() or '').strip() if len(cols) > idx else ''

                date_str = col_text(0)
                if not date_str:
                    continue

                visitor_team = col_text(2)
                visitor_pts = col_text(3)
                home_team = col_text(4)
                home_pts = col_text(5)

                game_id = None
                if len(cols) > 6:
                    box_link = cols[6].css('a')
                    if box_link:
                        href = box_link[0].attrib.get('href', '')
                        match = re.search(r'/boxscores/(\w+)\.html', href)
                        if match:
                            game_id = match.group(1)

                all_games.append({
                    'date': date_str,
                    'visitor_team': visitor_team,
                    'visitor_pts': visitor_pts,
                    'home_team': home_team,
                    'home_pts': home_pts,
                    'game_id': game_id,
                })

        except Exception as e:
            print(f"    Error scraping {month}: {e}")
            continue

    return all_games


def scrape_player_gamelog(player_id, player_name, season=2026):
    """Scrape player game log from Basketball Reference.
    player_id: Basketball Reference player ID (e.g., 'curryst01')
    """
    if not player_id:
        return None

    url = f"https://www.basketball-reference.com/players/{player_id[0]}/{player_id}/gamelog/{season}"

    try:
        page = safe_request(url)

        gamelog_tables = page.css('table#pgl_basic')
        if not gamelog_tables:
            return None

        gamelog_table = gamelog_tables[0]
        games = []

        for row in gamelog_table.css('tbody tr'):
            if 'thead' in (row.attrib.get('class') or ''):
                continue
            # Skip "Did Not Play" / "Inactive" rows
            if row.css('td[data-stat="reason"]'):
                continue

            cols = row.css('th, td')
            if len(cols) < 10:
                continue

            def stat(name):
                el = row.css(f'td[data-stat="{name}"]::text')
                return (el.get() or '').strip()

            game_data = {
                'PLAYER_NAME': player_name,
                'Player_ID': player_id,
                'GAME_DATE': stat('date_game'),
                'team': stat('team_id'),
                'opponent': stat('opp_id'),
                'game_result': stat('game_result'),
                'MIN': stat('mp'),
                'FGM': stat('fg'),
                'FGA': stat('fga'),
                'FG_PCT': stat('fg_pct'),
                'FG3M': stat('fg3'),
                'FG3A': stat('fg3a'),
                'FG3_PCT': stat('fg3_pct'),
                'FTM': stat('ft'),
                'FTA': stat('fta'),
                'FT_PCT': stat('ft_pct'),
                'OREB': stat('orb'),
                'DREB': stat('drb'),
                'REB': stat('trb'),
                'AST': stat('ast'),
                'STL': stat('stl'),
                'BLK': stat('blk'),
                'TOV': stat('tov'),
                'PF': stat('pf'),
                'PTS': stat('pts'),
                'game_score': stat('game_score'),
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
    df.to_csv(os.path.join(current_dir, 'cached_all_teams.csv'), index=False)

    with open(os.path.join(current_dir, 'cached_all_teams.json'), 'w') as f:
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

                player_info_list.append({
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
                    'college': player.get('college', ''),
                })

                active_players.append({
                    'id': player_id,
                    'full_name': player_name,
                    'first_name': player_name.split()[0] if ' ' in player_name else '',
                    'last_name': player_name.split()[-1] if ' ' in player_name else player_name,
                })

        except Exception as e:
            print(f"    Error: {e}")
            continue

    pd.DataFrame(player_info_list).to_csv(
        os.path.join(current_dir, 'cached_player_info.csv'), index=False
    )
    df_players = pd.DataFrame(active_players)
    df_players.to_csv(os.path.join(current_dir, 'cached_all_players.csv'), index=False)

    with open(os.path.join(current_dir, 'cached_all_players.json'), 'w') as f:
        json.dump(active_players, f)

    print(f"Cached {len(active_players)} active players from team rosters")
    return active_players, player_info_list


def scrape_player_gamelogs(current_dir, player_list, player_info_list, season=2026, last_n_days=30):
    """Cache recent game logs for all active players"""
    print(f"Scraping recent game logs (season {season})...")
    all_gamelogs = []

    end_date = datetime.now()
    start_date = end_date - timedelta(days=last_n_days)

    for i, player in enumerate(player_list):
        try:
            player_name = player['full_name']
            player_id = player['id']

            print(f"  {i+1}/{len(player_list)}: {player_name} ({player_id})")

            games = scrape_player_gamelog(player_id, player_name, season)

            if games:
                df = pd.DataFrame(games)

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

        if 'MIN' in combined_df.columns:
            def convert_minutes(val):
                try:
                    if isinstance(val, str) and ':' in val:
                        parts = val.split(':')
                        return round(float(parts[0]) + float(parts[1]) / 60, 1)
                    return float(val)
                except (ValueError, TypeError):
                    return 0.0
            combined_df['MIN'] = combined_df['MIN'].apply(convert_minutes)

        numeric_cols = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
                        'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'game_score']
        for col in numeric_cols:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)

        combined_df.to_csv(os.path.join(current_dir, 'cached_player_gamelogs.csv'), index=False)
        print(f"Cached game logs for {len(all_gamelogs)} players")

    return all_gamelogs


def scrape_todays_games(current_dir, season=2025):
    """Cache today's NBA schedule"""
    print("Scraping today's games...")
    today = datetime.now().strftime("%Y-%m-%d")

    csv_path = os.path.join(current_dir, 'cached_todays_games.csv')
    json_path = os.path.join(current_dir, 'cached_todays_games.json')

    try:
        all_games = scrape_schedule(season)

        if not all_games:
            print("No schedule data found")
            pd.DataFrame().to_csv(csv_path, index=False)
            with open(json_path, 'w') as f:
                json.dump([], f)
            return

        schedule_df = pd.DataFrame(all_games)

        print(f"Schedule columns: {list(schedule_df.columns)}")
        print(f"Schedule shape: {schedule_df.shape}")

        schedule_df['date_parsed'] = pd.to_datetime(schedule_df['date'], errors='coerce')
        todays_games = schedule_df[schedule_df['date_parsed'].dt.strftime('%Y-%m-%d') == today]

        if not todays_games.empty:
            todays_games = todays_games.drop('date_parsed', axis=1)

            name_to_id = {info['full_name']: info['id'] for info in NBA_TEAM_INFO.values()}
            todays_games = todays_games.copy()
            todays_games['HOME_TEAM_ID'] = todays_games['home_team'].map(name_to_id)
            todays_games['VISITOR_TEAM_ID'] = todays_games['visitor_team'].map(name_to_id)
            todays_games['GAME_ID'] = todays_games['game_id']

            todays_games.to_csv(csv_path, index=False)

            games_list = todays_games.to_dict('records')
            with open(json_path, 'w') as f:
                json.dump(games_list, f, default=str)

            print(f"Cached {len(todays_games)} games for {today}")
        else:
            print("No games today")
            pd.DataFrame().to_csv(csv_path, index=False)
            with open(json_path, 'w') as f:
                json.dump([], f)

    except Exception as e:
        print(f"Error fetching today's games: {e}")
        import traceback
        traceback.print_exc()
        pd.DataFrame().to_csv(csv_path, index=False)
        with open(json_path, 'w') as f:
            json.dump([], f)


def load_cached_players(current_dir):
    """Load active_players and player_info_list from cached CSV files"""
    df_players = pd.read_csv(os.path.join(current_dir, 'cached_all_players.csv'))
    active_players = df_players.to_dict('records')

    df_info = pd.read_csv(os.path.join(current_dir, 'cached_player_info.csv'))
    player_info_list = df_info.to_dict('records')

    print(f"Loaded {len(active_players)} active players from cache")
    return active_players, player_info_list


def main():
    """Main scraping workflow"""
    print("="*60)
    print("NBA DATA SCRAPER - Basketball Reference (Scrapling)")
    print("="*60)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Saving files to: {current_dir}")

    start_time = time.time()
    current_season = 2026

    # 1. Load active players from cached files
    active_players, player_info_list = load_cached_players(current_dir)

    # 2. Scrape recent game logs for all active players
    scrape_player_gamelogs(current_dir, active_players, player_info_list, season=current_season)

    # 3. Scrape today's games
    scrape_todays_games(current_dir, season=2025)  # 2025-26 season

    # 3. Create metadata file
    metadata = {
        'last_updated': datetime.now().isoformat(),
        'active_players': len(active_players),
        'season': current_season,
        'source': 'Basketball Reference (Scrapling)',
    }
    with open(os.path.join(current_dir, 'cache_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_time
    print("="*60)
    print(f"SCRAPING COMPLETE! Took {elapsed/60:.1f} minutes")
    print(f"Active players: {len(active_players)}")
    print("="*60)


if __name__ == "__main__":
    main()
