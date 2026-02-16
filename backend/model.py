import os
import csv
from time import time
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import pickle
from datetime import datetime, timedelta
import warnings

# import nba_usage_rate_scraper
warnings.filterwarnings('ignore')

class NBAProjectionModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def initialize_models(self):
        """Initialize all the ML models"""
        self.models = {
            'linear': LinearRegression(),
            'bayesian': BayesianRidge(),
            'random_forest': RandomForestRegressor(n_estimators=50, random_state=42, max_depth=8),
            'gradient_boost': GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=6),
            'xgboost': xgb.XGBRegressor(n_estimators=50, random_state=42, max_depth=6, learning_rate=0.1),
            'lightgbm': lgb.LGBMRegressor(n_estimators=50, random_state=42, max_depth=6, learning_rate=0.1),
        }
    
    def create_synthetic_training_data(self, player_features, game_context, defense_stats, player_name):
        """
        Create synthetic training data with real usage rate
        """
        # Get real usage rate for the player
        real_usage_rate = calculate_usage_rate(player_name)
        
        # Base features from the current player
        base_features = {
            'pts_5g_avg': player_features.get('pts_roll_avg', 20),
            'reb_5g_avg': player_features.get('reb_roll_avg', 8),
            'ast_5g_avg': player_features.get('ast_roll_avg', 6),
            'mins_5g_avg': player_features.get('min_roll_avg', 32),
            'fga_5g_avg': player_features.get('fga_roll_avg', 15),
            'fg3a_5g_avg': player_features.get('fg3a_roll_avg', 5),
            'fta_5g_avg': player_features.get('fta_roll_avg', 6),
            'stl_5g_avg': player_features.get('stl_roll_avg', 1.2),
            'blk_5g_avg': player_features.get('blk_roll_avg', 0.8),
            'tov_5g_avg': player_features.get('tov_roll_avg', 2.5),
            'usage_rate': real_usage_rate,  # Use real usage rate instead of rolling average
            'is_home': game_context['is_home'],
            'spread': game_context['spread'],
            'total': game_context['total'],
            'opp_pts_allowed': defense_stats['opp_pts_allowed'],
            'opp_reb_allowed': defense_stats['opp_reb_allowed'],
            'opp_ast_allowed': defense_stats['opp_ast_allowed'],
            'opp_fd_allowed': defense_stats['opp_fd_allowed'],
        }
        
        # Create synthetic variations for training
        num_samples = 100
        features_list = []
        targets = {
            'Points': [],
            'Rebounds': [],
            'Assists': [],
            'fantasy_points': []
        }
        
        for i in range(num_samples):
            sample_features = {}
            
            # Add some noise to create variation (except usage rate which we keep stable)
            for key, value in base_features.items():
                if key in ['is_home', 'usage_rate']:  # Keep usage rate and home as is
                    sample_features[key] = value
                elif isinstance(value, (int, float)):
                    # Add 10-20% variation for numeric features
                    variation = np.random.uniform(0.8, 1.2)
                    sample_features[key] = value * variation
                else:
                    sample_features[key] = value
            
            features_list.append(sample_features)
            
            # Create synthetic targets with usage rate as key factor
            usage_factor = sample_features['usage_rate'] / 25.0  # Normalize to league average
            
            base_pts = sample_features['pts_5g_avg'] * np.random.uniform(0.9, 1.1) * usage_factor
            base_reb = sample_features['reb_5g_avg'] * np.random.uniform(0.9, 1.1)
            base_ast = sample_features['ast_5g_avg'] * np.random.uniform(0.9, 1.1) * usage_factor
            
            # Adjust for opponent defense
            opp_factor_pts = sample_features['opp_pts_allowed'] / 25.0
            opp_factor_reb = sample_features['opp_reb_allowed'] / 8.0
            opp_factor_ast = sample_features['opp_ast_allowed'] / 6.0
            
            # Adjust for game context
            home_boost = 1.05 if sample_features['is_home'] else 0.98
            total_factor = sample_features['total'] / 220.0
            
            # Final predictions with all factors
            predicted_pts = base_pts * opp_factor_pts * home_boost * total_factor * np.random.uniform(0.95, 1.05)
            predicted_reb = base_reb * opp_factor_reb * home_boost * np.random.uniform(0.95, 1.05)
            predicted_ast = base_ast * opp_factor_ast * home_boost * np.random.uniform(0.95, 1.05)
            
            targets['Points'].append(max(0, predicted_pts))
            targets['Rebounds'].append(max(0, predicted_reb))
            targets['Assists'].append(max(0, predicted_ast))
            
            # Fantasy points calculation
            fantasy_pts = (predicted_pts + 
                         1.2 * predicted_reb + 
                         1.5 * predicted_ast + 
                         3 * sample_features['stl_5g_avg'] + 
                         3 * sample_features['blk_5g_avg'] - 
                         sample_features['tov_5g_avg'])
            targets['fantasy_points'].append(max(0, fantasy_pts))
        
        features_df = pd.DataFrame(features_list)
        return features_df, targets
    
    def train_models_on_the_fly(self, player_features, game_context, defense_stats, player_name, target_name='fantasy_points'):
        """
        Train models using synthetic data with real usage rate
        """
        print(f"Training models for {target_name} using synthetic data...")
        
        # Create synthetic training data
        features_df, targets = self.create_synthetic_training_data(player_features, game_context, defense_stats, player_name)
        y = targets[target_name]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[target_name] = scaler
        
        trained_models = {}
        model_scores = {}  # Store R² scores
        
        for name, model in self.models.items():
            try:
                # Choose scaled or unscaled data
                if name in ['linear', 'bayesian']:
                    X_train_used, X_test_used = X_train_scaled, X_test_scaled
                else:
                    X_train_used, X_test_used = X_train, X_test
                
                # Train model
                model.fit(X_train_used, y_train)
                trained_models[name] = model
                
                # Calculate R² score (goodness of fit)
                score = model.score(X_test_used, y_test)
                model_scores[name] = score
                
                # Make prediction on test set for evaluation
                y_pred = model.predict(X_test_used)
                mae = mean_absolute_error(y_test, y_pred)
                
                print(f"  {name:15} - MAE: {mae:.2f}, R²: {score:.3f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
        
        # Store scores for confidence calculation
        self.model_scores = model_scores
        
        return trained_models, features_df.columns.tolist()

    def ensemble_prediction(self, features_df, models, feature_columns, target_name='fantasy_points'):
        """
        Make ensemble prediction with confidence based on model performance
        """
        predictions = []
        model_names = []
        
        prediction_features = features_df[feature_columns]
        
        for name, model in models.items():
            try:
                if name in ['linear', 'bayesian']:
                    X_scaled = self.scalers[target_name].transform(prediction_features)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(prediction_features)
                
                predictions.append(pred[0])
                model_names.append(name)
                
            except Exception as e:
                print(f"Error in {name} prediction: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models produced valid predictions")
        
        # Ensemble prediction
        ensemble_pred = np.mean(predictions)
        
        # Calculate confidence from model R² scores
        avg_r2 = np.mean([self.model_scores.get(name, 0.8) for name in model_names])
        
        # Also factor in model agreement
        pred_std = np.std(predictions)
        agreement_factor = 1 - min(pred_std / ensemble_pred, 0.3) if ensemble_pred > 0 else 0.85
        
        # Combined confidence: 70% from R², 30% from agreement
        confidence = (avg_r2 * 0.7 + agreement_factor * 0.3) * 100
        confidence = min(99, max(60, confidence))  # Between 60-99%
        
        individual_predictions = dict(zip(model_names, predictions))
        
        return ensemble_pred, confidence, individual_predictions

def get_player_info(player_name):
    """
    Get player's information from cached data instead of live API
    """
    try:
        # Load cached player info
        player_info_df = pd.read_csv('backend/cached_player_info.csv')
        
        # Clean player name for matching
        player_name_clean = player_name.upper().strip()
        
        # Try exact match first
        exact_match = player_info_df[player_info_df['player_name'].str.upper() == player_name_clean]
        if not exact_match.empty:
            row = exact_match.iloc[0]
            return {
                'id': str(row['player_id']),
                'team': row['team_abbreviation'],
                'team_name': row['team_name'],
                'position': row['position']
            }
        
        # Try partial match
        partial_match = player_info_df[player_info_df['player_name'].str.upper().str.contains(player_name_clean, na=False)]
        if not partial_match.empty:
            row = partial_match.iloc[0]
            print(f"Found partial match: {row['player_name']}")
            return {
                'id': str(row['player_id']),
                'team': row['team_abbreviation'],
                'team_name': row['team_name'],
                'position': row['position']
            }
        
        # Try last name match
        last_name = player_name_clean.split()[-1]
        last_name_match = player_info_df[player_info_df['player_name'].str.upper().str.contains(last_name, na=False)]
        if not last_name_match.empty:
            row = last_name_match.iloc[0]
            print(f"Found last name match: {row['player_name']}")
            return {
                'id': str(row['player_id']),
                'team': row['team_abbreviation'],
                'team_name': row['team_name'],
                'position': row['position']
            }
        
        print(f"Player '{player_name}' not found in cached data")
        return None
        
    except Exception as e:
        print(f"Error loading player info from cache: {e}")
        return None

def get_all_players_cached():
    """
    Get all players from cache (replaces players.get_players())
    """
    try:
        with open('backend/cached_all_players.json', 'r') as f:
            return json.load(f)
    except:
        # Fallback to CSV
        df = pd.read_csv('backend/cached_all_players.csv')
        return df.to_dict('records')

def get_player_position(player_name):
    """
    Retrieves player's specific position from CSV file scraped from Basketball Reference.
    Falls back to cached_player_info.csv if player not found.
    """
    player_positions = {}
    try:
        with open("backend/players_positions.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header row (Player,Position)
            for row in reader:
                if len(row) >= 2:
                    name, position = row[0], row[1]
                    player_positions[name] = position
    except FileNotFoundError:
        pass

    if player_name in player_positions:
        return player_positions[player_name]

    # Fallback: get position from cached_player_info.csv (BBRef roster data)
    try:
        player_info_df = pd.read_csv('backend/cached_player_info.csv')
        match = player_info_df[player_info_df['player_name'].str.upper() == player_name.upper().strip()]
        if match.empty:
            # Try partial match
            last_name = player_name.strip().split()[-1]
            match = player_info_df[player_info_df['player_name'].str.upper().str.contains(last_name.upper(), na=False)]
        if not match.empty:
            pos = str(match.iloc[0]['position'])
            # BBRef positions may be multi-position like "SG-SF"; take primary
            primary_pos = pos.split('-')[0].strip() if pos else 'SG'
            # Map BBRef position labels to standard abbreviations used by defense data
            pos_map = {'G': 'SG', 'F': 'SF', 'C': 'C', 'GF': 'SG', 'FG': 'SF', 'FC': 'PF', 'CF': 'C'}
            return pos_map.get(primary_pos, primary_pos)
    except Exception:
        pass

    print(f"Warning: Could not find position for {player_name}, defaulting to SG")
    return 'SG'

def calculate_usage_rate(player_name):
    """
    Get usage rate for a player from the scraped NBA usage data
    Returns the season-long usage rate percentage
    """
    try:
        # Load the latest usage data
        usage_df = load_latest_usage_data()
        
        if usage_df is None or usage_df.empty:
            print("No usage data available. Please run the usage scraper first.")
            return 25.0  # Default fallback
        
        # Clean player name for matching
        player_name_clean = player_name.upper().strip()
        
        # Try different matching strategies
        # Exact match first
        exact_match = usage_df[usage_df['PLAYER'].str.upper() == player_name_clean]
        if not exact_match.empty:
            usg_rate = exact_match.iloc[0]['USGPCT']
            print(f"Found usage rate for {player_name}: {usg_rate}%")
            return float(usg_rate)
        
        # Partial match (handles variations in name formatting)
        partial_match = usage_df[usage_df['PLAYER'].str.upper().str.contains(player_name_clean, na=False)]
        if not partial_match.empty:
            usg_rate = partial_match.iloc[0]['USGPCT']
            print(f"Found usage rate for {player_name} (partial match): {usg_rate}%")
            return float(usg_rate)
        
        # Last name match
        last_name = player_name_clean.split()[-1]
        last_name_match = usage_df[usage_df['PLAYER'].str.upper().str.contains(last_name, na=False)]
        if not last_name_match.empty:
            usg_rate = last_name_match.iloc[0]['USGPCT']
            print(f"Found usage rate for {player_name} (last name match): {usg_rate}%")
            return float(usg_rate)
        
        print(f"Warning: Could not find usage rate for {player_name}. Using default 20.0%")
        return 20.0  # League average fallback
        
    except Exception as e:
        print(f"Error calculating usage rate for {player_name}: {e}")
        return 25.0  # Default fallback

def load_latest_usage_data():
    """
    Load the latest usage data from CSV file
    """
    try:
        csv_file = 'backend/nba_usage_rates_latest.csv'
        if os.path.exists(csv_file):
            # Check if file less than 24 hours old
            file_time = datetime.fromtimestamp(os.path.getmtime(csv_file))
            if datetime.now() - file_time < timedelta(hours=24):
                usage_df = pd.read_csv(csv_file)
                print(f"Loaded usage data for {len(usage_df)} players")
                return usage_df
            else:
                print("Usage data is older than 24 hours. Consider re-scraping.")
                # Still return the data, but warn about age
                usage_df = pd.read_csv(csv_file)
                return usage_df
        else:
            print("No usage data file found. Please run the usage scraper.")
            return None
    except Exception as e:
        print(f"Error loading usage data: {e}")
        return None
    
def fetch_recent_gamelog(player_id, team_abbr, retries=1):
    """
    Fetch recent game logs from cached data instead of live API.
    Handles both BBRef string IDs and nba_api numeric IDs.
    """
    try:
        print(f"Loading game logs from cache for player {player_id}...")

        # Load cached game logs
        gamelogs_df = pd.read_csv('backend/cached_player_gamelogs.csv')

        # Try matching by Player_ID first (works if both are same format)
        player_logs = gamelogs_df[gamelogs_df['Player_ID'].astype(str) == str(player_id)]

        # Fallback: match by PLAYER_NAME if Player_ID format differs (BBRef vs nba_api)
        if player_logs.empty and 'PLAYER_NAME' in gamelogs_df.columns:
            try:
                player_info_df = pd.read_csv('backend/cached_player_info.csv')
                player_row = player_info_df[player_info_df['player_id'].astype(str) == str(player_id)]
                if not player_row.empty:
                    player_name = player_row.iloc[0]['player_name']
                    player_logs = gamelogs_df[gamelogs_df['PLAYER_NAME'] == player_name]
                    if not player_logs.empty:
                        print(f"Matched by PLAYER_NAME: {player_name}")
            except Exception:
                pass

        if not player_logs.empty:
            print(f"Found {len(player_logs)} cached games")
            return player_logs
        else:
            print(f"No cached games found for player {player_id}")
            return get_player_season_averages(player_id)

    except FileNotFoundError:
        print("Game log cache file not found, using season averages...")
        return get_player_season_averages(player_id)
    except Exception as e:
        print(f"Error loading game logs from cache: {e}")
        return get_player_season_averages(player_id)

def get_player_season_averages(player_id):
    """
    Get player season averages from cached usage data as fallback
    """
    try:
        # Load usage data
        usage_df = load_latest_usage_data()
        
        if usage_df is not None:
            # Get player name from cached players
            all_players = get_all_players_cached()
            player_name = next((p['full_name'] for p in all_players if p['id'] == player_id), None)
            
            if player_name:
                player_row = usage_df[usage_df['PLAYER'].str.contains(player_name, case=False, na=False)]
                
                if not player_row.empty:
                    # Create a fake gamelog with season averages
                    avg_stats = pd.DataFrame({
                        'GAME_DATE': [datetime.now().strftime("%Y-%m-%d")],
                        'PTS': [player_row['PTS'].values[0] if 'PTS' in player_row else 0],
                        'REB': [player_row['REB'].values[0] if 'REB' in player_row else 0],
                        'AST': [player_row['AST'].values[0] if 'AST' in player_row else 0],
                        'MIN': [player_row['MIN'].values[0] if 'MIN' in player_row else 0],
                        'FGA': [player_row.get('FGA', pd.Series([0])).values[0]],
                        'FG3A': [player_row.get('FG3A', pd.Series([0])).values[0]],
                        'FTA': [player_row.get('FTA', pd.Series([0])).values[0]],
                        'STL': [player_row.get('STL', pd.Series([0])).values[0]],
                        'BLK': [player_row.get('BLK', pd.Series([0])).values[0]],
                        'TOV': [player_row.get('TOV', pd.Series([0])).values[0]],
                    })
                    print(f"Using season averages for {player_name}")
                    return avg_stats
        
        print("Could not find season averages, using defaults")
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error getting season averages: {e}")
        return pd.DataFrame()

def calculate_rolling_features(gamelog_df, window=5):
    if gamelog_df.empty:
        return {}

    if 'GAME_DATE' in gamelog_df.columns:
        df_sorted = gamelog_df.sort_values('GAME_DATE').reset_index(drop=True)
    else:
        df_sorted = gamelog_df.reset_index(drop=True)
    
    numeric_columns = ['PTS', 'REB', 'AST', 'MIN', 'FGA', 'FG3A', 'FTA', 'STL', 'BLK', 'TOV', 'USAGE_RATE']
    available_columns = [col for col in numeric_columns if col in df_sorted.columns]
    
    if not available_columns:
        return {}
    
    rolling_avg = df_sorted[available_columns].rolling(window=window, min_periods=1).mean()
    
    features = {}
    for col in available_columns:
        feature_name = f"{col.lower()}_roll_avg"
        features[feature_name] = rolling_avg[col].iloc[-1] if len(rolling_avg) > 0 else 0
    
    return features
def get_player_team_id(player_id):
    """
    Get team ID for a player from cached data
    """
    try:
        player_info_df = pd.read_csv('backend/cached_player_info.csv')
        player_row = player_info_df[player_info_df['player_id'].astype(str) == str(player_id)]
        
        if not player_row.empty:
            team_id = int(player_row.iloc[0]['team_id'])
            return team_id if team_id != 0 else None
        else:
            print(f"Player {player_id} not found in cache")
            return None
            
    except Exception as e:
        print(f"Error getting team ID from cache: {e}")
        return None
def get_team_schedule(team_id):
    """
    Get today's game from cached schedule data
    """
    try:
        today = datetime.now().strftime("%Y-%m-%d")

        # Load cached games
        games_df = pd.read_csv('backend/cached_todays_games.csv')

        if games_df.empty:
            print(f"No games scheduled for today")
            return None

        # Primary: match by numeric team ID columns
        game = pd.DataFrame()
        if 'HOME_TEAM_ID' in games_df.columns and 'VISITOR_TEAM_ID' in games_df.columns:
            game = games_df[(games_df["HOME_TEAM_ID"] == team_id) | (games_df["VISITOR_TEAM_ID"] == team_id)]

        # Fallback: match by full team name
        if game.empty and 'home_team' in games_df.columns and 'visitor_team' in games_df.columns:
            try:
                teams_df = pd.read_csv('backend/cached_all_teams.csv')
                team_row = teams_df[teams_df['id'] == team_id]
                if not team_row.empty:
                    team_full_name = team_row.iloc[0]['full_name']
                    game = games_df[
                        (games_df['home_team'] == team_full_name) |
                        (games_df['visitor_team'] == team_full_name)
                    ]
            except Exception:
                pass

        if not game.empty:
            game_info = game.iloc[0]

            home_team_id = game_info.get("HOME_TEAM_ID", None)
            visitor_team_id = game_info.get("VISITOR_TEAM_ID", None)

            # If IDs are missing/NaN, derive from team names
            if pd.isna(home_team_id) or pd.isna(visitor_team_id):
                teams_df = pd.read_csv('backend/cached_all_teams.csv')
                name_to_id = dict(zip(teams_df['full_name'], teams_df['id']))
                if pd.isna(home_team_id):
                    home_team_id = name_to_id.get(game_info.get('home_team'), 0)
                if pd.isna(visitor_team_id):
                    visitor_team_id = name_to_id.get(game_info.get('visitor_team'), 0)

            # Handle NaN GAME_ID (BBRef has null game_id for future games)
            game_id = game_info.get("GAME_ID", None)
            if pd.isna(game_id):
                game_id = game_info.get("game_id", None)
            if pd.isna(game_id):
                game_id = ""

            return {
                "GAME_DATE": today,
                "GAME_ID": game_id,
                "HOME_TEAM_ID": int(home_team_id),
                "VISITOR_TEAM_ID": int(visitor_team_id)
            }
        else:
            print(f"No scheduled game found for team {team_id} on {today}")
            return None

    except FileNotFoundError:
        print("Today's games cache not found")
        return None
    except Exception as e:
        print(f"Error loading game schedule from cache: {e}")
        return None

def get_tonights_game_context(player_name, spread, total):
    """
    Get game context using cached data
    """
    # Get player info from cache
    player_info = get_player_info(player_name)
    if not player_info:
        print(f"Player '{player_name}' not found.")
        return None
    
    player_id = player_info['id']
    team_abbr = player_info['team']
    
    print(f"FOUND {player_name} on {team_abbr}")
    
    # Get the player's team ID
    team_id = get_player_team_id(player_id)
    if not team_id:
        return None
    
    # Get the scheduled game
    game_data = get_team_schedule(team_id)
    if not game_data:
        return None
    
    # Determine if home or away
    is_home = (game_data['HOME_TEAM_ID'] == team_id)
    opponent_team_id = game_data['VISITOR_TEAM_ID'] if is_home else game_data['HOME_TEAM_ID']
    
    # Get opponent abbreviation from cached teams
    try:
        teams_df = pd.read_csv('backend/cached_all_teams.csv')
        opponent_row = teams_df[teams_df['id'] == opponent_team_id]
        
        if not opponent_row.empty:
            opponent_team = opponent_row.iloc[0]['abbreviation']
        else:
            opponent_team = "UNK"
    except:
        opponent_team = "UNK"
    
    context = {
        'opponent': opponent_team,
        'is_home': 1 if is_home else 0,
        'spread': spread,
        'total': total,
    }
    
    print(f"Game context for {team_abbr}: {context}")
    return context
def get_team_full_name(abbreviation):
    """
    Map team abbreviations to full names used by FantasyPros
    """
    team_map = {
        'ATL': ['Atlanta Hawks', 'Hawks'],
        'BOS': ['Boston Celtics', 'Celtics'],
        'BKN': ['Brooklyn Nets', 'Nets'],
        'CHA': ['Charlotte Hornets', 'Hornets'],
        'CHI': ['Chicago Bulls', 'Bulls'],
        'CLE': ['Cleveland Cavaliers', 'Cavaliers'],
        'DAL': ['Dallas Mavericks', 'Mavericks'],
        'DEN': ['Denver Nuggets', 'Nuggets'],
        'DET': ['Detroit Pistons', 'Pistons'],
        'GSW': ['Golden State Warriors', 'Warriors'],
        'HOU': ['Houston Rockets', 'Rockets'],
        'IND': ['Indiana Pacers', 'Pacers'],
        'LAC': ['LA Clippers', 'Clippers'],
        'LAL': ['Los Angeles Lakers', 'Lakers'],
        'MEM': ['Memphis Grizzlies', 'Grizzlies'],
        'MIA': ['Miami Heat', 'Heat'],
        'MIL': ['Milwaukee Bucks', 'Bucks'],
        'MIN': ['Minnesota Timberwolves', 'Timberwolves'],
        'NOP': ['New Orleans Pelicans', 'Pelicans'],
        'NYK': ['New York Knicks', 'Knicks'],
        'OKC': ['Oklahoma City Thunder', 'Thunder'],
        'ORL': ['Orlando Magic', 'Magic'],
        'PHI': ['Philadelphia 76ers', '76ers'],
        'PHX': ['Phoenix Suns', 'Suns'],
        'POR': ['Portland Trail Blazers', 'Trail Blazers'],
        'SAC': ['Sacramento Kings', 'Kings'],
        'SAS': ['San Antonio Spurs', 'Spurs'],
        'TOR': ['Toronto Raptors', 'Raptors'],
        'UTA': ['Utah Jazz', 'Jazz'],
        'WAS': ['Washington Wizards', 'Wizards']
    }
    return team_map.get(abbreviation, [abbreviation])[0]

# --- Step 5: Get Opponent Defense Stats ---
def get_cached_defense_data():
    """
    Get defense data from Excel file with multiple sheets
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    excel_file = os.path.join(current_dir, 'nba_defense_data.xlsx')
    
    try:
        if os.path.exists(excel_file):
            positions = ['PG', 'SG', 'SF', 'PF', 'C']
            defense_data = {}
            
            for position in positions:
                defense_data[position] = pd.read_excel(excel_file, sheet_name=position)
            
            print(f"Loaded defense data for {len(defense_data)} positions from Excel")
            return defense_data
        else:
            print(f"Defense data file not found at {excel_file}")
            return {}
            
    except Exception as e:
        print(f"Error loading defense data: {e}")
        return {}

def get_opponent_defense_stats(opponent_team, player_position):
    """
    Get real opponent defense stats using cached Hashtag Basketball data
    """
    # Get cached defense data
    positions_data = get_cached_defense_data()
    
    if not positions_data:
        print("Failed to get defense data, using default stats")
        return get_default_defense_stats()
    
    # Get the data for the specific position
    position_data = positions_data.get(player_position)
    if position_data is None or position_data.empty:
        print(f"No data found for position {player_position}")
        return get_default_defense_stats()
    
    # Complete NBA team abbreviation mapping
    team_mapping = {
        # NBA API abbrev → Hashtag Basketball abbrev
        'ATL': 'ATL',
        'BOS': 'BOS',
        'BKN': 'BKN',
        'CHA': 'CHA',
        'CHI': 'CHI',
        'CLE': 'CLE',
        'DAL': 'DAL',
        'DEN': 'DEN',
        'DET': 'DET',
        'GSW': 'GS',   # Golden State Warriors
        'HOU': 'HOU',
        'IND': 'IND',
        'LAC': 'LAC',
        'LAL': 'LAL',
        'MEM': 'MEM',
        'MIA': 'MIA',
        'MIL': 'MIL',
        'MIN': 'MIN',
        'NOP': 'NO',   # New Orleans Pelicans
        'NYK': 'NY',   # New York Knicks
        'OKC': 'OKC',
        'ORL': 'ORL',
        'PHI': 'PHI',
        'PHX': 'PHO',  # Phoenix Suns
        'POR': 'POR',
        'SAC': 'SAC',
        'SAS': 'SA',   # San Antonio Spurs
        'TOR': 'TOR',
        # BBRef abbreviations
        'BRK': 'BKN',  # Brooklyn Nets
        'CHO': 'CHA',  # Charlotte Hornets
        'PHO': 'PHO',  # Phoenix Suns
        'UTA': 'UTA',
        'WAS': 'WAS'
    }
    
    # Try to map the abbreviation
    search_team = team_mapping.get(opponent_team, opponent_team)
    
    # Find the opponent team in the position data
    team_row = None
    for idx, row in position_data.iterrows():
        team_name = str(row.get('Team', '')).strip()
        
        # Check for exact match or partial match
        if (team_name == search_team or 
            team_name == opponent_team or
            search_team in team_name or
            opponent_team in team_name):
            team_row = row
            print(f"Found match: {team_name} for {opponent_team}")
            break
    
    if team_row is None:
        print(f"Could not find defense data for {opponent_team} vs {player_position}")
        print(f"Tried searching for: {search_team}, {opponent_team}")
        return get_default_defense_stats()
    
    # Extract the defensive stats
    try:
        defense_stats = {
            'opp_pts_allowed': float(str(team_row.get('PTS', 0)).split()[0]),
            'opp_reb_allowed': float(str(team_row.get('REB', 0)).split()[0]),
            'opp_ast_allowed': float(str(team_row.get('AST', 0)).split()[0]),
            'opp_fd_allowed': float(str(team_row.get('PTS', 0)).split()[0])  # Using PTS as proxy
        }
        print(f"Defense stats for {opponent_team} vs {player_position}: {defense_stats}")
        return defense_stats
    except Exception as e:
        print(f"Error extracting defense stats: {e}")
        return get_default_defense_stats()

def get_team_full_name(abbreviation):
    """
    Convert team abbreviation to full name for matching
    """
    team_mapping = {
        'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BRK': 'Brooklyn Nets', 'BKN': 'Brooklyn Nets',
        'CHI': 'Chicago Bulls', 'CHO': 'Charlotte Hornets', 'CHA': 'Charlotte Hornets', 
        'CLE': 'Cleveland Cavaliers', 'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets',
        'DET': 'Detroit Pistons', 'GSW': 'Golden State Warriors', 'GS': 'Golden State Warriors',
        'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers', 'LAC': 'LA Clippers',
        'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies', 'MIA': 'Miami Heat',
        'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves', 'NOP': 'New Orleans Pelicans',
        'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder', 'ORL': 'Orlando Magic',
        'PHI': 'Philadelphia 76ers', 'PHO': 'Phoenix Suns', 'PHX': 'Phoenix Suns',
        'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
        'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
    }
    return team_mapping.get(abbreviation.upper(), abbreviation)

def get_default_defense_stats():
    """
    Return default defense stats when real data is not available
    """
    return {
        'opp_pts_allowed': 25.0,
        'opp_reb_allowed': 8.0,
        'opp_ast_allowed': 6.0,
        'opp_stl_allowed': 1.5,
        'opp_blk_allowed': 1.0,
        'opp_to_forced': 3.0,
        'opp_fg_pct_allowed': 45.0,
        'opp_3pm_allowed': 2.5,
        'opp_fd_allowed': 45.0
    }

def safe_float(value):
    """Safely convert to float, handling any data type issues"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

# Function to analyze defense matchups for your model
def analyze_defense_matchup(player_team, opponent_team, player_position):
    """
    Analyze the defense matchup for a specific player
    """
    defense_stats = get_opponent_defense_stats(opponent_team, player_position)
    
    # Calculate matchup advantage/disadvantage
    # Lower values = better for offense (weaker defense)
    matchup_score = (
        defense_stats['opp_pts_allowed'] / 25.0 +  # Normalize to league average ~25
        defense_stats['opp_fg_pct_allowed'] / 45.0 +  # Normalize to league average ~45%
        defense_stats['opp_fd_allowed'] / 40.0 +  # Normalize to league average ~40 FD points
        defense_stats['opp_reb_allowed'] / 8.0 +  # Normalize to league average ~8
        defense_stats['opp_ast_allowed'] / 6.0    # Normalize to league average ~6
    ) / 5.0  # Average the factors
    
    # Invert so higher = better matchup
    matchup_rating = 1.0 / matchup_score if matchup_score > 0 else 1.0
    
    print(f"Defense matchup rating for {player_team} {player_position} vs {opponent_team}: {matchup_rating:.2f}")
    return matchup_rating

def safe_float(value):
    """Safely convert to float, handling any data type issues"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def get_default_defense_stats():
    """Fallback defense stats if scraping fails"""
    return {
        'opp_pts_allowed': 25.0,
        'opp_reb_allowed': 8.0,
        'opp_ast_allowed': 6.0,
        'opp_fd_allowed': 45.0,
    }

def create_complete_prediction(player_name, target_stat, spread, total):
    """
    Complete prediction pipeline with real usage rate integration
    """
    print(f"🚀 Creating ML prediction for {player_name}...")
    
    # Update usage data if needed
    # update_usage_data_if_needed()
    
    # 1. Get Player Info
    player_info = get_player_info(player_name)
    if not player_info:
        print(f"Player '{player_name}' not found.")
        return None
    
    player_id = player_info['id']
    team_abbr = player_info['team']
    
    # 2. Get Game Logs and Calculate Features
    gamelogs = fetch_recent_gamelog(player_id, team_abbr)
    if gamelogs.empty:
        print("No game logs found.")
        return None
        
    player_features = calculate_rolling_features(gamelogs)
    
    # 3. Get real usage rate
    real_usage_rate = calculate_usage_rate(player_name)
    player_features['usage_rate'] = real_usage_rate
    
    print(f"Player features - Usage Rate: {real_usage_rate}%")
    print(f"Other features: { {k: round(v, 2) for k, v in player_features.items() if k != 'usage_rate'} }")
    
    # 4. Get Game Context
    game_context = get_tonights_game_context(player_name, spread, total)
    
    # 5. Get Opponent Defense Stats
    position = get_player_position(player_name)
    defense_stats = get_opponent_defense_stats(game_context['opponent'], position)
    
    # 6. Initialize and Train ML Models
    ml_system = NBAProjectionModel()
    ml_system.initialize_models()
    
    # Train models using synthetic data with real usage rate
    trained_models, feature_columns = ml_system.train_models_on_the_fly(
        player_features, game_context, defense_stats, player_name, target_stat
    )
    
    # 7. Create feature vector for prediction
    final_feature_dict = {
        'pts_5g_avg': player_features.get('pts_roll_avg', 0),
        'reb_5g_avg': player_features.get('reb_roll_avg', 0),
        'ast_5g_avg': player_features.get('ast_roll_avg', 0),
        'mins_5g_avg': player_features.get('min_roll_avg', 0),
        'fga_5g_avg': player_features.get('fga_roll_avg', 0),
        'fg3a_5g_avg': player_features.get('fg3a_roll_avg', 0),
        'fta_5g_avg': player_features.get('fta_roll_avg', 0),
        'stl_5g_avg': player_features.get('stl_roll_avg', 0),
        'blk_5g_avg': player_features.get('blk_roll_avg', 0),
        'tov_5g_avg': player_features.get('tov_roll_avg', 0),
        'usage_rate': real_usage_rate,  # Use real usage rate
        'is_home': game_context['is_home'],
        'spread': game_context['spread'],
        'total': game_context['total'],
        'opp_pts_allowed': defense_stats['opp_pts_allowed'],
        'opp_reb_allowed': defense_stats['opp_reb_allowed'],
        'opp_ast_allowed': defense_stats['opp_ast_allowed'],
        'opp_fd_allowed': defense_stats['opp_fd_allowed'],
    }
    
    prediction_input = pd.DataFrame([final_feature_dict])
    
    # 8. Make ensemble prediction
    try:
        ensemble_pred, confidence, individual_preds = ml_system.ensemble_prediction(
            prediction_input, trained_models, feature_columns, target_stat
        )
        
        return {
            'player': player_name,
            'prediction': ensemble_pred,
            'confidence': confidence,
            'individual_predictions': individual_preds,
            'features': final_feature_dict,
            'game_context': game_context,
            'target_stat': target_stat,
            'usage_rate': real_usage_rate
        }
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def print_prediction_results(result):
    """Print formatted prediction results"""
    if not result:
        return
    
    print("\n" + "="*60)
    print(f"ML PREDICTION RESULTS: {result['player']} ({get_player_position(result['player'])})")
    print("="*60)
    
    target_name = result['target_stat'].replace('_', ' ').title()
    print(f"Projected {target_name}: {result['prediction']:.1f}")
    print(f"Confidence Range: ±{result['confidence']:.1f}")
    
    print(f"\nIndividual Model Predictions:")
    for model, pred in result['individual_predictions'].items():
        print(f"  {model:15}: {pred:.1f}")
    
    print(f"\nGame Context:")
    print(f"  Opponent: {result['game_context']['opponent']}")
    print(f"  Home/Away: {'Home' if result['game_context']['is_home'] else 'Away'}")
    print(f"  Spread: {result['game_context']['spread']}")
    print(f"  Total: {result['game_context']['total']}")
    
    print(f"\nKey Features:")
    features = result['features']
    print(f"  Recent PPG: {features['pts_5g_avg']:.1f}")
    print(f"  Recent RPG: {features['reb_5g_avg']:.1f}")
    print(f"  Recent APG: {features['ast_5g_avg']:.1f}")
    print(f"  Opponent PTS Allowed: {features['opp_pts_allowed']:.1f}")
    print(f"  Opponent REB Allowed: {features['opp_reb_allowed']:.1f}")
    print(f"  Opponent AST Allowed: {features['opp_ast_allowed']:.1f}")

# Main execution
if __name__ == "__main__":
    player_to_predict = "LeBron James"
    spread = -3.5
    total = 235.5

    # Choose what to predict
    target_options = ['Points', 'Rebounds', 'Assists']
    target_choice = 'Points'  # Change this to predict different stats
    
    print("🏀 NBA Player Projection System")
    print("Using Multiple ML Models: Linear, Bayesian, Random Forest, XGBoost, LightGBM, Neural Networks")
    print("="*70)
    
    result = create_complete_prediction(player_to_predict, target_choice, spread, total)
    
    if result:
        print_prediction_results(result)
    else:
        print("Prediction failed")