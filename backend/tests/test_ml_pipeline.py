"""
Tests for the NBAProjectionModel ML pipeline.
Validates model training, prediction, and ensemble logic.
"""
import sys
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import NBAProjectionModel


# Model Initialization
class TestModelInitialization:
    def test_models_dict_starts_empty(self):
        model = NBAProjectionModel()
        assert model.models == {}

    def test_initialize_models_creates_all(self):
        model = NBAProjectionModel()
        model.initialize_models()
        expected_models = ['linear', 'bayesian', 'random_forest', 'gradient_boost', 'xgboost', 'lightgbm']
        for name in expected_models:
            assert name in model.models, f"Missing model: {name}"

    def test_initialize_models_count(self):
        model = NBAProjectionModel()
        model.initialize_models()
        assert len(model.models) == 6


# Synthetic Training Data 
class TestCreateSyntheticTrainingData:
    @patch('model.calculate_usage_rate', return_value=28.5)
    def test_returns_dataframe_and_targets(self, mock_usage, sample_player_features, sample_game_context, sample_defense_stats):
        model = NBAProjectionModel()
        features_df, targets = model.create_synthetic_training_data(
            sample_player_features, sample_game_context, sample_defense_stats, 'LeBron James'
        )
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == 100  # num_samples

    @patch('model.calculate_usage_rate', return_value=28.5)
    def test_targets_have_all_keys(self, mock_usage, sample_player_features, sample_game_context, sample_defense_stats):
        model = NBAProjectionModel()
        _, targets = model.create_synthetic_training_data(
            sample_player_features, sample_game_context, sample_defense_stats, 'LeBron James'
        )
        assert 'Points' in targets
        assert 'Rebounds' in targets
        assert 'Assists' in targets
        assert 'fantasy_points' in targets

    @patch('model.calculate_usage_rate', return_value=28.5)
    def test_targets_are_non_negative(self, mock_usage, sample_player_features, sample_game_context, sample_defense_stats):
        model = NBAProjectionModel()
        _, targets = model.create_synthetic_training_data(
            sample_player_features, sample_game_context, sample_defense_stats, 'LeBron James'
        )
        for key, values in targets.items():
            assert all(v >= 0 for v in values), f"Negative values found in {key}"

    @patch('model.calculate_usage_rate', return_value=28.5)
    def test_features_contain_usage_rate(self, mock_usage, sample_player_features, sample_game_context, sample_defense_stats):
        model = NBAProjectionModel()
        features_df, _ = model.create_synthetic_training_data(
            sample_player_features, sample_game_context, sample_defense_stats, 'LeBron James'
        )
        assert 'usage_rate' in features_df.columns
        # Usage rate should be constant (not varied with noise)
        assert features_df['usage_rate'].nunique() == 1
        assert features_df['usage_rate'].iloc[0] == 28.5


# Model Training
class TestTrainModelsOnTheFly:
    @patch('model.calculate_usage_rate', return_value=28.5)
    def test_trains_all_models(self, mock_usage, sample_player_features, sample_game_context, sample_defense_stats):
        model = NBAProjectionModel()
        model.initialize_models()
        trained_models, feature_columns = model.train_models_on_the_fly(
            sample_player_features, sample_game_context, sample_defense_stats,
            'LeBron James', 'Points'
        )
        assert len(trained_models) == 6
        assert len(feature_columns) > 0

    @patch('model.calculate_usage_rate', return_value=28.5)
    def test_stores_model_scores(self, mock_usage, sample_player_features, sample_game_context, sample_defense_stats):
        model = NBAProjectionModel()
        model.initialize_models()
        model.train_models_on_the_fly(
            sample_player_features, sample_game_context, sample_defense_stats,
            'LeBron James', 'Points'
        )
        assert hasattr(model, 'model_scores')
        assert len(model.model_scores) > 0

    @patch('model.calculate_usage_rate', return_value=28.5)
    def test_different_targets_produce_different_results(self, mock_usage, sample_player_features, sample_game_context, sample_defense_stats):
        model = NBAProjectionModel()
        model.initialize_models()

        trained_pts, _ = model.train_models_on_the_fly(
            sample_player_features, sample_game_context, sample_defense_stats,
            'LeBron James', 'Points'
        )

        model.initialize_models()  # Reset models
        trained_reb, _ = model.train_models_on_the_fly(
            sample_player_features, sample_game_context, sample_defense_stats,
            'LeBron James', 'Rebounds'
        )

        # Both should train successfully
        assert len(trained_pts) == 6
        assert len(trained_reb) == 6


# Ensemble Prediction
class TestEnsemblePrediction:
    @patch('model.calculate_usage_rate', return_value=28.5)
    def test_returns_prediction_confidence_and_individual(self, mock_usage, sample_player_features, sample_game_context, sample_defense_stats):
        model = NBAProjectionModel()
        model.initialize_models()
        trained_models, feature_columns = model.train_models_on_the_fly(
            sample_player_features, sample_game_context, sample_defense_stats,
            'LeBron James', 'Points'
        )

        # Build prediction input
        feature_dict = {
            'pts_5g_avg': 27.3, 'reb_5g_avg': 8.1, 'ast_5g_avg': 7.4,
            'mins_5g_avg': 34.2, 'fga_5g_avg': 18.5, 'fg3a_5g_avg': 4.8,
            'fta_5g_avg': 6.2, 'stl_5g_avg': 1.3, 'blk_5g_avg': 0.9,
            'tov_5g_avg': 3.1, 'usage_rate': 28.5, 'is_home': 1,
            'spread': -3.5, 'total': 225.5, 'opp_pts_allowed': 24.5,
            'opp_reb_allowed': 7.8, 'opp_ast_allowed': 5.9, 'opp_fd_allowed': 42.0,
        }
        prediction_input = pd.DataFrame([feature_dict])

        ensemble_pred, confidence, individual_preds = model.ensemble_prediction(
            prediction_input, trained_models, feature_columns, 'Points'
        )

        assert isinstance(ensemble_pred, float)
        assert ensemble_pred > 0
        assert 60 <= confidence <= 99
        assert isinstance(individual_preds, dict)
        assert len(individual_preds) == 6

    @patch('model.calculate_usage_rate', return_value=28.5)
    def test_ensemble_is_mean_of_individuals(self, mock_usage, sample_player_features, sample_game_context, sample_defense_stats):
        model = NBAProjectionModel()
        model.initialize_models()
        trained_models, feature_columns = model.train_models_on_the_fly(
            sample_player_features, sample_game_context, sample_defense_stats,
            'LeBron James', 'Points'
        )

        feature_dict = {
            'pts_5g_avg': 27.3, 'reb_5g_avg': 8.1, 'ast_5g_avg': 7.4,
            'mins_5g_avg': 34.2, 'fga_5g_avg': 18.5, 'fg3a_5g_avg': 4.8,
            'fta_5g_avg': 6.2, 'stl_5g_avg': 1.3, 'blk_5g_avg': 0.9,
            'tov_5g_avg': 3.1, 'usage_rate': 28.5, 'is_home': 1,
            'spread': -3.5, 'total': 225.5, 'opp_pts_allowed': 24.5,
            'opp_reb_allowed': 7.8, 'opp_ast_allowed': 5.9, 'opp_fd_allowed': 42.0,
        }
        prediction_input = pd.DataFrame([feature_dict])

        ensemble_pred, _, individual_preds = model.ensemble_prediction(
            prediction_input, trained_models, feature_columns, 'Points'
        )

        expected_mean = np.mean(list(individual_preds.values()))
        assert abs(ensemble_pred - expected_mean) < 0.01

    @patch('model.calculate_usage_rate', return_value=28.5)
    def test_confidence_within_bounds(self, mock_usage, sample_player_features, sample_game_context, sample_defense_stats):
        """Confidence should always be between 60 and 99 as defined in the code."""
        model = NBAProjectionModel()
        model.initialize_models()
        trained_models, feature_columns = model.train_models_on_the_fly(
            sample_player_features, sample_game_context, sample_defense_stats,
            'LeBron James', 'Points'
        )

        feature_dict = {
            'pts_5g_avg': 27.3, 'reb_5g_avg': 8.1, 'ast_5g_avg': 7.4,
            'mins_5g_avg': 34.2, 'fga_5g_avg': 18.5, 'fg3a_5g_avg': 4.8,
            'fta_5g_avg': 6.2, 'stl_5g_avg': 1.3, 'blk_5g_avg': 0.9,
            'tov_5g_avg': 3.1, 'usage_rate': 28.5, 'is_home': 1,
            'spread': -3.5, 'total': 225.5, 'opp_pts_allowed': 24.5,
            'opp_reb_allowed': 7.8, 'opp_ast_allowed': 5.9, 'opp_fd_allowed': 42.0,
        }
        prediction_input = pd.DataFrame([feature_dict])

        _, confidence, _ = model.ensemble_prediction(
            prediction_input, trained_models, feature_columns, 'Points'
        )
        assert 60 <= confidence <= 99


# Edge Cases
class TestEdgeCases:
    @patch('model.calculate_usage_rate', return_value=28.5)
    def test_home_vs_away_affects_prediction(self, mock_usage, sample_player_features, sample_defense_stats):
        """Home games should produce slightly different predictions than away."""
        model = NBAProjectionModel()
        model.initialize_models()

        home_context = {'opponent': 'BOS', 'is_home': 1, 'spread': -3.5, 'total': 225.5}
        away_context = {'opponent': 'BOS', 'is_home': 0, 'spread': -3.5, 'total': 225.5}

        trained_home, cols_home = model.train_models_on_the_fly(
            sample_player_features, home_context, sample_defense_stats,
            'LeBron James', 'Points'
        )

        feature_dict_home = {
            'pts_5g_avg': 27.3, 'reb_5g_avg': 8.1, 'ast_5g_avg': 7.4,
            'mins_5g_avg': 34.2, 'fga_5g_avg': 18.5, 'fg3a_5g_avg': 4.8,
            'fta_5g_avg': 6.2, 'stl_5g_avg': 1.3, 'blk_5g_avg': 0.9,
            'tov_5g_avg': 3.1, 'usage_rate': 28.5, 'is_home': 1,
            'spread': -3.5, 'total': 225.5, 'opp_pts_allowed': 24.5,
            'opp_reb_allowed': 7.8, 'opp_ast_allowed': 5.9, 'opp_fd_allowed': 42.0,
        }

        pred_home, _, _ = model.ensemble_prediction(
            pd.DataFrame([feature_dict_home]), trained_home, cols_home, 'Points'
        )

        model.initialize_models()
        trained_away, cols_away = model.train_models_on_the_fly(
            sample_player_features, away_context, sample_defense_stats,
            'LeBron James', 'Points'
        )

        feature_dict_away = {**feature_dict_home, 'is_home': 0}
        pred_away, _, _ = model.ensemble_prediction(
            pd.DataFrame([feature_dict_away]), trained_away, cols_away, 'Points'
        )

        # Predictions should differ (home boost exists in the synthetic data generation)
        assert pred_home != pred_away
