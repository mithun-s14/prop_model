"""create_complete_prediction must check for a scheduled game before doing any prediction work."""
import sys
import os
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import model
from model import create_complete_prediction, NoGameError

PLAYER = {'id': 'jamesle01', 'full_name': 'LeBron James', 'team': 'LAL'}


@patch('model.get_player_last15_features')
@patch('model.get_tonights_game_context', return_value=None)
@patch('model.get_player_info', return_value=PLAYER)
def test_no_game_raises_before_feature_work(_info, _ctx, last15):
    with pytest.raises(NoGameError, match='does not have a game today'):
        create_complete_prediction('LeBron James', 'Points')
    last15.assert_not_called()


@patch('model.get_tonights_game_context')
@patch('model.get_player_info', return_value=None)
def test_unknown_player_returns_none_without_game_lookup(_info, ctx):
    assert create_complete_prediction('Nobody', 'Points') is None
    ctx.assert_not_called()


@patch('model.get_player_last15_features', return_value=None)
@patch('model.get_tonights_game_context', return_value={'opponent': 'BOS', 'is_home': 1})
@patch('model.get_player_info', return_value=PLAYER)
def test_has_game_proceeds_to_features(_info, _ctx, last15):
    assert create_complete_prediction('LeBron James', 'Points') is None
    last15.assert_called_once()


def test_api_returns_404_on_no_game():
    pytest.importorskip('flask_cors')
    import api
    with patch.object(api, 'MODEL_LOADED', True), \
         patch.object(api, 'create_complete_prediction', side_effect=NoGameError('LeBron James does not have a game today.')):
        resp = api.app.test_client().post('/predict', json={'player_name': 'LeBron James', 'target_stat': 'Points'})
    assert resp.status_code == 404
    assert resp.get_json() == {'error': 'LeBron James does not have a game today.'}
