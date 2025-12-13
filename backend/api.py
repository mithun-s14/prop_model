# api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import create_complete_prediction

app = Flask(__name__)
CORS(app)  # Allow requests from React app

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        player_name = data.get('player_name')
        target_stat = data.get('target_stat')
        spread = data.get('spread')
        total = data.get('total')
        
        # Validate inputs
        if not all([player_name, target_stat, spread is not None, total is not None]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Call your prediction function
        result = create_complete_prediction(player_name, target_stat, spread, total)
        
        if result is None:
            return jsonify({'error': 'Prediction failed. Check player name or data availability.'}), 404
        
        # Format response to match frontend expectations
        response = {
            'player': result['player'],
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'target_stat': result['target_stat'],
            'usage_rate': result['usage_rate'],
            'game_context': result['game_context'],
            'individual_predictions': result['individual_predictions']
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)