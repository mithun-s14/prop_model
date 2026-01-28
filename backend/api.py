import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import traceback
import gc

# Try importing model
try:
    from model import create_complete_prediction
    MODEL_LOADED = True
    print("Model loaded successfully")
except Exception as e:
    MODEL_LOADED = False
    print(f"Warning: Could not load model - {e}")
    traceback.print_exc()

app = Flask(__name__)

# Configure CORS properly for production
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "https://prop-model.vercel.app",
            "https://*.vercel.app"  # Allow all Vercel preview deployments
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Add CORS headers manually as well (belt and suspenders approach)
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin:
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

def convert_to_json_serializable(obj):
    """Convert numpy/pandas types to native Python types"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight request
    if request.method == 'OPTIONS':
        return '', 204
    
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    
    try:
        data = request.json
        
        player_name = data.get('player_name')
        target_stat = data.get('target_stat')
        spread = data.get('spread')
        total = data.get('total')
        
        print(f"\n=== Received Request ===")
        print(f"Player: {player_name}")
        print(f"Target Stat: {target_stat}")
        print(f"Spread: {spread}")
        print(f"Total: {total}")
        print("=" * 30)
        
        # Validate inputs
        if not all([player_name, target_stat, spread is not None, total is not None]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Call your prediction function
        print("Calling create_complete_prediction...")
        result = create_complete_prediction(player_name, target_stat, spread, total)
        
        if result is None:
            error_msg = f'Prediction failed for {player_name}. Player may not be found or insufficient data available.'
            print(f"ERROR: {error_msg}")
            return jsonify({'error': error_msg}), 404
        
        gc.collect()  # Clean up memory
        
        print("Prediction successful! Converting to JSON...")
        
        # Convert all numpy/pandas types to native Python types
        result = convert_to_json_serializable(result)
        
        # Format response to match frontend expectations
        response = {
            'player': result['player'],
            'prediction': float(result['prediction']),
            'confidence': float(result['confidence']),
            'target_stat': result['target_stat'],
            'usage_rate': float(result['usage_rate']),
            'game_context': result['game_context'],
            'individual_predictions': result['individual_predictions']
        }
        
        print("Response ready, sending to frontend...")
        return jsonify(response)
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n!!! ERROR !!!")
        print(f"Error message: {error_msg}")
        print(f"Full traceback:")
        traceback.print_exc()
        print("=" * 50)
        
        return jsonify({
            'error': f'Prediction error: {error_msg}',
            'details': 'Check server console for full traceback'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': MODEL_LOADED})

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'NBA Prediction API is running!', 'model_loaded': MODEL_LOADED})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)