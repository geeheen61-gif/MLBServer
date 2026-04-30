from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import HRPredictor
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Fix CORS for Flutter Web (Chrome sends preflight OPTIONS requests)
CORS(app,
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"],
     supports_credentials=False)

# Initialize predictor with Groq API key from environment
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    print("⚠️ WARNING: GROQ_API_KEY not found in environment.")
predictor = HRPredictor(API_KEY)

@app.route('/predict_batter', methods=['POST'])
def predict_batter():
    data = request.get_json()
    player_name = data.get('player_name')
    sportsbook_odds = data.get('sportsbook_odds')
    pitcher_hr9 = data.get('pitcher_hr9', 1.1)
    park_factor = data.get('park_factor', 1.0)
    is_home = data.get('is_home', True)

    if not all([player_name, sportsbook_odds]):
        return jsonify({"error": "Missing player_name or sportsbook_odds"}), 400

    try:
        result = predictor.predict(
            player_name, 
            float(sportsbook_odds), 
            float(pitcher_hr9), 
            float(park_factor), 
            pitcher_name=data.get('pitcher_name', 'Unknown Pitcher'),
            stadium_name=data.get('stadium_name', 'Unknown Stadium'),
            is_home=is_home
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_pitcher', methods=['POST'])
def predict_pitcher_route():
    data = request.get_json()
    pitcher_name = data.get('pitcher_name')
    if not pitcher_name:
        return jsonify({"error": "Missing pitcher_name"}), 400
    
    try:
        result = predictor.predict_pitcher(
            pitcher_name,
            player_name=data.get('player_name', 'Unknown Batter'),
            stadium_name=data.get('stadium_name', 'Unknown Stadium')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/park_history', methods=['POST'])
def park_history():
    data = request.get_json()
    stadium_name = data.get('stadium_name')
    if not stadium_name:
        return jsonify({"error": "Missing stadium_name"}), 400
    
    try:
        result = predictor.ballpark_factor(
            stadium_name,
            player_name=data.get('player_name', 'Unknown Batter'),
            pitcher_name=data.get('pitcher_name', 'Unknown Pitcher')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze_game', methods=['POST'])
def analyze_game():
    # Keeping this for legacy/consolidated if needed, but the user wants separate sections
    data = request.get_json()
    player_name = data.get('player_name')
    pitcher_name = data.get('pitcher_name')
    stadium_name = data.get('stadium_name')
    is_home = data.get('is_home', True)
    sportsbook_odds = data.get('sportsbook_odds')
    pitcher_hr9 = data.get('pitcher_hr9', 1.1)

    if not all([player_name, pitcher_name, sportsbook_odds]):
        return jsonify({"error": "Missing player_name, pitcher_name, or sportsbook_odds"}), 400

    try:
        park_info = predictor.ballpark_factor(stadium_name)
        pitcher_result = predictor.predict_pitcher(pitcher_name)
        batter_result = predictor.predict(
            player_name, 
            float(sportsbook_odds), 
            float(pitcher_hr9), 
            park_info['hr_factor'], 
            is_home=is_home
        )

        return jsonify({
            "batter_analysis": batter_result,
            "pitcher_analysis": pitcher_result,
            "stadium_analysis": park_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    # For production level, we usually use waitress or gunicorn
    # But for local dev/testing, this is fine. 
    # I'll include a waitress start script later.
    app.run(host='0.0.0.0', port=5000, debug=True)
