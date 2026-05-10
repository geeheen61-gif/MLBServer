from flask import Flask, request, jsonify, render_template
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

# Initialize predictor with Groq API keys from environment
# Provide keys separated by commas, e.g. GROQ_API_KEY=key1,key2,key3
API_KEYS_RAW = os.getenv("GROQ_API_KEY", "")
API_KEYS = [k.strip() for k in API_KEYS_RAW.split(",") if k.strip()]

if not API_KEYS:
    print("⚠️ WARNING: No GROQ_API_KEY(s) found in environment.")
    # Fallback for local testing if needed, but in production, this should fail early
    API_KEYS = ["dummy_key"] 

predictor = HRPredictor(API_KEYS)

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
            manual_team=data.get('manual_team'),
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
            park_info['metrics']['hr_factor'], 
            is_home=is_home
        )

        return jsonify({
            "batter_analysis": batter_result,
            "pitcher_analysis": pitcher_result,
            "stadium_analysis": park_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/update_player_team', methods=['POST'])
def update_player_team():
    data = request.get_json()
    player_id = data.get('player_id')
    team_name = data.get('team_name')
    if not player_id or not team_name:
        return jsonify({"error": "Missing player_id or team_name"}), 400
    
    try:
        result = predictor.update_player_team(player_id, team_name)
        return jsonify({"success": True, "updated_info": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
