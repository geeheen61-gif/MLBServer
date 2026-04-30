from pybaseball import statcast_batter, playerid_lookup, statcast_pitcher
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from groq import Groq
import threading

class HRPredictor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = Groq(api_key=self.api_key)
        self.bankroll = 1000
        self.simulations = 4000
        self.league_avg_hr9 = 1.1
        self.max_prob_cap = 0.38
        self._id_cache = {}
        self._data_cache = {}
        self._cache_lock = threading.Lock()

    def _groq_summary(self, prompt):
        """Call Groq LLM for AI-generated summaries."""
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a professional MLB betting analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=250
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "AI summary unavailable."

    def _get_player_id(self, first_name, last_name):
        cache_key = f"{first_name}_{last_name}".lower()
        with self._cache_lock:
            if cache_key in self._id_cache:
                return self._id_cache[cache_key]
        
        # Use playerid_lookup - it can be slow
        df = playerid_lookup(last_name, first_name)
        if not df.empty:
            pid = df.iloc[0]['key_mlbam']
            with self._cache_lock:
                self._id_cache[cache_key] = pid
            return pid
        return None

    def predict(self, player_name, sportsbook_odds, pitcher_hr9, park_factor, pitcher_name="Unknown Pitcher", stadium_name="Unknown Stadium", is_home=True):
        """Full Calibrated Pipeline for Batter Prediction with Linked Analysis."""
        try:
            base_result = self._get_base_batter_data(player_name, sportsbook_odds)
            if "error" in base_result:
                return base_result
            
            base_prob = base_result["probability"]

            # Dampened Pitcher Adjustment (Baseline 1.1)
            pitcher_adj = 1 + ((float(pitcher_hr9) - self.league_avg_hr9) * 0.15)
            home_adj = 1.05 if is_home else 0.95

            final_prob = base_prob * pitcher_adj * float(park_factor) * home_adj
            final_prob = min(max(final_prob, 0.05), self.max_prob_cap)

            # Betting Math
            implied_probability = 100 / (sportsbook_odds + 100)
            edge = final_prob - implied_probability
            payout = sportsbook_odds / 100
            ev = (final_prob * payout) - (1 - final_prob)
            
            b = sportsbook_odds / 100
            kelly = max((final_prob * (b + 1) - 1) / b, 0)
            recommended_bet = self.bankroll * (kelly * 0.25)

            # Value statement for AI prompt
            if edge > 0:
                value_statement = "The model indicates potential value on the over."
            else:
                value_statement = "The model does not indicate value on the over at current pricing."

            prompt = f"""
Provide a high-stakes professional MLB betting analysis for this specific matchup:
BATTER: {player_name}
PITCHER: {pitcher_name} (HR/9: {pitcher_hr9})
STADIUM: {stadium_name} (Factor: {park_factor})

DATA METRICS:
- HR Probability: {final_prob:.3f}
- Market Odds: {sportsbook_odds} (Implied: {implied_probability:.3f})
- Edge: {edge:.3f}
- EV: {ev:.3f}

{value_statement}

REQUIRED ANALYSIS STRUCTURE:
1. SITUATIONAL BREAKDOWN: Explain EXACTLY why this batter is good (or bad) in these specific conditions. Mention the matchup against {pitcher_name} and how {stadium_name}'s unique traits affect the batter's swing path or power output.
2. PERFORMANCE DEEP DIVE: Analyze recent form, power trends, and "overall performance" indicators for both the batter and pitcher.
3. COMPLETE MATCH ANALYSIS: How does this specific prop fit into the context of the entire game? Mention atmospheric conditions if relevant.
4. OVERALL MATCH PREDICTION: A final, authoritative verdict on whether to bet the over, and a projection of the match's home run volatility.

Rules:
- Be extremely analytical. Use terms like "exit velocity," "launch angle," or "pitch mix" if relevant.
- Clearly state the SITUATION in which the batter excels.
- End with a clear "OVERALL MATCH PREDICTION" section.
"""
            client_summary = self._groq_summary(prompt)

            return {
                "player_name": player_name,
                "probability": float(final_prob),
                "implied_probability": float(implied_probability),
                "edge": float(edge),
                "expected_value": float(ev),
                "recommended_bet": float(recommended_bet),
                "freshness": base_result["freshness"],
                "summary": client_summary,
                "value_present": bool(ev > 0)
            }
        except Exception as e:
            return {"error": f"Internal Error: {str(e)}"}

    def _get_base_batter_data(self, player_name, sportsbook_odds):
        try:
            names = player_name.split()
            if len(names) < 2:
                return {"error": "Provide first and last name."}
            
            player_id = self._get_player_id(names[0], names[-1])
            if not player_id:
                return {"error": f"Batter '{player_name}' not found."}
            
            today = datetime.now()
            cache_key = f"data_batter_{player_id}"
            with self._cache_lock:
                if cache_key in self._data_cache:
                    ts, data = self._data_cache[cache_key]
                    if (today - ts).seconds < 3600:
                        return data

            # Optimization: Try 120 days first, then fallback to 365
            start_120 = today - timedelta(days=120)
            data_raw = statcast_batter(start_120.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"), player_id)
            
            if data_raw.empty:
                start_365 = today - timedelta(days=365)
                data_raw = statcast_batter(start_365.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"), player_id)

            if data_raw.empty:
                return {"error": f"No Statcast data found for {player_name}."}
            
            data_clean = data_raw.dropna(subset=['launch_speed', 'launch_angle'])
            bip_df = data_clean[data_clean['events'].notna()]
            
            if len(bip_df) < 20:
                return {"error": f"Not enough data for {player_name} ({len(bip_df)} BIP found)."}
            
            hr_per_bip = len(bip_df[bip_df['events']=='home_run']) / len(bip_df)
            avg_bip_game = len(bip_df) / data_raw['game_date'].nunique()

            sims = []
            for _ in range(self.simulations):
                simulated_bip = np.random.poisson(avg_bip_game)
                simulated_bip = max(simulated_bip, 1)
                hrs = np.random.binomial(simulated_bip, hr_per_bip)
                sims.append(1 if hrs >= 1 else 0)
            
            last_game = pd.to_datetime(data_raw['game_date']).max()
            days_since = (today - last_game).days
            freshness = "Fully Current" if days_since <= 3 else "Delayed" if days_since <= 10 else "Stale"

            res = {
                "player_name": player_name,
                "probability": np.mean(sims),
                "freshness": freshness
            }
            with self._cache_lock:
                self._data_cache[cache_key] = (today, res)
            return res
        except Exception as e:
            return {"error": f"Statcast Processing Error: {str(e)}"}

    def predict_pitcher(self, pitcher_name, player_name="Unknown Batter", stadium_name="Unknown Stadium"):
        """Calibrated Pitcher Model with Regression to Mean and Contextual Analysis."""
        try:
            print(f"DEBUG: Starting prediction for pitcher: {pitcher_name}")
            names = pitcher_name.split()
            p_id = self._get_player_id(names[0], names[-1])
            if not p_id:
                return {"error": f"Pitcher '{pitcher_name}' not found."}
            
            today = datetime.now()
            cache_key = f"data_pitcher_{p_id}"
            with self._cache_lock:
                if cache_key in self._data_cache:
                    ts, data = self._data_cache[cache_key]
                    if (today - ts).seconds < 3600:
                        return data

            # Optimized Fetch (120 days)
            start_120 = today - timedelta(days=120)
            data_raw = statcast_pitcher(start_120.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"), p_id)
            
            if data_raw.empty:
                start_365 = today - timedelta(days=365)
                data_raw = statcast_pitcher(start_365.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"), p_id)
            
            if data_raw.empty:
                return {"error": "No pitcher data found."}
            
            data_clean = data_raw.dropna(subset=["launch_speed", "launch_angle"])
            bip_df = data_clean[data_clean["events"].notna()]
            
            if len(bip_df) < 20:
                return {"error": f"Not enough recent data for {pitcher_name}."}
            
            hr_per_bip_allowed = len(bip_df[bip_df["events"]=="home_run"]) / len(bip_df)
            hard_hit_allowed = len(bip_df[bip_df["launch_speed"]>=95]) / len(bip_df)
            gb_rate = len(bip_df[bip_df["launch_angle"]<10]) / len(bip_df)
            barrels = bip_df[(bip_df["launch_speed"]>=98) & (bip_df["launch_angle"].between(26, 30))]
            barrel_rate_allowed = len(barrels) / len(bip_df)
            
            # Regression to mean for HR/9
            total_hr = len(data_raw[data_raw['events']=='home_run'])
            outs_list = ['field_out','strikeout','double_play','force_out','field_double_play']
            total_outs = len(data_raw[data_raw['events'].isin(outs_list)])
            innings = total_outs / 3.0 if total_outs > 0 else 1
            hr9_raw = (total_hr / innings) * 9
            hr9 = (0.6 * hr9_raw) + (0.4 * self.league_avg_hr9)
            
            v_score = min(max((hr_per_bip_allowed * 400) + (barrel_rate_allowed * 300) + (hr9 * 10) - (gb_rate * 20), 0), 100)
            projected_k = round(np.random.normal(5.8, 1.3), 1)

            prompt = f"""
Provide a professional pitching vulnerability analysis for {pitcher_name}.
MATCHUP CONTEXT:
- Facing Batter: {player_name}
- Playing at Stadium: {stadium_name}

METRICS:
- HR/BIP Allowed: {hr_per_bip_allowed:.3f}
- Barrel Rate Allowed: {barrel_rate_allowed*100:.1f}%
- Adj. HR/9: {hr9:.2f}
- Ground Ball Rate: {gb_rate*100:.1f}%

REQUIRED STRUCTURE:
1. PITCHER SITUATIONAL ANALYSIS: Under what specific conditions does {pitcher_name} struggle? Analyze pitch mix vulnerabilities against a batter like {player_name} and how they perform in {stadium_name}'s specific environment.
2. OVERALL PERFORMANCE: Evaluate current season trends, recent velocity changes, and consistency.
3. COMPLETE MATCH ANALYSIS: How does this pitcher's profile impact the overall match scoring and home run potential for the opposing team today?
4. OVERALL MATCH PREDICTION: Final verdict on this pitcher's likely performance floor/ceiling for today.

End with "OVERALL MATCH PREDICTION".
"""
            pitcher_summary = self._groq_summary(prompt)

            res = {
                "pitcher_name": pitcher_name,
                "HR_per_BIP_allowed": round(hr_per_bip_allowed, 3),
                "HardHit_allowed_pct": round(hard_hit_allowed * 100, 1),
                "GroundBall_rate": round(gb_rate, 3),
                "Barrel_allowed": round(barrel_rate_allowed, 3),
                "HR_per_9": round(hr9, 2),
                "projected_k": projected_k,
                "vulnerability_score": round(v_score, 1),
                "confidence": "High" if len(bip_df) > 150 else "Medium",
                "summary": pitcher_summary
            }
            
            with self._cache_lock:
                self._data_cache[cache_key] = (today, res)
            return res
        except Exception as e:
            return {"error": f"Pitcher Error: {str(e)}"}

    def ballpark_factor(self, stadium_name, player_name="Unknown Batter", pitcher_name="Unknown Pitcher"):
        parks = {
            "Coors Field": {"hr_factor": 1.25, "runs_factor": 1.34, "description": "High altitude, extreme hitter friendly."},
            "Yankee Stadium": {"hr_factor": 1.18, "runs_factor": 1.05, "description": "Short porch in right field."},
            "Fenway Park": {"hr_factor": 1.05, "runs_factor": 1.12, "description": "Green Monster impacts doubles/HRs."},
            "Dodger Stadium": {"hr_factor": 1.10, "runs_factor": 0.95, "description": "Pitcher friendly at night, neutral day."},
            "Oracle Park": {"hr_factor": 0.82, "runs_factor": 0.92, "description": "Extreme pitcher friendly, heavy air."},
        }
        info = parks.get(stadium_name, {"hr_factor": 1.0, "runs_factor": 1.0, "description": "Neutral park factors."})
        info["hr_factor"] = min(max(info["hr_factor"], 0.85), 1.25)

        prompt = f"""
Provide a deep analytical summary of {stadium_name} for a betting audience.
MATCHUP CONTEXT:
- Key Batter: {player_name}
- Key Pitcher: {pitcher_name}

HR Factor: {info['hr_factor']}
Runs Factor: {info['runs_factor']}

REQUIRED STRUCTURE:
1. BALLPARK SITUATIONAL ANALYSIS: How do specific types of hitters like {player_name} benefit from this stadium? Mention dimensions, wind patterns, and how {pitcher_name}'s pitch style might interact with the air density here.
2. OVERALL PERFORMANCE: How has this stadium performed recently in terms of total scoring and park-adjusted trends?
3. COMPLETE MATCH ANALYSIS: How does this ballpark's profile change the overall strategy of the match between these specific teams/players?
4. OVERALL MATCH PREDICTION: Final verdict on the "Over/Under" environment for today's match in this stadium.

End with "OVERALL MATCH PREDICTION".
"""
        stadium_summary = self._groq_summary(prompt)
        info["summary"] = stadium_summary
        return info
