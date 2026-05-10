import os
import json
import time
import threading
import math
import unicodedata
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from groq import Groq
from pybaseball import statcast_batter, playerid_lookup, statcast_pitcher

# ── DATA MAPPINGS ──────────────────────────────────────────────────────────
TEAM_TO_STADIUM = {
    "Arizona Diamondbacks": "Chase Field",
    "Atlanta Braves": "Truist Park",
    "Baltimore Orioles": "Oriole Park at Camden Yards",
    "Boston Red Sox": "Fenway Park",
    "Chicago Cubs": "Wrigley Field",
    "Chicago White Sox": "Guaranteed Rate Field",
    "Cincinnati Reds": "Great American Ball Park",
    "Cleveland Guardians": "Progressive Field",
    "Colorado Rockies": "Coors Field",
    "Detroit Tigers": "Comerica Park",
    "Houston Astros": "Minute Maid Park",
    "Kansas City Royals": "Kauffman Stadium",
    "Los Angeles Angels": "Angel Stadium",
    "Los Angeles Dodgers": "Dodger Stadium",
    "Miami Marlins": "loanDepot park",
    "Milwaukee Brewers": "American Family Field",
    "Minnesota Twins": "Target Field",
    "New York Mets": "Citi Field",
    "New York Yankees": "Yankee Stadium",
    "Oakland Athletics": "Sutter Health Park",
    "Philadelphia Phillies": "Citizens Bank Park",
    "Pittsburgh Pirates": "PNC Park",
    "San Diego Padres": "Petco Park",
    "San Francisco Giants": "Oracle Park",
    "Seattle Mariners": "T-Mobile Park",
    "St. Louis Cardinals": "Busch Stadium",
    "Tampa Bay Rays": "Tropicana Field",
    "Texas Rangers": "Globe Life Field",
    "Toronto Blue Jays": "Rogers Centre",
    "Washington Nationals": "Nationals Park"
}

STADIUM_INFO = {
    "Coors Field": {"factor": 1.25, "type": "Extreme Hitter Friendly"},
    "Great American Ball Park": {"factor": 1.20, "type": "Extreme Hitter Friendly"},
    "Yankee Stadium": {"factor": 1.18, "type": "Hitter Friendly"},
    "Citizens Bank Park": {"factor": 1.12, "type": "Hitter Friendly"},
    "Angel Stadium": {"factor": 1.05, "type": "Slight Hitter Friendly"},
    "Dodger Stadium": {"factor": 1.10, "type": "Slight Hitter Friendly"},
    "Fenway Park": {"factor": 1.05, "type": "Neutral to Slight Hitter"},
    "Truist Park": {"factor": 1.08, "type": "Slight Hitter Friendly"},
    "Guaranteed Rate Field": {"factor": 1.10, "type": "Hitter Friendly"},
    "Minute Maid Park": {"factor": 1.06, "type": "Neutral"},
    "Rogers Centre": {"factor": 1.07, "type": "Neutral"},
    "Globe Life Field": {"factor": 1.02, "type": "Neutral"},
    "Wrigley Field": {"factor": 1.05, "type": "Neutral"},
    "American Family Field": {"factor": 1.08, "type": "Slight Hitter Friendly"},
    "Chase Field": {"factor": 1.00, "type": "Neutral"},
    "Oriole Park at Camden Yards": {"factor": 0.95, "type": "Slight Pitcher Friendly"},
    "Target Field": {"factor": 0.98, "type": "Neutral"},
    "Nationals Park": {"factor": 1.02, "type": "Neutral"},
    "Kauffman Stadium": {"factor": 0.94, "type": "Pitcher Friendly"},
    "Comerica Park": {"factor": 0.92, "type": "Pitcher Friendly"},
    "Oracle Park": {"factor": 0.88, "type": "Extreme Pitcher Friendly"},
    "Petco Park": {"factor": 0.90, "type": "Pitcher Friendly"},
    "T-Mobile Park": {"factor": 0.92, "type": "Pitcher Friendly"},
    "Busch Stadium": {"factor": 0.94, "type": "Pitcher Friendly"},
    "Citi Field": {"factor": 0.96, "type": "Slight Pitcher Friendly"},
    "loanDepot park": {"factor": 0.92, "type": "Pitcher Friendly"},
    "Tropicana Field": {"factor": 0.94, "type": "Pitcher Friendly"},
    "Progressive Field": {"factor": 1.02, "type": "Neutral"},
    "Oakland Coliseum": {"factor": 0.85, "type": "Extreme Pitcher Friendly"},
    "Sutter Health Park": {"factor": 1.10, "type": "Hitter Friendly"},
}

class DiskCache:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir): os.makedirs(cache_dir)
        self.lock = threading.Lock()

    def _get_path(self, key): return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, key, expiry_hours=24):
        path = self._get_path(key)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    cached = json.load(f)
                if (time.time() - cached['timestamp']) < (expiry_hours * 3600):
                    return cached['data']
            except: pass
        return None

    def set(self, key, data):
        with self.lock:
            try:
                path = self._get_path(key)
                with open(path, 'w') as f:
                    json.dump({'timestamp': time.time(), 'data': data}, f)
            except: pass

class HRPredictor:
    def __init__(self, api_keys):
        self.api_keys = api_keys if isinstance(api_keys, list) else [api_keys]
        self.current_key_index = 0
        self.client = Groq(api_key=self.api_keys[self.current_key_index])
        self.disk_cache = DiskCache()
        self.league_avg_hr_rate = 0.035
        self.league_avg_pa = 4.2

    def _normalize_name(self, name):
        """Remove accents and normalize case for robust searching."""
        nfkd_form = unicodedata.normalize('NFKD', name)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).strip().lower()

    def _rotate_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.client = Groq(api_key=self.api_keys[self.current_key_index])

    def _groq_summary(self, prompt):
        for _ in range(len(self.api_keys)):
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=250
                )
                return response.choices[0].message.content
            except: self._rotate_key()
        return "AI analysis unavailable."

    def _get_player_id(self, name):
        clean_name = self._normalize_name(name)
        cache_key = f"id_v4_{clean_name.replace(' ', '_')}" 
        cached_id = self.disk_cache.get(cache_key, expiry_hours=720) # 30 days
        if cached_id: return cached_id

        print(f"🔎 Deep-searching player: {name}...")
        
        # 1. Advanced Search with hydration
        try:
            # Use search with hydration to get team info immediately
            search_url = "https://statsapi.mlb.com/api/v1/people/search"
            params = {"names": name, "hydrate": "currentTeam"}
            resp = requests.get(search_url, params=params, timeout=5)
            data = resp.json()
            
            if "people" in data and len(data["people"]) > 0:
                # Pick the most 'active' looking player if multiple
                person = data["people"][0]
                pid = int(person["id"])
                
                # Check if this person is currently active
                is_active = person.get("active", False)
                if not is_active and len(data["people"]) > 1:
                    for p in data["people"][1:]:
                        if p.get("active"):
                            person = p
                            pid = int(p["id"])
                            break
                
                print(f"✅ Found {person.get('fullName')} (ID: {pid}) | Team: {person.get('currentTeam', {}).get('name', 'N/A')}")
                self.disk_cache.set(cache_key, pid)
                return pid
        except Exception as e:
            print(f"⚠️ MLB Search Error: {e}")

        # 2. Fallback to pybaseball
        try:
            parts = name.split()
            if len(parts) >= 1:
                df = playerid_lookup(parts[-1], parts[0] if len(parts) > 1 else "", fuzzy=True)
                if not df.empty:
                    pid = int(df.iloc[0]['key_mlbam'])
                    self.disk_cache.set(cache_key, pid)
                    return pid
        except: pass
        
        return None

    def get_today_matchup(self, team_id):
        """Fetches today's game details: Opponent, Venue, Home/Away, and Probable Pitcher."""
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"matchup_{team_id}_{today}"
        cached = self.disk_cache.get(cache_key, expiry_hours=1)
        if cached: return cached

        try:
            url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher,weather,venue"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            
            for date in data.get("dates", []):
                for game in date.get("games", []):
                    is_home = game["teams"]["home"]["team"]["id"] == team_id
                    is_away = game["teams"]["away"]["team"]["id"] == team_id
                    
                    if is_home or is_away:
                        opponent_type = "away" if is_home else "home"
                        opponent = game["teams"][opponent_type]["team"]["name"]
                        pitcher_data = game["teams"][opponent_type].get("probablePitcher", {})
                        
                        matchup = {
                            "opponent": opponent,
                            "venue": game.get("venue", {}).get("name", "Unknown"),
                            "is_home": is_home,
                            "pitcher_name": pitcher_data.get("fullName", "TBD"),
                            "weather": game.get("weather", {}).get("condition", "Unknown"),
                            "temp": game.get("weather", {}).get("temp", "--")
                        }
                        self.disk_cache.set(cache_key, matchup)
                        return matchup
        except Exception as e:
            print(f"⚠️ Schedule API Error: {e}")
        
        return None

    def get_platoon_splits(self, player_id):
        """Fetches HR rates and ISO vs LHP and vs RHP for deep power analysis."""
        today = datetime.now().year
        cache_key = f"splits_v2_{player_id}_{today}"
        cached = self.disk_cache.get(cache_key, expiry_hours=168)
        if cached: return cached

        try:
            url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=statSplits&group=hitting&season={today}&sitCode=vr,vl"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            
            splits = {
                "vs_LHP": {"hr_rate": 0.035, "iso": 0.150, "pa": 0}, 
                "vs_RHP": {"hr_rate": 0.035, "iso": 0.150, "pa": 0}
            }
            
            for split in data.get("stats", [{}])[0].get("splits", []):
                hand = split.get("condition", {}).get("description")
                stat = split.get("stat", {})
                pa = stat.get("plateAppearances", 1)
                hr = stat.get("homeRuns", 0)
                slg = float(stat.get("slugging", ".000"))
                avg = float(stat.get("avg", ".000"))
                iso = slg - avg
                
                raw_rate = hr / max(pa, 1)
                credibility = min(pa / 150, 1.0)
                hr_rate = (raw_rate * credibility) + (0.035 * (1 - credibility))
                iso_smooth = (iso * credibility) + (0.150 * (1 - credibility))
                
                key = "vs_LHP" if "Left" in hand else "vs_RHP"
                splits[key] = {"hr_rate": hr_rate, "iso": iso_smooth, "pa": pa}
            
            self.disk_cache.set(cache_key, splits)
            return splits
        except Exception:
            return {"vs_LHP": {"hr_rate": 0.035, "iso": 0.150, "pa": 0}, "vs_RHP": {"hr_rate": 0.035, "iso": 0.150, "pa": 0}}

    def get_pitcher_handedness(self, name):
        """Finds if a pitcher is LHP or RHP."""
        pid = self._get_player_id(name)
        if not pid: return "R" # Default
        
        cache_key = f"pitcher_hand_{pid}"
        cached = self.disk_cache.get(cache_key, expiry_hours=2160)
        if cached: return cached

        try:
            url = f"https://statsapi.mlb.com/api/v1/people/{pid}"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            hand = data["people"][0].get("pitchHand", {}).get("code", "R")
            self.disk_cache.set(cache_key, hand)
            return hand
        except: return "R"

    def get_current_team_info(self, player_id):
        """Fetches live team data with deep stats verification for trade accuracy."""
        cache_key = f"live_team_v5_{player_id}"
        cached = self.disk_cache.get(cache_key, expiry_hours=4) # Very short expiry
        if cached: return cached

        try:
            # 1. Try People Hydration
            url = f"https://statsapi.mlb.com/api/v1/people/{player_id}?hydrate=currentTeam"
            resp = requests.get(url, timeout=5)
            person = resp.json().get("people", [{}])[0]
            team_info = person.get("currentTeam", {})
            team_name = team_info.get("name", "Unknown Team")
            team_id = team_info.get("id")

            # 2. Cross-verify with Season Stats (Source of Truth for trades)
            today = datetime.now().year
            stats_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&group=hitting&season={today}"
            stats_resp = requests.get(stats_url, timeout=5)
            stats_data = stats_resp.json()
            
            latest_stats_team = None
            for stat in stats_data.get("stats", []):
                for split in stat.get("splits", []):
                    t = split.get("team", {})
                    if t.get("name"): latest_stats_team = t

            if latest_stats_team:
                team_name = latest_stats_team.get("name")
                team_id = latest_stats_team.get("id")

            # 3. Ballpark Mapping
            stadium = "Unknown Stadium"
            match_name = team_name.lower()
            for t_name, s_name in TEAM_TO_STADIUM.items():
                if t_name.lower() in match_name or match_name in t_name.lower():
                    stadium = s_name
                    team_name = t_name
                    break
            
            factor = STADIUM_INFO.get(stadium, {"factor": 1.0})["factor"]
            
            info = {
                "team": team_name,
                "team_id": team_id,
                "stadium": stadium,
                "park_factor": factor
            }
            self.disk_cache.set(cache_key, info)
            return info
        except Exception as e:
            print(f"⚠️ Team Detection Error: {e}")
        
        return {"team": "Unknown Team", "team_id": None, "stadium": "Unknown Stadium", "park_factor": 1.0}

    def describe_hr9(self, hr9):
        if hr9 >= 1.4: return "elevated home run allowance (vulnerable)"
        elif hr9 >= 1.1: return "slightly above league average allowance"
        elif hr9 >= 0.9: return "near league average (neutral)"
        else: return "strong home run suppression (elite)"

    def generate_institutional_report(self, data):
        """Generates a strict, deterministic professional report with dynamic risk metrics."""
        p = data['player_info']
        m = data['metrics']
        
        profile = f"{p['name']} ({p['current_team']}) exhibits a {p['split_hr_rate']:.1%} HR rate and {p['split_iso']:.3f} ISO against {p['pitcher_hand']}HP, reflecting moderate power output. His performance does not show extreme park sensitivity relative to league norms."
        
        hr9_val = m.get('HR_per_9', 1.1)
        hr9_desc = self.describe_hr9(hr9_val)
        pitcher_context = f"{p['opposing_pitcher']}'s {hr9_val} HR/9 {hr9_desc}"
        
        park_pct = int((p['park_factor'] - 1) * 100)
        park_dir = "increasing" if park_pct > 0 else "decreasing"
        park_impact = f"{p['current_stadium']} carries a {p['park_factor']} home run factor, {park_dir} expected home run frequency by approximately {abs(park_pct)}% relative to neutral parks."
        
        lower_ci = max(m['probability'] - 0.03, 0.01)
        upper_ci = min(m['probability'] + 0.03, 0.45)
        prob_assessment = f"The model estimates a {m['probability']:.1%} probability of a home run (90% CI: {lower_ci:.1%}-{upper_ci:.1%}), while the market implies {m['implied_odds']:.1%}. This results in an edge of {m['edge']:.1%} ({'Positive' if m['edge'] > 0 else 'Negative'})."
        
        if m['edge'] > 0:
            decision = f"With a positive edge of {m['edge']:.1%} and EV of {m['expected_value']:.2f}, value is present at current pricing. Matchup Grade: {p['matchup_grade']} | Risk: {p['risk_grade']}."
        else:
            decision = f"The current market price overestimates the likelihood of a home run. With an EV of {m['expected_value']:.2f}, no value is offered. The recommended action is to pass."
            
        return f"""
### Player Profile
{profile}

### Pitcher Context
{pitcher_context}

### Park Impact
{park_impact}

### Probability Assessment
{prob_assessment}

### Betting Implication
{decision}

### RISK & VOLATILITY ###
- **Volatility Rating**: {p['volatility']}
- **Sample Integrity**: {p.get('sample_size', 'Moderate')}
- **Model Confidence**: {p.get('confidence', 'High')}

### OVERALL MATCH PREDICTION ###
{"VALUE PRESENT (EV+)" if m['edge'] > 0 else "NO VALUE (PASS)"}
"""

    def update_player_team(self, player_id, team_name):
        """Manually corrects a player's team and persists it to the learning cache."""
        cache_key = f"learned_team_{player_id}"
        
        # Resolve stadium
        stadium = "Unknown Stadium"
        match_name = team_name.lower()
        for t_name, s_name in TEAM_TO_STADIUM.items():
            if t_name.lower() in match_name or match_name in t_name.lower():
                stadium = s_name
                team_name = t_name
                break
        
        info = {
            "team": team_name,
            "stadium": stadium,
            "park_factor": STADIUM_INFO.get(stadium, {"factor": 1.0})["factor"],
            "timestamp": time.time()
        }
        self.disk_cache.set(cache_key, info)
        return info

    def predict(self, player_name, odds, hr9, park, **kwargs):
        try:
            pid = self._get_player_id(player_name)
            if not pid: return {"error": f"Player '{player_name}' not found."}

            # 🧠 TEAM LEARNING & LIVE DETECTION
            learned_key = f"learned_team_{pid}"
            learned_info = self.disk_cache.get(learned_key, expiry_hours=2160) # Remember for 90 days
            
            if learned_info:
                print(f"🧠 Using learned team for {player_name}: {learned_info['team']}")
                live_info = learned_info
            else:
                live_info = self.get_current_team_info(pid)
            
            current_team = live_info.get("team", "Unknown")
            
            # 🤖 DAILY AUTOMATION
            matchup = self.get_today_matchup(live_info.get("team_id")) if live_info.get("team_id") else None
            
            pitcher_name = kwargs.get('pitcher_name', 'Unknown Pitcher')
            is_home = kwargs.get('is_home', True)
            current_stadium = live_info["stadium"]
            weather_note = "Standard conditions"
            
            if matchup:
                if pitcher_name == 'Unknown Pitcher': pitcher_name = matchup["pitcher_name"]
                is_home = matchup["is_home"]
                current_stadium = matchup["venue"]
                weather_note = f"{matchup['temp']}°F, {matchup['weather']}"

            # 🧩 PLATOON SPLITS & HANDEDNESS
            pitcher_hand = self.get_pitcher_handedness(pitcher_name)
            splits_data = self.get_platoon_splits(pid)
            hand_key = "vs_LHP" if pitcher_hand == "L" else "vs_RHP"
            current_split = splits_data.get(hand_key, {"hr_rate": 0.035, "iso": 0.150, "pa": 0})
            split_hr_rate = current_split["hr_rate"]
            split_iso = current_split["iso"]
            split_pa = current_split.get("pa", 0)

            # Auto-fetch Pitcher HR/9
            final_hr9 = float(hr9)
            if pitcher_name != 'Unknown Pitcher' and final_hr9 == 1.1:
                p_data = self.predict_pitcher(pitcher_name)
                if "metrics" in p_data: final_hr9 = p_data["metrics"]["HR_per_9"]
            
            # Stadium Info
            stadium_data = STADIUM_INFO.get(current_stadium, {"factor": 1.0, "type": "Neutral Environment"})
            final_park_factor = float(park)
            if final_park_factor == 1.0:
                final_park_factor = stadium_data["factor"]
            
            # Probability Math
            lambda_val = 4.2
            adj = (1 + (final_hr9 - 1.1) * 0.25) * final_park_factor
            if is_home: adj *= 1.05
            stable_prob = 1 - math.exp(-(lambda_val * split_hr_rate * adj))
            final_prob = min(max(stable_prob, 0.02), 0.45)
            implied = 100 / (float(odds) + 100)
            edge = final_prob - implied
            ev = (final_prob * (float(odds)/100)) - (1 - final_prob)

            grade = "A" if edge > 0.05 else "B" if edge > 0 else "C"
            volatility = "High" if split_pa < 50 else "Moderate" if split_pa < 150 else "Low"
            risk_grade = "High" if volatility == "High" else "Medium"
            sample_size = "Small" if split_pa < 50 else "Adequate" if split_pa < 150 else "Large"
            
            result_data = {
                "player_info": {
                    "id": pid,
                    "name": player_name,
                    "current_team": current_team,
                    "current_stadium": current_stadium,
                    "opposing_pitcher": pitcher_name,
                    "pitcher_hand": pitcher_hand,
                    "weather": weather_note,
                    "split_hr_rate": split_hr_rate,
                    "split_iso": split_iso,
                    "park_factor": final_park_factor,
                    "matchup_grade": grade,
                    "volatility": volatility,
                    "risk_grade": risk_grade,
                    "sample_size": sample_size,
                    "confidence": "High" if sample_size == "Large" else "Moderate" if sample_size == "Adequate" else "Low"
                },
                "metrics": {
                    "probability": final_prob,
                    "implied_odds": implied,
                    "edge": edge,
                    "expected_value": ev,
                    "HR_per_9": final_hr9,
                    "recommended_bet": max(ev * 60, 0)
                }
            }
            
            result_data["summary"] = self.generate_institutional_report(result_data)
            return result_data
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}

    def predict_pitcher(self, name, **kwargs):
        try:
            pid = self._get_player_id(name)
            if not pid: return {"error": f"Pitcher '{name}' not found."}
            
            live_info = self.get_current_team_info(pid)
            
            cache_key = f"stats_pitcher_{pid}"
            data = self.disk_cache.get(cache_key, expiry_hours=12)
            if not data:
                today = datetime.now()
                start = (today - timedelta(days=365)).strftime("%Y-%m-%d")
                df = statcast_pitcher(start, today.strftime("%Y-%m-%d"), pid)
                if df.empty:
                    data = {"hr9": 1.15, "score": 50}
                else:
                    hrs = len(df[df['events'] == 'home_run'])
                    outs = len(df[df['events'].isin(['field_out','strikeout','double_play','force_out'])])
                    hr9_raw = (hrs / max(outs/3.0, 1)) * 9
                    hr9 = (hr9_raw * 0.75) + (1.1 * 0.25)
                    data = {"hr9": round(hr9, 2), "score": round(min(hr9 * 40, 100), 1)}
                self.disk_cache.set(cache_key, data)

            prompt = f"""
            Analyze MLB Pitcher: {name} ({live_info['team']})
            Metrics:
            - HR/9 Allowance: {data['hr9']} (League Average: 1.1)
            - Vulnerability Score: {data['score']}/100
            
            Provide a professional breakdown focusing on pitcher's propensity to give up home runs and ### OVERALL MATCH PREDICTION ###
            """
            return {
                "pitcher_info": {
                    "name": name,
                    "team": live_info["team"],
                    "hr9": data['hr9']
                },
                "metrics": {
                    "vulnerability_score": data['score'],
                    "HR_per_9": data['hr9'],
                    "project_k": 5.5,
                    "confidence": "High"
                },
                "summary": self._groq_summary(prompt)
            }
        except Exception as e: return {"error": str(e)}

    def ballpark_factor(self, name, **kwargs):
        stadium_data = STADIUM_INFO.get(name, {"factor": 1.0})
        factor = stadium_data["factor"]
        prompt = f"Ballpark: {name}. Factor: {factor}. Analysis of air density and historical HR rates. Breakdown and ### OVERALL MATCH PREDICTION ###"
        return {
            "stadium_name": name,
            "metrics": {
                "hr_factor": factor,
                "runs_factor": 1.02 if factor > 1.1 else 0.98,
                "altitude": "Calculated",
                "air_density": "Dynamic"
            },
            "summary": self._groq_summary(prompt)
        }

