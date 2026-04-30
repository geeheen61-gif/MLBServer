import os
import json
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from groq import Groq
from pybaseball import statcast_batter, playerid_lookup, statcast_pitcher

class DiskCache:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.lock = threading.Lock()

    def _get_path(self, key):
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, key, expiry_hours=24):
        path = self._get_path(key)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    cached = json.load(f)
                
                # Check expiry
                if (time.time() - cached['timestamp']) < (expiry_hours * 3600):
                    return cached['data']
            except:
                pass
        return None

    def set(self, key, data):
        with self.lock:
            try:
                path = self._get_path(key)
                with open(path, 'w') as f:
                    json.dump({'timestamp': time.time(), 'data': data}, f)
            except:
                pass

class HRPredictor:
    def __init__(self, api_keys):
        self.api_keys = api_keys if isinstance(api_keys, list) else [api_keys]
        self.current_key_index = 0
        self.client = Groq(api_key=self.api_keys[self.current_key_index])
        self.disk_cache = DiskCache()
        self.simulations = 2000 # Reduced for speed
        self.league_avg_hr9 = 1.1

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
                    max_tokens=200
                )
                return response.choices[0].message.content
            except:
                self._rotate_key()
        return "AI analysis temporarily unavailable."

    def _get_player_id(self, name):
        cache_key = f"id_{name.replace(' ', '_').lower()}"
        cached_id = self.disk_cache.get(cache_key, expiry_hours=720) # Cache IDs for 30 days
        if cached_id: return cached_id

        print(f"Finding ID for {name}...")
        parts = name.split()
        if len(parts) < 2: return None
        
        try:
            df = playerid_lookup(parts[-1], parts[0])
            if not df.empty:
                pid = int(df.iloc[0]['key_mlbam'])
                self.disk_cache.set(cache_key, pid)
                return pid
        except: pass
        return None

    def predict(self, player_name, odds, hr9, park, **kwargs):
        try:
            pid = self._get_player_id(player_name)
            if not pid: return {"error": "Player not found"}

            # Try to get probability from cache
            cache_key = f"prob_{pid}"
            base_prob = self.disk_cache.get(cache_key, expiry_hours=12)
            
            if base_prob is None:
                print(f"Calculating base probability for {player_name}...")
                today = datetime.now()
                start = (today - timedelta(days=120)).strftime("%Y-%m-%d")
                df = statcast_batter(start, today.strftime("%Y-%m-%d"), pid)
                
                if df.empty:
                    base_prob = 0.12 # Fallback to league average profile
                else:
                    df = df.dropna(subset=['events'])
                    hr_rate = len(df[df['events'] == 'home_run']) / max(len(df), 1)
                    avg_pa = len(df) / max(df['game_date'].nunique(), 1)
                    
                    sims = np.random.binomial(np.random.poisson(avg_pa, self.simulations), hr_rate)
                    base_prob = np.mean(sims >= 1)
                
                self.disk_cache.set(cache_key, float(base_prob))

            # Adjustments
            adj = (1 + (float(hr9) - 1.1) * 0.2) * float(park)
            if kwargs.get('is_home'): adj *= 1.05
            
            final_prob = min(max(base_prob * adj, 0.05), 0.40)
            implied = 100 / (float(odds) + 100)
            edge = final_prob - implied

            prompt = f"Analyze MLB Batter {player_name} vs HR/9 {hr9} at Park Factor {park}. Prob: {final_prob:.2f}, Odds: {odds}. Short breakdown and ### OVERALL MATCH PREDICTION ###"
            
            return {
                "metrics": {
                    "probability": f"{final_prob:.1%}",
                    "implied_odds": f"{implied:.1%}",
                    "edge": f"{edge:+.1%}",
                    "ev": f"{(final_prob * (float(odds)/100) - (1-final_prob)):.2f}"
                },
                "summary": self._groq_summary(prompt)
            }
        except Exception as e:
            return {"error": str(e)}

    def predict_pitcher(self, name, **kwargs):
        try:
            pid = self._get_player_id(name)
            if not pid: return {"error": "Pitcher not found"}

            cache_key = f"pitcher_{pid}"
            data = self.disk_cache.get(cache_key, expiry_hours=12)
            
            if not data:
                today = datetime.now()
                start = (today - timedelta(days=90)).strftime("%Y-%m-%d")
                df = statcast_pitcher(start, today.strftime("%Y-%m-%d"), pid)
                
                if df.empty: data = {"hr9": 1.2, "score": 50}
                else:
                    hrs = len(df[df['events'] == 'home_run'])
                    outs = len(df[df['events'].isin(['field_out','strikeout','double_play'])])
                    hr9 = (hrs / max(outs/3.0, 1)) * 9
                    data = {"hr9": round(hr9, 2), "score": round(min(hr9 * 40, 100), 1)}
                
                self.disk_cache.set(cache_key, data)

            prompt = f"Analyze Pitcher {name} with HR/9 of {data['hr9']}. Short breakdown and ### OVERALL MATCH PREDICTION ###"
            return {
                "metrics": {"hr9": data['hr9'], "vulnerability_score": data['score']},
                "summary": self._groq_summary(prompt)
            }
        except Exception as e:
            return {"error": str(e)}

    def ballpark_factor(self, name, **kwargs):
        parks = {"Coors Field": 1.25, "Yankee Stadium": 1.18, "Fenway Park": 1.05, "Dodger Stadium": 1.10}
        factor = parks.get(name, 1.0)
        prompt = f"Analyze {name} Ballpark. Factor: {factor}. Short breakdown and ### OVERALL MATCH PREDICTION ###"
        return {
            "metrics": {"park_factor": factor},
            "summary": self._groq_summary(prompt)
        }
