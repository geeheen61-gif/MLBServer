import os
import json
import time
import threading
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from groq import Groq
from pybaseball import statcast_batter, playerid_lookup, statcast_pitcher

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
        self.league_avg_hr_rate = 0.035 # ~3.5% of PAs result in HR
        self.league_avg_pa = 4.2 # Average Plate Appearances per Game

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
        cache_key = f"id_{name.replace(' ', '_').lower()}"
        cached_id = self.disk_cache.get(cache_key, expiry_hours=720)
        if cached_id: return cached_id
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
        """Stable, Regression-to-Mean calibrated prediction engine."""
        try:
            pid = self._get_player_id(player_name)
            if not pid: return {"error": "Player not found"}

            cache_key = f"prob_v2_{pid}" # V2 for stable logic
            player_stats = self.disk_cache.get(cache_key, expiry_hours=12)
            
            if player_stats is None:
                print(f"Analyzing stability for {player_name}...")
                today = datetime.now()
                # Use a larger window for stability (365 days) but weight recent more? 
                # For now, 365 days with regression to mean is most stable.
                start = (today - timedelta(days=365)).strftime("%Y-%m-%d")
                df = statcast_batter(start, today.strftime("%Y-%m-%d"), pid)
                
                if df.empty:
                    hr_rate = self.league_avg_hr_rate
                    avg_pa = self.league_avg_pa
                else:
                    pa_total = len(df)
                    hr_total = len(df[df['events'] == 'home_run'])
                    raw_hr_rate = hr_total / max(pa_total, 1)
                    
                    # STABILITY: Regression to Mean (Credibility Theory)
                    # We need ~500 PAs to 'believe' a HR rate.
                    credibility = min(pa_total / 500, 1.0)
                    hr_rate = (raw_hr_rate * credibility) + (self.league_avg_hr_rate * (1 - credibility))
                    avg_pa = pa_total / max(df['game_date'].nunique(), 1)
                
                player_stats = {'hr_rate': hr_rate, 'avg_pa': avg_pa}
                self.disk_cache.set(cache_key, player_stats)

            # CALCULATION: Prob of >= 1 HR = 1 - e^(-lambda * p)
            # This is mathematically stable and doesn't rely on random simulations.
            lambda_val = player_stats['avg_pa']
            p_val = player_stats['hr_rate']
            
            # Apply Environment Adjustments
            adj = (1 + (float(hr9) - 1.1) * 0.25) * float(park)
            if kwargs.get('is_home'): adj *= 1.05
            
            # Scaled lambda_p
            stable_prob = 1 - math.exp(-(lambda_val * p_val * adj))
            
            # Professional Caps
            final_prob = min(max(stable_prob, 0.04), 0.35)
            implied = 100 / (float(odds) + 100)
            edge = final_prob - implied
            ev = (final_prob * (float(odds)/100)) - (1 - final_prob)

            prompt = f"MLB Analysis: {player_name} (HR Rate: {p_val:.1%}) vs Pitcher HR/9 {hr9}. Park: {park}. Final Prob: {final_prob:.1%}. Edge: {edge:+.1%}. Provide deep breakdown and ### OVERALL MATCH PREDICTION ###"
            
            return {
                "metrics": {
                    "probability": f"{final_prob:.1%}",
                    "implied_odds": f"{implied:.1%}",
                    "edge": f"{edge:+.1%}",
                    "ev": f"{ev:.2f}",
                    "recommended_bet": f"${max(ev * 50, 0):.2f}" # Half-Kelly inspired
                },
                "summary": self._groq_summary(prompt)
            }
        except Exception as e:
            return {"error": str(e)}

    def predict_pitcher(self, name, **kwargs):
        try:
            pid = self._get_player_id(name)
            if not pid: return {"error": "Pitcher not found"}
            cache_key = f"pitcher_v2_{pid}"
            data = self.disk_cache.get(cache_key, expiry_hours=12)
            if not data:
                today = datetime.now()
                start = (today - timedelta(days=180)).strftime("%Y-%m-%d")
                df = statcast_pitcher(start, today.strftime("%Y-%m-%d"), pid)
                if df.empty: data = {"hr9": 1.2, "score": 50}
                else:
                    hrs = len(df[df['events'] == 'home_run'])
                    outs = len(df[df['events'].isin(['field_out','strikeout','double_play','force_out'])])
                    hr9_raw = (hrs / max(outs/3.0, 1)) * 9
                    # Regression for pitcher
                    hr9 = (hr9_raw * 0.7) + (1.1 * 0.3)
                    data = {"hr9": round(hr9, 2), "score": round(min(hr9 * 40, 100), 1)}
                self.disk_cache.set(cache_key, data)

            prompt = f"Pitcher Analysis: {name}. HR/9: {data['hr9']}. Vulnerability: {data['score']}/100. Breakdown and ### OVERALL MATCH PREDICTION ###"
            return {
                "metrics": {"hr9": data['hr9'], "vulnerability_score": data['score']},
                "summary": self._groq_summary(prompt)
            }
        except Exception as e: return {"error": str(e)}

    def ballpark_factor(self, name, **kwargs):
        parks = {"Coors Field": 1.25, "Yankee Stadium": 1.18, "Fenway Park": 1.05, "Dodger Stadium": 1.10}
        factor = parks.get(name, 1.0)
        prompt = f"Ballpark: {name}. Factor: {factor}. Breakdown and ### OVERALL MATCH PREDICTION ###"
        return {"metrics": {"park_factor": factor}, "summary": self._groq_summary(prompt)}
