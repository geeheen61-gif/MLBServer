import os
import json
import time
import threading
import math
import unicodedata
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
        cache_key = f"id_v3_{clean_name.replace(' ', '_')}" # V3 for robust lookup
        cached_id = self.disk_cache.get(cache_key, expiry_hours=2160) # 90 days
        if cached_id: return cached_id

        print(f"🔎 Deep searching for player: {name}...")
        parts = name.split()
        if len(parts) < 1: return None
        
        last = parts[-1]
        first = parts[0] if len(parts) > 1 else ""

        try:
            # 1. Try exact match
            df = playerid_lookup(last, first)
            
            # 2. If no exact match, try fuzzy match
            if df.empty:
                print(f"⚠️ Exact match failed for {name}. Trying fuzzy search...")
                df = playerid_lookup(last, first, fuzzy=True)
            
            # 3. If still nothing, try just the last name
            if df.empty:
                print(f"⚠️ Fuzzy search failed. Searching last name only: {last}")
                df = playerid_lookup(last, fuzzy=True)

            if not df.empty:
                # Get the first result's ID
                pid = int(df.iloc[0]['key_mlbam'])
                self.disk_cache.set(cache_key, pid)
                return pid
        except Exception as e:
            print(f"❌ Player ID lookup error for {name}: {e}")
        
        return None

    def predict(self, player_name, odds, hr9, park, **kwargs):
        try:
            pid = self._get_player_id(player_name)
            if not pid: return {"error": f"Player '{player_name}' could not be identified in the database. Please check the spelling."}

            cache_key = f"stats_batter_{pid}"
            player_stats = self.disk_cache.get(cache_key, expiry_hours=12)
            
            if player_stats is None:
                print(f"📊 Fetching latest Statcast data for ID {pid}...")
                today = datetime.now()
                
                # Check last 365 days for enough sample size
                start = (today - timedelta(days=365)).strftime("%Y-%m-%d")
                df = statcast_batter(start, today.strftime("%Y-%m-%d"), pid)
                
                if df.empty:
                    # If still empty, try historical data to at least get a profile
                    print(f"⚠️ No data in last year for {player_name}. Checking historical record...")
                    start_hist = (today - timedelta(days=1000)).strftime("%Y-%m-%d")
                    df = statcast_batter(start_hist, today.strftime("%Y-%m-%d"), pid)

                if df.empty:
                    hr_rate = self.league_avg_hr_rate
                    avg_pa = self.league_avg_pa
                else:
                    df = df.dropna(subset=['events'])
                    pa_total = len(df)
                    hr_total = len(df[df['events'] == 'home_run'])
                    raw_hr_rate = hr_total / max(pa_total, 1)
                    
                    # Regression to mean
                    credibility = min(pa_total / 400, 1.0)
                    hr_rate = (raw_hr_rate * credibility) + (self.league_avg_hr_rate * (1 - credibility))
                    avg_pa = pa_total / max(df['game_date'].nunique(), 1)
                
                player_stats = {'hr_rate': hr_rate, 'avg_pa': avg_pa}
                self.disk_cache.set(cache_key, player_stats)

            # CALCULATION: 1 - e^(-lambda * p)
            lambda_val = player_stats['avg_pa']
            p_val = player_stats['hr_rate']
            adj = (1 + (float(hr9) - 1.1) * 0.25) * float(park)
            if kwargs.get('is_home'): adj *= 1.05
            
            stable_prob = 1 - math.exp(-(lambda_val * p_val * adj))
            final_prob = min(max(stable_prob, 0.02), 0.45)
            implied = 100 / (float(odds) + 100)
            edge = final_prob - implied
            ev = (final_prob * (float(odds)/100)) - (1 - final_prob)

            prompt = f"MLB Analysis: {player_name} (HR Rate: {p_val:.1%}) vs Pitcher HR/9 {hr9}. Park: {park}. Prob: {final_prob:.1%}. Edge: {edge:+.1%}. Provide deep breakdown and ### OVERALL MATCH PREDICTION ###"
            
            return {
                "metrics": {
                    "probability": f"{final_prob:.1%}",
                    "implied_odds": f"{implied:.1%}",
                    "edge": f"{edge:+.1%}",
                    "ev": f"{ev:.2f}",
                    "recommended_bet": f"${max(ev * 60, 0):.2f}"
                },
                "summary": self._groq_summary(prompt)
            }
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}

    def predict_pitcher(self, name, **kwargs):
        try:
            pid = self._get_player_id(name)
            if not pid: return {"error": f"Pitcher '{name}' not found."}
            
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
