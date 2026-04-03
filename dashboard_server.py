#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  STORMCHASER v5.2 — Weather Prediction Market Trading Bot      ║
║  3-Bot Swarm · Kelly Sizing · GFS Ensemble · Kalshi Demo       ║
╚══════════════════════════════════════════════════════════════════╝

Architecture:
  - Center Fader:      Fades overpriced brackets when ensemble disagrees with market
  - Tail Sniper:       Buys cheap tail contracts when ensemble says they're underpriced
  - Model Divergence:  Exploits lag between NWS forecast updates and market repricing

Run:  python3 dashboard_server.py
View: http://localhost:8000
"""

import os
import sys
import json
import time
import uuid
import math
import base64
import signal
import logging
import threading
from datetime import datetime, timezone, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import requests
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

# ─── Load config ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ─── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stormchaser")


# ═══════════════════════════════════════════════════════════════
# KALSHI API CLIENT — RSA-PSS Authentication
# ═══════════════════════════════════════════════════════════════

class KalshiClient:
    """Handles RSA-PSS signed requests to Kalshi REST API v2."""

    def __init__(self, key_id: str, private_key_path: str, base_url: str):
        self.key_id = key_id
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

        # Extract the path prefix from base_url for signing
        # e.g., "https://demo-api.kalshi.co/trade-api/v2" → "/trade-api/v2"
        from urllib.parse import urlparse
        parsed = urlparse(self.base_url)
        self.path_prefix = parsed.path.rstrip("/")  # "/trade-api/v2"

        # Load RSA private key
        try:
            with open(os.path.expanduser(private_key_path), "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None
                )
            log.info("RSA private key loaded successfully")
        except FileNotFoundError:
            log.error(f"Private key not found: {private_key_path}")
            log.error("Make sure private_key.pem is in your stormchaser folder")
            sys.exit(1)
        except Exception as e:
            log.error(f"Failed to load private key: {e}")
            sys.exit(1)

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        message = f"{timestamp_ms}{method}{path}".encode("utf-8")
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self, method: str, path: str) -> dict:
        # Kalshi requires signing the FULL path: /trade-api/v2/portfolio/balance
        full_sign_path = f"{self.path_prefix}{path}".split("?")[0]  # Strip query params
        ts = str(int(time.time() * 1000))
        sig = self._sign(ts, method, full_sign_path)
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    def get(self, path: str, params: dict = None) -> dict:
        url = f"{self.base_url}{path}"
        headers = self._auth_headers("GET", path)
        try:
            r = self.session.get(url, headers=headers, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            log.warning(f"GET {path} → {e.response.status_code}: {e.response.text[:200]}")
            return {}
        except Exception as e:
            log.warning(f"GET {path} failed: {e}")
            return {}

    def post(self, path: str, data: dict = None) -> dict:
        url = f"{self.base_url}{path}"
        headers = self._auth_headers("POST", path)
        try:
            r = self.session.post(url, headers=headers, json=data or {}, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            log.warning(f"POST {path} → {e.response.status_code}: {e.response.text[:200]}")
            return {}
        except Exception as e:
            log.warning(f"POST {path} failed: {e}")
            return {}

    def delete(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        headers = self._auth_headers("DELETE", path)
        try:
            r = self.session.delete(url, headers=headers, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning(f"DELETE {path} failed: {e}")
            return {}

    def get_balance(self) -> float:
        data = self.get("/portfolio/balance")
        # Balance returned in cents
        bal = data.get("balance", 0)
        return bal / 100.0 if isinstance(bal, (int, float)) and bal > 1 else bal

    def get_positions(self) -> list:
        data = self.get("/portfolio/positions")
        return data.get("positions", [])

    def get_markets(self, series_ticker: str) -> list:
        """Get all open markets for a series (e.g., KXHIGHNY)."""
        data = self.get("/markets", params={
            "series_ticker": series_ticker,
            "status": "open",
            "limit": 50,
        })
        return data.get("markets", [])

    def get_orderbook(self, ticker: str) -> dict:
        data = self.get(f"/markets/{ticker}/orderbook")
        return data.get("orderbook_fp", data.get("orderbook", {}))

    def place_order(self, ticker: str, side: str, action: str,
                    count: int, price_dollars: str) -> dict:
        order = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count_fp": f"{count}.00",
            "yes_price_dollars": price_dollars,
            "client_order_id": str(uuid.uuid4()),
        }
        log.info(f"ORDER: {action} {count}x {side} {ticker} @ ${price_dollars}")
        return self.post("/portfolio/orders", data=order)


# ═══════════════════════════════════════════════════════════════
# WEATHER DATA — NWS + Open-Meteo GFS Ensemble
# ═══════════════════════════════════════════════════════════════

class WeatherEngine:
    """Fetches NWS point forecasts and GFS 31-member ensemble data."""

    NWS_BASE = "https://api.weather.gov"
    OPENMETEO_ENSEMBLE = "https://ensemble-api.open-meteo.com/v1/ensemble"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Stormchaser/5.2 (weather trading bot)",
            "Accept": "application/json",
        })
        self._nws_cache = {}  # city -> (timestamp, data)
        self._ensemble_cache = {}

    def get_nws_forecast(self, city_key: str) -> dict:
        """Get NWS point forecast for a city. Returns high temp forecast."""
        city = config.CITIES[city_key]
        cache_key = city_key
        now = time.time()

        # Cache for 10 minutes
        if cache_key in self._nws_cache:
            ts, data = self._nws_cache[cache_key]
            if now - ts < 600:
                return data

        try:
            # Step 1: Get grid point
            r = self.session.get(
                f"{self.NWS_BASE}/points/{city['lat']},{city['lon']}",
                timeout=10,
            )
            r.raise_for_status()
            grid = r.json()["properties"]
            forecast_url = grid["forecast"]

            # Step 2: Get forecast
            r2 = self.session.get(forecast_url, timeout=10)
            r2.raise_for_status()
            periods = r2.json()["properties"]["periods"]

            # Find today's daytime forecast
            for p in periods:
                if p.get("isDaytime", False):
                    result = {
                        "high_f": p["temperature"],
                        "short_forecast": p.get("shortForecast", ""),
                        "detail": p.get("detailedForecast", ""),
                        "update_time": r2.json()["properties"].get("updateTime", ""),
                        "city": city_key,
                    }
                    self._nws_cache[cache_key] = (now, result)
                    return result

            return {"high_f": None, "city": city_key, "error": "No daytime forecast found"}

        except Exception as e:
            log.warning(f"NWS forecast failed for {city_key}: {e}")
            return {"high_f": None, "city": city_key, "error": str(e)}

    def get_ensemble_forecast(self, city_key: str) -> dict:
        """Get GFS 31-member ensemble forecast from Open-Meteo.
        Returns probability distribution of max temperature."""
        city = config.CITIES[city_key]
        now = time.time()

        if city_key in self._ensemble_cache:
            ts, data = self._ensemble_cache[city_key]
            if now - ts < 600:
                return data

        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

            r = self.session.get(self.OPENMETEO_ENSEMBLE, params={
                "latitude": city["lat"],
                "longitude": city["lon"],
                "daily": "temperature_2m_max",
                "temperature_unit": "fahrenheit",
                "start_date": today,
                "end_date": tomorrow,
                "models": "gfs_seamless",
            }, timeout=15)
            r.raise_for_status()
            data = r.json()

            # Extract all ensemble member forecasts for today
            daily = data.get("daily", {})
            members = []

            # The ensemble API returns temperature_2m_max for each member
            # as temperature_2m_max, temperature_2m_max_member01, etc.
            for key, values in daily.items():
                if key.startswith("temperature_2m_max") and isinstance(values, list) and len(values) > 0:
                    val = values[0]  # Today's value
                    if val is not None:
                        members.append(val)

            if not members:
                # Fallback: check if data is nested differently
                log.warning(f"No ensemble members found for {city_key}, raw keys: {list(daily.keys())}")
                return {"members": [], "city": city_key, "mean": None, "std": None}

            mean_temp = sum(members) / len(members)
            variance = sum((m - mean_temp) ** 2 for m in members) / len(members)
            std_temp = math.sqrt(variance) if variance > 0 else 1.0

            result = {
                "members": members,
                "count": len(members),
                "mean": round(mean_temp, 1),
                "std": round(std_temp, 1),
                "min": round(min(members), 1),
                "max": round(max(members), 1),
                "city": city_key,
            }
            self._ensemble_cache[city_key] = (now, result)
            return result

        except Exception as e:
            log.warning(f"Ensemble forecast failed for {city_key}: {e}")
            return {"members": [], "city": city_key, "mean": None, "std": None, "error": str(e)}

    def probability_above(self, members: list, threshold: float) -> float:
        """What fraction of ensemble members predict above threshold?"""
        if not members:
            return 0.5
        above = sum(1 for m in members if m >= threshold)
        return above / len(members)

    def probability_in_range(self, members: list, low: float, high: float) -> float:
        """What fraction of ensemble members predict within [low, high)?"""
        if not members:
            return 0.0
        count = sum(1 for m in members if low <= m < high)
        return count / len(members)


# ═══════════════════════════════════════════════════════════════
# MARKET PARSER — Extract bracket info from Kalshi tickers
# ═══════════════════════════════════════════════════════════════

def parse_temperature_market(market: dict) -> dict:
    """Extract temperature bracket info from a Kalshi market object."""
    ticker = market.get("ticker", "")
    title = market.get("title", "")
    subtitle = market.get("subtitle", "")

    # Extract threshold from ticker: e.g., KXHIGHNY-26APR01-T72 → 72
    # Or from floor/cap strike values in the market data
    floor_strike = market.get("floor_strike")
    cap_strike = market.get("cap_strike")

    # Try to get bracket from title/subtitle
    bracket_low = None
    bracket_high = None

    if floor_strike is not None and cap_strike is not None:
        bracket_low = float(floor_strike) if floor_strike else None
        bracket_high = float(cap_strike) if cap_strike else None
    else:
        # Parse from title like "72°F to 74°F" or "above 80°F"
        import re
        nums = re.findall(r'(\d+)°?F?', title + " " + subtitle)
        if len(nums) >= 2:
            bracket_low = float(nums[0])
            bracket_high = float(nums[1])
        elif len(nums) == 1:
            # Single threshold — "above X" or "below X"
            bracket_low = float(nums[0])
            bracket_high = bracket_low + 2  # Assume 2-degree bracket

    # Get prices (March 2026: dollar strings)
    yes_bid = market.get("yes_bid_dollars") or market.get("yes_bid")
    yes_ask = market.get("yes_ask_dollars") or market.get("yes_ask")
    no_bid = market.get("no_bid_dollars") or market.get("no_bid")
    no_ask = market.get("no_ask_dollars") or market.get("no_ask")

    def safe_float(v):
        if v is None:
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    return {
        "ticker": ticker,
        "title": title,
        "bracket_low": bracket_low,
        "bracket_high": bracket_high,
        "yes_bid": safe_float(yes_bid),
        "yes_ask": safe_float(yes_ask),
        "no_bid": safe_float(no_bid),
        "no_ask": safe_float(no_ask),
        "volume": market.get("volume", 0),
        "open_interest": market.get("open_interest", 0),
        "status": market.get("status", ""),
        "expiration": market.get("expiration_time", ""),
    }


# ═══════════════════════════════════════════════════════════════
# BOT BASE CLASS
# ═══════════════════════════════════════════════════════════════

class Bot:
    """Base class for trading bots with individual P&L and kill switch."""

    def __init__(self, name: str, allocation_pct: float):
        self.name = name
        self.allocation_pct = allocation_pct  # % of bankroll this bot can use
        self.trades = []
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.consecutive_losses = 0
        self.is_benched = False
        self.signals_found = 0
        self.signals_traded = 0

    @property
    def total_trades(self):
        return self.wins + self.losses

    @property
    def win_rate(self):
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    def check_bench(self):
        """Auto-bench if win rate drops below threshold after min trades."""
        if (self.total_trades >= config.BOT_BENCH_MIN_TRADES
                and self.win_rate < config.BOT_BENCH_THRESHOLD):
            if not self.is_benched:
                log.warning(f"BENCHING {self.name}: {self.win_rate:.0%} win rate "
                          f"after {self.total_trades} trades")
                self.is_benched = True

    def size_adjustment(self) -> float:
        """Kelly-inspired sizing: reduce after loss streaks."""
        if self.consecutive_losses >= config.LOSS_STREAK_THRESHOLD:
            return 0.5  # Half size after loss streak
        return 1.0

    def record_trade(self, won: bool, pnl: float, details: dict):
        self.trades.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "won": won,
            "pnl": round(pnl, 2),
            **details,
        })
        self.total_pnl += pnl
        if won:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1
        self.check_bench()

    def evaluate(self, market_info: dict, ensemble: dict, nws: dict) -> dict:
        """Override in subclass. Return signal dict or None."""
        raise NotImplementedError

    def status(self) -> dict:
        return {
            "name": self.name,
            "allocation": f"{self.allocation_pct:.0%}",
            "trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": f"{self.win_rate:.0%}" if self.total_trades > 0 else "N/A",
            "pnl": f"${self.total_pnl:+.2f}",
            "streak": f"{self.consecutive_losses} losses" if self.consecutive_losses > 0 else "OK",
            "benched": self.is_benched,
            "signals": self.signals_found,
            "traded": self.signals_traded,
        }


# ═══════════════════════════════════════════════════════════════
# BOT 1: CENTER FADER — The Anchor
# ═══════════════════════════════════════════════════════════════

class CenterFader(Bot):
    """Fades overpriced center brackets when ensemble disagrees.
    If the market prices a bracket at 60% but ensemble says 45%,
    sell YES (buy NO) on that bracket."""

    def __init__(self):
        super().__init__("Center Fader", allocation_pct=0.50)

    def evaluate(self, market_info: dict, ensemble: dict, nws: dict) -> dict:
        if not ensemble.get("members") or market_info["yes_bid"] is None:
            return None

        bracket_low = market_info.get("bracket_low")
        bracket_high = market_info.get("bracket_high")
        if bracket_low is None or bracket_high is None:
            return None

        # Model probability for this bracket
        model_prob = WeatherEngine().probability_in_range(
            ensemble["members"], bracket_low, bracket_high
        )

        # Market implied probability = yes_bid (what you'd sell at)
        market_prob = market_info["yes_bid"]

        # FADE: market thinks it's more likely than the model does
        if market_prob is not None and market_prob > 0.10:
            edge = market_prob - model_prob

            if edge >= config.MIN_EDGE_THRESHOLD:
                # Confidence: how tight is the ensemble?
                confidence = 1.0 - (ensemble.get("std", 5) / 10.0)
                confidence = max(0.0, min(1.0, confidence))

                if confidence >= config.MIN_CONFIDENCE:
                    self.signals_found += 1
                    return {
                        "bot": self.name,
                        "action": "sell_yes",
                        "side": "no",
                        "ticker": market_info["ticker"],
                        "edge": round(edge, 3),
                        "model_prob": round(model_prob, 3),
                        "market_prob": round(market_prob, 3),
                        "confidence": round(confidence, 3),
                        "bracket": f"{bracket_low}-{bracket_high}°F",
                        "price_dollars": f"{1.0 - market_prob:.4f}",
                    }
        return None


# ═══════════════════════════════════════════════════════════════
# BOT 2: TAIL SNIPER — High Reward Extremes
# ═══════════════════════════════════════════════════════════════

class TailSniper(Bot):
    """Buys cheap tail contracts when ensemble gives them higher
    probability than the market implies. Targets brackets priced
    below $0.15 where ensemble says probability is much higher."""

    def __init__(self):
        super().__init__("Tail Sniper", allocation_pct=0.20)

    def evaluate(self, market_info: dict, ensemble: dict, nws: dict) -> dict:
        if not ensemble.get("members") or market_info["yes_ask"] is None:
            return None

        bracket_low = market_info.get("bracket_low")
        bracket_high = market_info.get("bracket_high")
        if bracket_low is None or bracket_high is None:
            return None

        yes_ask = market_info["yes_ask"]

        # Only look at cheap contracts (tail events)
        if yes_ask is None or yes_ask > 0.20 or yes_ask < 0.02:
            return None

        # Model probability
        model_prob = WeatherEngine().probability_in_range(
            ensemble["members"], bracket_low, bracket_high
        )

        # Edge: model says higher probability than market ask
        edge = model_prob - yes_ask

        if edge >= config.MIN_EDGE_THRESHOLD:
            # Confidence based on how many members agree
            members_in_range = sum(
                1 for m in ensemble["members"]
                if bracket_low <= m < bracket_high
            )
            confidence = members_in_range / len(ensemble["members"])

            if confidence >= 0.10:  # At least ~3 members agree for tails
                self.signals_found += 1
                return {
                    "bot": self.name,
                    "action": "buy_yes",
                    "side": "yes",
                    "ticker": market_info["ticker"],
                    "edge": round(edge, 3),
                    "model_prob": round(model_prob, 3),
                    "market_prob": round(yes_ask, 3),
                    "confidence": round(confidence, 3),
                    "bracket": f"{bracket_low}-{bracket_high}°F",
                    "price_dollars": f"{yes_ask:.4f}",
                }
        return None


# ═══════════════════════════════════════════════════════════════
# BOT 3: MODEL DIVERGENCE — NWS Lag Exploiter
# ═══════════════════════════════════════════════════════════════

class ModelDivergence(Bot):
    """Exploits the information lag between NWS forecast updates
    and market repricing. When NWS forecast diverges significantly
    from ensemble mean, the market is likely still priced on old info."""

    def __init__(self):
        super().__init__("Model Divergence", allocation_pct=0.30)

    def evaluate(self, market_info: dict, ensemble: dict, nws: dict) -> dict:
        if not ensemble.get("members") or not nws.get("high_f"):
            return None

        bracket_low = market_info.get("bracket_low")
        bracket_high = market_info.get("bracket_high")
        if bracket_low is None or bracket_high is None:
            return None

        nws_temp = nws["high_f"]
        ensemble_mean = ensemble.get("mean")
        if ensemble_mean is None:
            return None

        # Divergence: NWS and ensemble disagree
        divergence = abs(nws_temp - ensemble_mean)

        # Only trade if there's meaningful divergence (>2°F)
        if divergence < 2.0:
            return None

        # Use the MORE RECENT data source (ensemble updates more often)
        # If ensemble shifted away from NWS, market is probably still
        # priced on the NWS number

        # Model probability using ensemble (fresher data)
        model_prob = WeatherEngine().probability_in_range(
            ensemble["members"], bracket_low, bracket_high
        )

        # Estimate what market is probably priced at using NWS-based prob
        # Simple normal distribution centered on NWS forecast
        nws_std = ensemble.get("std", 3.0)
        if nws_std < 0.5:
            nws_std = 0.5
        bracket_mid = (bracket_low + bracket_high) / 2.0
        z = (bracket_mid - nws_temp) / nws_std
        # Approximate normal CDF for the bracket
        nws_prob = math.exp(-0.5 * z * z) / (nws_std * math.sqrt(2 * math.pi))
        nws_prob *= (bracket_high - bracket_low)
        nws_prob = max(0.01, min(0.99, nws_prob))

        yes_ask = market_info.get("yes_ask")
        yes_bid = market_info.get("yes_bid")

        if yes_ask is not None and model_prob > yes_ask:
            edge = model_prob - yes_ask
            if edge >= config.MIN_EDGE_THRESHOLD:
                confidence = min(1.0, divergence / 5.0)  # More divergence = more confident
                if confidence >= config.MIN_CONFIDENCE:
                    self.signals_found += 1
                    return {
                        "bot": self.name,
                        "action": "buy_yes",
                        "side": "yes",
                        "ticker": market_info["ticker"],
                        "edge": round(edge, 3),
                        "model_prob": round(model_prob, 3),
                        "market_prob": round(yes_ask, 3),
                        "confidence": round(confidence, 3),
                        "bracket": f"{bracket_low}-{bracket_high}°F",
                        "nws_temp": nws_temp,
                        "ensemble_mean": ensemble_mean,
                        "divergence": round(divergence, 1),
                        "price_dollars": f"{yes_ask:.4f}",
                    }

        if yes_bid is not None and model_prob < yes_bid:
            edge = yes_bid - model_prob
            if edge >= config.MIN_EDGE_THRESHOLD:
                confidence = min(1.0, divergence / 5.0)
                if confidence >= config.MIN_CONFIDENCE:
                    self.signals_found += 1
                    return {
                        "bot": self.name,
                        "action": "sell_yes",
                        "side": "no",
                        "ticker": market_info["ticker"],
                        "edge": round(edge, 3),
                        "model_prob": round(model_prob, 3),
                        "market_prob": round(yes_bid, 3),
                        "confidence": round(confidence, 3),
                        "bracket": f"{bracket_low}-{bracket_high}°F",
                        "nws_temp": nws_temp,
                        "ensemble_mean": ensemble_mean,
                        "divergence": round(divergence, 1),
                        "price_dollars": f"{1.0 - yes_bid:.4f}",
                    }

        return None


# ═══════════════════════════════════════════════════════════════
# RISK MANAGER
# ═══════════════════════════════════════════════════════════════

class RiskManager:
    """Enforces bankroll limits, position sizing, and kill switches."""

    def __init__(self, starting_bankroll: float):
        self.starting_bankroll = starting_bankroll
        self.bankroll = starting_bankroll
        self.daily_loss = 0.0
        self.daily_loss_reset_date = datetime.now(timezone.utc).date()
        self.total_trades_today = 0
        self.city_exposure = {}  # city -> dollar amount
        self.halted = False
        self.halt_reason = ""

    def check_daily_reset(self):
        today = datetime.now(timezone.utc).date()
        if today != self.daily_loss_reset_date:
            self.daily_loss = 0.0
            self.total_trades_today = 0
            self.daily_loss_reset_date = today
            if self.halted and "daily" in self.halt_reason.lower():
                self.halted = False
                self.halt_reason = ""
                log.info("Daily loss counter reset — trading resumed")

    def can_trade(self) -> tuple:
        """Returns (allowed: bool, reason: str)."""
        self.check_daily_reset()

        if self.halted:
            return False, f"HALTED: {self.halt_reason}"

        # Hard drawdown floor
        floor = self.starting_bankroll * (1 - config.MAX_DRAWDOWN_PCT)
        if self.bankroll <= floor:
            self.halted = True
            self.halt_reason = f"Max drawdown hit (${self.bankroll:.2f} <= ${floor:.2f})"
            return False, self.halt_reason

        # Daily loss limit
        if self.daily_loss >= config.DAILY_LOSS_LIMIT:
            self.halted = True
            self.halt_reason = f"Daily loss limit (${self.daily_loss:.2f} >= ${config.DAILY_LOSS_LIMIT:.2f})"
            return False, self.halt_reason

        return True, "OK"

    def calculate_position_size(self, bot: Bot, edge: float, confidence: float) -> int:
        """Kelly Criterion sizing with caps."""
        # Kelly fraction: edge * confidence / (1 - edge)
        if edge <= 0 or confidence <= 0:
            return 0

        kelly = (edge * confidence) / max(0.01, 1.0 - edge)
        kelly = min(kelly, 0.15)  # Cap Kelly at 15%

        # Bot's allocation of bankroll
        bot_bankroll = self.bankroll * bot.allocation_pct

        # Apply loss streak adjustment
        kelly *= bot.size_adjustment()

        # Dollar amount
        dollar_amount = bot_bankroll * kelly

        # Cap at max risk per trade
        max_risk = self.bankroll * config.MAX_RISK_PER_TRADE
        dollar_amount = min(dollar_amount, max_risk)

        # Convert to contract count (each contract costs the price)
        # Minimum 1 contract, max from config
        contracts = max(1, min(int(dollar_amount), config.MAX_CONTRACTS_PER_TRADE))

        return contracts

    def check_city_concentration(self, city: str, amount: float) -> bool:
        """Check if adding this trade would exceed city concentration limit."""
        current = self.city_exposure.get(city, 0)
        limit = self.bankroll * config.CITY_CONCENTRATION_LIMIT
        return (current + amount) <= limit

    def record_result(self, pnl: float, city: str = None):
        self.bankroll += pnl
        if pnl < 0:
            self.daily_loss += abs(pnl)
        if city and pnl < 0:
            self.city_exposure[city] = self.city_exposure.get(city, 0) + abs(pnl)

    def status(self) -> dict:
        return {
            "bankroll": f"${self.bankroll:.2f}",
            "starting": f"${self.starting_bankroll:.2f}",
            "pnl": f"${self.bankroll - self.starting_bankroll:+.2f}",
            "pnl_pct": f"{((self.bankroll / self.starting_bankroll) - 1) * 100:+.1f}%",
            "daily_loss": f"${self.daily_loss:.2f} / ${config.DAILY_LOSS_LIMIT:.2f}",
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "drawdown_floor": f"${self.starting_bankroll * (1 - config.MAX_DRAWDOWN_PCT):.2f}",
            "trades_today": self.total_trades_today,
        }


# ═══════════════════════════════════════════════════════════════
# SWARM ENGINE — Orchestrates everything
# ═══════════════════════════════════════════════════════════════

class StormchaserSwarm:
    """Main trading engine. Coordinates bots, weather data, and risk."""

    def __init__(self):
        self.kalshi = KalshiClient(
            key_id=config.KALSHI_API_KEY_ID,
            private_key_path=config.KALSHI_PRIVATE_KEY_PATH,
            base_url=config.BASE_URL,
        )
        self.weather = WeatherEngine()
        self.risk = RiskManager(config.STARTING_BANKROLL)

        # The 3-bot swarm
        self.bots = [
            CenterFader(),
            TailSniper(),
            ModelDivergence(),
        ]

        self.scan_count = 0
        self.event_log = []
        self.traded_tickers_today = set()
        self._traded_reset_date = datetime.now(timezone.utc).date()
        self.running = True

        # Try to load saved state
        self._load_state()

    def _state_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "swarm_state.json")

    def _load_state(self):
        path = self._state_path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    state = json.load(f)
                if not isinstance(state, dict):
                    log.warning("Old state file format detected — starting fresh")
                    os.remove(path)
                    return
                self.risk.bankroll = state.get("bankroll", config.STARTING_BANKROLL)
                self.risk.daily_loss = state.get("daily_loss", 0)
                self.event_log = state.get("event_log", [])[-200:]  # Keep last 200
                for i, bot_state in enumerate(state.get("bots", [])):
                    if i < len(self.bots):
                        self.bots[i].wins = bot_state.get("wins", 0)
                        self.bots[i].losses = bot_state.get("losses", 0)
                        self.bots[i].total_pnl = bot_state.get("total_pnl", 0)
                        self.bots[i].consecutive_losses = bot_state.get("consecutive_losses", 0)
                        self.bots[i].is_benched = bot_state.get("is_benched", False)
                        self.bots[i].trades = bot_state.get("trades", [])[-50:]
                log.info(f"Loaded state: bankroll=${self.risk.bankroll:.2f}")
            except Exception as e:
                log.warning(f"Could not load state: {e} — starting fresh")

    def save_state(self):
        state = {
            "bankroll": self.risk.bankroll,
            "daily_loss": self.risk.daily_loss,
            "event_log": self.event_log[-200:],
            "bots": [{
                "name": b.name,
                "wins": b.wins,
                "losses": b.losses,
                "total_pnl": b.total_pnl,
                "consecutive_losses": b.consecutive_losses,
                "is_benched": b.is_benched,
                "trades": b.trades[-50:],
            } for b in self.bots],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(self._state_path(), "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save state: {e}")

    def log_event(self, msg: str, level: str = "info"):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = {"time": ts, "msg": msg, "level": level}
        self.event_log.append(entry)
        if len(self.event_log) > 500:
            self.event_log = self.event_log[-300:]
        getattr(log, level, log.info)(msg)

    def scan_city(self, city_key: str) -> list:
        """Scan a single city for trading signals."""
        city = config.CITIES[city_key]
        signals = []

        # Get weather data
        nws = self.weather.get_nws_forecast(city_key)
        ensemble = self.weather.get_ensemble_forecast(city_key)

        if not ensemble.get("members"):
            self.log_event(f"{city_key}: No ensemble data available", "warning")
            return signals

        nws_temp = nws.get("high_f", "?")
        self.log_event(
            f"{city_key}: NWS={nws_temp}°F  Ensemble={ensemble.get('mean', '?')}°F "
            f"(±{ensemble.get('std', '?')}°F, {ensemble.get('count', 0)} members)"
        )

        # Get markets for this city's series
        markets = self.kalshi.get_markets(city["series"])
        if not markets:
            self.log_event(f"{city_key}: No open markets found for {city['series']}", "warning")
            return signals

        self.log_event(f"{city_key}: Found {len(markets)} open markets")

        for market in markets:
            minfo = parse_temperature_market(market)

            if not minfo["bracket_low"] or not minfo["bracket_high"]:
                continue

            # Check deduplication
            if minfo["ticker"] in self.traded_tickers_today:
                continue

            # Run each bot
            for bot in self.bots:
                if bot.is_benched:
                    continue

                signal = bot.evaluate(minfo, ensemble, nws)
                if signal:
                    signal["city"] = city_key
                    signals.append(signal)

        return signals

    def execute_signal(self, signal: dict) -> bool:
        """Execute a trading signal through the Kalshi API."""
        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            self.log_event(f"BLOCKED: {reason}", "warning")
            return False

        # Find the bot
        bot = None
        for b in self.bots:
            if b.name == signal["bot"]:
                bot = b
                break
        if not bot:
            return False

        # Size the trade
        contracts = self.risk.calculate_position_size(
            bot, signal["edge"], signal["confidence"]
        )
        if contracts < 1:
            self.log_event(f"Skip {signal['ticker']}: position too small")
            return False

        # Check city concentration
        price = float(signal["price_dollars"])
        trade_cost = contracts * price
        if not self.risk.check_city_concentration(signal["city"], trade_cost):
            self.log_event(f"Skip {signal['ticker']}: city concentration limit", "warning")
            return False

        # Place the order
        result = self.kalshi.place_order(
            ticker=signal["ticker"],
            side=signal["side"],
            action="buy",
            count=contracts,
            price_dollars=signal["price_dollars"],
        )

        if result:
            signal["contracts"] = contracts
            signal["cost"] = round(trade_cost, 2)
            bot.signals_traded += 1
            self.traded_tickers_today.add(signal["ticker"])
            self.risk.total_trades_today += 1
            self.log_event(
                f"TRADE: {signal['bot']} | {signal['action']} {contracts}x "
                f"{signal['ticker']} @ ${signal['price_dollars']} | "
                f"Edge={signal['edge']:.1%} Conf={signal['confidence']:.1%}",
                "info"
            )
            return True
        else:
            self.log_event(f"ORDER FAILED: {signal['ticker']}", "warning")
            return False

    def run_scan(self):
        """Run a full scan across all cities."""
        self.scan_count += 1

        # Reset daily ticker dedup
        today = datetime.now(timezone.utc).date()
        if today != self._traded_reset_date:
            self.traded_tickers_today.clear()
            self._traded_reset_date = today

        self.log_event(f"═══ SCAN #{self.scan_count} ═══")

        # Check if we can trade at all
        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            self.log_event(f"HALTED: {reason}", "warning")
            self.save_state()
            return

        # Update bankroll from Kalshi balance
        try:
            live_balance = self.kalshi.get_balance()
            if live_balance and live_balance > 0:
                self.risk.bankroll = live_balance
                self.log_event(f"Balance: ${live_balance:.2f}")
        except Exception:
            pass

        all_signals = []
        for city_key in config.CITIES:
            try:
                signals = self.scan_city(city_key)
                all_signals.extend(signals)
            except Exception as e:
                self.log_event(f"{city_key}: Scan error — {e}", "warning")

        # Sort by edge * confidence (best signals first)
        all_signals.sort(key=lambda s: s["edge"] * s["confidence"], reverse=True)

        # Execute top signals
        executed = 0
        for signal in all_signals:
            if self.execute_signal(signal):
                executed += 1

        self.log_event(
            f"Scan complete: {len(all_signals)} signals found, {executed} traded"
        )

        # Check settlements / position updates
        self._check_positions()

        self.save_state()

    def _check_positions(self):
        """Check current positions for settled contracts."""
        try:
            positions = self.kalshi.get_positions()
            if positions:
                for pos in positions:
                    ticker = pos.get("ticker", "")
                    settlement = pos.get("settlement_value")
                    if settlement is not None:
                        # Position has settled
                        pnl = float(settlement) * pos.get("position", 0)
                        self.log_event(f"SETTLED: {ticker} → ${pnl:+.2f}")
                        self.risk.record_result(pnl)

                        # Record to the appropriate bot (simplified — record to first matching)
                        for bot in self.bots:
                            for trade in bot.trades:
                                if trade.get("ticker") == ticker and not trade.get("settled"):
                                    trade["settled"] = True
                                    bot.record_trade(pnl > 0, pnl, {"ticker": ticker})
                                    break
        except Exception as e:
            log.debug(f"Position check: {e}")

    def get_dashboard_data(self) -> dict:
        """Compile all data for the dashboard."""
        return {
            "version": "5.2",
            "scan_count": self.scan_count,
            "risk": self.risk.status(),
            "bots": [b.status() for b in self.bots],
            "event_log": self.event_log[-100:],
            "cities": {k: v["label"] for k, v in config.CITIES.items()},
            "config": {
                "min_edge": f"{config.MIN_EDGE_THRESHOLD:.0%}",
                "min_confidence": f"{config.MIN_CONFIDENCE:.0%}",
                "scan_interval": f"{config.SCAN_INTERVAL_SECONDS}s",
                "max_risk_per_trade": f"{config.MAX_RISK_PER_TRADE:.1%}",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ═══════════════════════════════════════════════════════════════
# DASHBOARD SERVER — Embedded HTML
# ═══════════════════════════════════════════════════════════════

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stormchaser v5.2</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  :root {
    --bg: #0a0a0f; --surface: #12121a; --border: #1e1e2e;
    --text: #e0e0e8; --muted: #6b6b80; --accent: #4fc3f7;
    --green: #66bb6a; --red: #ef5350; --yellow: #ffd54f;
    --blue: #42a5f5;
  }
  body {
    font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
    background: var(--bg); color: var(--text);
    padding: 20px; max-width: 1400px; margin: 0 auto;
  }
  .header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 16px 20px; background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; margin-bottom: 16px;
  }
  .header h1 { font-size: 18px; color: var(--accent); font-weight: 600; }
  .header .meta { font-size: 12px; color: var(--muted); }
  .status-badge {
    display: inline-block; padding: 4px 12px; border-radius: 4px;
    font-size: 12px; font-weight: 600; text-transform: uppercase;
  }
  .status-live { background: rgba(102,187,106,0.15); color: var(--green); }
  .status-halted { background: rgba(239,83,80,0.15); color: var(--red); }

  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 16px; }

  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
  }
  .card h2 {
    font-size: 13px; color: var(--muted); text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 12px;
  }
  .big-number { font-size: 32px; font-weight: 700; }
  .big-number.green { color: var(--green); }
  .big-number.red { color: var(--red); }
  .big-number.neutral { color: var(--text); }

  .stat-row {
    display: flex; justify-content: space-between; padding: 6px 0;
    border-bottom: 1px solid var(--border); font-size: 13px;
  }
  .stat-row:last-child { border-bottom: none; }
  .stat-label { color: var(--muted); }
  .stat-value { font-weight: 600; }

  .bot-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; position: relative;
  }
  .bot-card.benched { opacity: 0.5; }
  .bot-card .bot-name { font-size: 14px; font-weight: 600; color: var(--accent); margin-bottom: 8px; }
  .bot-card .bot-benched {
    position: absolute; top: 8px; right: 12px;
    color: var(--red); font-size: 11px; font-weight: 600;
  }

  .event-log {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; max-height: 400px; overflow-y: auto;
  }
  .event-log h2 {
    font-size: 13px; color: var(--muted); text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 12px;
  }
  .log-entry {
    font-size: 12px; padding: 4px 0; border-bottom: 1px solid rgba(30,30,46,0.5);
    display: flex; gap: 8px;
  }
  .log-entry .log-time { color: var(--muted); min-width: 65px; }
  .log-entry .log-msg { flex: 1; }
  .log-entry.warning .log-msg { color: var(--yellow); }
  .log-entry.trade .log-msg { color: var(--green); }

  .footer { text-align: center; padding: 16px; color: var(--muted); font-size: 11px; }
  @media (max-width: 768px) { .grid, .grid-3 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
  <div class="header">
    <div>
      <h1>⛈ STORMCHASER v5.2</h1>
      <div class="meta">Weather Prediction Market Bot · Kalshi Demo</div>
    </div>
    <div>
      <span class="status-badge" id="status">LOADING...</span>
      <div class="meta" id="scan-count" style="margin-top:4px;text-align:right;"></div>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Bankroll</h2>
      <div class="big-number" id="bankroll">—</div>
      <div style="margin-top:8px;">
        <div class="stat-row"><span class="stat-label">Starting</span><span class="stat-value" id="starting">—</span></div>
        <div class="stat-row"><span class="stat-label">P&L</span><span class="stat-value" id="pnl">—</span></div>
        <div class="stat-row"><span class="stat-label">P&L %</span><span class="stat-value" id="pnl-pct">—</span></div>
      </div>
    </div>
    <div class="card">
      <h2>Risk Controls</h2>
      <div style="margin-top:8px;">
        <div class="stat-row"><span class="stat-label">Daily Loss</span><span class="stat-value" id="daily-loss">—</span></div>
        <div class="stat-row"><span class="stat-label">Drawdown Floor</span><span class="stat-value" id="dd-floor">—</span></div>
        <div class="stat-row"><span class="stat-label">Trades Today</span><span class="stat-value" id="trades-today">—</span></div>
        <div class="stat-row"><span class="stat-label">Min Edge</span><span class="stat-value" id="min-edge">—</span></div>
        <div class="stat-row"><span class="stat-label">Min Confidence</span><span class="stat-value" id="min-conf">—</span></div>
        <div class="stat-row"><span class="stat-label">Scan Interval</span><span class="stat-value" id="scan-int">—</span></div>
      </div>
    </div>
  </div>

  <div class="grid-3" id="bot-grid"></div>

  <div class="event-log">
    <h2>Event Log</h2>
    <div id="log-entries"></div>
  </div>

  <div class="footer">
    Auto-refreshes every 15s · Stormchaser v5.2 · Kalshi Demo Environment
  </div>

<script>
function updateDashboard() {
  fetch('/api/data')
    .then(r => r.json())
    .then(d => {
      // Status
      const halted = d.risk.halted;
      const statusEl = document.getElementById('status');
      statusEl.textContent = halted ? 'HALTED' : 'LIVE';
      statusEl.className = 'status-badge ' + (halted ? 'status-halted' : 'status-live');
      document.getElementById('scan-count').textContent = 'Scan #' + d.scan_count;

      // Bankroll
      const bankEl = document.getElementById('bankroll');
      bankEl.textContent = d.risk.bankroll;
      const pnlVal = parseFloat(d.risk.pnl.replace(/[^\\d.+-]/g, ''));
      bankEl.className = 'big-number ' + (pnlVal > 0 ? 'green' : pnlVal < 0 ? 'red' : 'neutral');

      document.getElementById('starting').textContent = d.risk.starting;
      const pnlEl = document.getElementById('pnl');
      pnlEl.textContent = d.risk.pnl;
      pnlEl.style.color = pnlVal > 0 ? 'var(--green)' : pnlVal < 0 ? 'var(--red)' : 'var(--text)';
      document.getElementById('pnl-pct').textContent = d.risk.pnl_pct;
      document.getElementById('daily-loss').textContent = d.risk.daily_loss;
      document.getElementById('dd-floor').textContent = d.risk.drawdown_floor;
      document.getElementById('trades-today').textContent = d.risk.trades_today;
      document.getElementById('min-edge').textContent = d.config.min_edge;
      document.getElementById('min-conf').textContent = d.config.min_confidence;
      document.getElementById('scan-int').textContent = d.config.scan_interval;

      // Bots
      const botGrid = document.getElementById('bot-grid');
      botGrid.innerHTML = '';
      d.bots.forEach(bot => {
        const card = document.createElement('div');
        card.className = 'bot-card' + (bot.benched ? ' benched' : '');
        const pnlColor = bot.pnl.includes('+') ? 'var(--green)' : bot.pnl.includes('-') ? 'var(--red)' : 'var(--text)';
        card.innerHTML = '<div class="bot-name">' + bot.name + '</div>'
          + (bot.benched ? '<div class="bot-benched">BENCHED</div>' : '')
          + '<div class="stat-row"><span class="stat-label">W/L</span><span class="stat-value">' + bot.wins + '/' + bot.losses + '</span></div>'
          + '<div class="stat-row"><span class="stat-label">Win Rate</span><span class="stat-value">' + bot.win_rate + '</span></div>'
          + '<div class="stat-row"><span class="stat-label">P&L</span><span class="stat-value" style="color:' + pnlColor + '">' + bot.pnl + '</span></div>'
          + '<div class="stat-row"><span class="stat-label">Allocation</span><span class="stat-value">' + bot.allocation + '</span></div>'
          + '<div class="stat-row"><span class="stat-label">Signals</span><span class="stat-value">' + bot.signals + ' found / ' + bot.traded + ' traded</span></div>'
          + '<div class="stat-row"><span class="stat-label">Streak</span><span class="stat-value">' + bot.streak + '</span></div>';
        botGrid.appendChild(card);
      });

      // Event Log (reverse — newest first)
      const logEl = document.getElementById('log-entries');
      logEl.innerHTML = '';
      const logs = d.event_log.slice().reverse();
      logs.forEach(e => {
        const entry = document.createElement('div');
        let cls = 'log-entry';
        if (e.level === 'warning') cls += ' warning';
        if (e.msg.includes('TRADE:')) cls += ' trade';
        entry.className = cls;
        entry.innerHTML = '<span class="log-time">' + e.time + '</span>'
          + '<span class="log-msg">' + e.msg + '</span>';
        logEl.appendChild(entry);
      });
    })
    .catch(err => {
      document.getElementById('status').textContent = 'OFFLINE';
      document.getElementById('status').className = 'status-badge status-halted';
    });
}

updateDashboard();
setInterval(updateDashboard, 15000);
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for the dashboard and API."""

    swarm = None  # Set by main()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())

        elif parsed.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = self.swarm.get_dashboard_data() if self.swarm else {}
            self.wfile.write(json.dumps(data).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress HTTP logs


# ═══════════════════════════════════════════════════════════════
# MAIN — Start everything
# ═══════════════════════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  ⛈  STORMCHASER v5.2                                       ║")
    print("║  Weather Prediction Market Trading Bot                      ║")
    print("║  3-Bot Swarm · Kelly Sizing · GFS Ensemble · Kalshi Demo   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Initialize swarm
    swarm = StormchaserSwarm()
    DashboardHandler.swarm = swarm

    # Verify API connection
    print("  Connecting to Kalshi API...")
    balance = swarm.kalshi.get_balance()
    if balance:
        print(f"  ✅ Connected — Balance: ${balance:.2f}")
        swarm.risk.bankroll = balance
    else:
        print(f"  ⚠️  Could not fetch balance — using config value: ${config.STARTING_BANKROLL:.2f}")
        print(f"      (This is normal if demo has no balance set)")

    print(f"\n  Bots: {', '.join(b.name for b in swarm.bots)}")
    print(f"  Cities: {', '.join(config.CITIES.keys())}")
    print(f"  Bankroll: ${swarm.risk.bankroll:.2f}")
    print(f"  Drawdown floor: ${swarm.risk.bankroll * (1 - config.MAX_DRAWDOWN_PCT):.2f}")
    print(f"  Scan interval: {config.SCAN_INTERVAL_SECONDS}s")
    print(f"  Dashboard: http://localhost:{config.DASHBOARD_PORT}")
    print()

    # Start dashboard server
    server = HTTPServer(("0.0.0.0", config.DASHBOARD_PORT), DashboardHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"  Dashboard live at http://localhost:{config.DASHBOARD_PORT}")
    print("  Press Ctrl+C to stop\n")

    # Graceful shutdown
    def shutdown(sig, frame):
        print("\n  Shutting down...")
        swarm.running = False
        swarm.save_state()
        server.shutdown()
        print("  State saved. Goodbye.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Main loop
    while swarm.running:
        try:
            swarm.run_scan()
            log.info(f"Next scan in {config.SCAN_INTERVAL_SECONDS}s...")
            time.sleep(config.SCAN_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            shutdown(None, None)
        except Exception as e:
            log.error(f"Scan error: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
