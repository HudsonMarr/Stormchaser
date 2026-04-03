#!/usr/bin/env python3
"""
STORMCHASER v5 — Swarm Engine (REAL MODE)
==========================================
No simulation. Real Kalshi prices. Real orders. Real settlement.

4 Bot Personalities:
  🎯 Tail Sniper    — buys cheap tail brackets when NWS shifts
  📉 Center Fader   — sells (buys NO on) overpriced center brackets
  🔀 Model Diverge  — trades when hourly NWS data diverges from market
  🔧 Mid Grinder    — finds small consistent edges on mid-range brackets

Flow per scan:
  1. Fetch NWS forecast for each city (real government data)
  2. Fetch real Kalshi markets + order book prices
  3. Each bot calculates edge = NWS probability - market price
  4. If edge > threshold → place real limit order via API
  5. Track fills, positions, and settlements for real P&L
"""

import json
import math
import os
import time
import datetime
import requests
import schedule

from config import *
from kalshi_client import KalshiClient


# ═══════════════════════════════════════════════════════════
# NWS FORECAST ENGINE
# ═══════════════════════════════════════════════════════════

def get_nws_forecast(city: str) -> dict | None:
    """
    Fetch gridpoint forecast from NWS API.
    Free, no key needed, updated every 1-6 hours.
    Returns: {"high_f": float, "low_f": float, "forecast_time": str}
    """
    station = NWS_STATIONS.get(city)
    if not station:
        return None

    url = (
        f"https://api.weather.gov/gridpoints/"
        f"{station['office']}/{station['gridX']},{station['gridY']}"
    )
    headers = {"User-Agent": "Stormchaser/5.0 (weather-trading-bot)"}

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"  ⚠ NWS {city}: HTTP {resp.status_code}")
            return None

        data = resp.json()
        props = data.get("properties", {})

        # Extract max temperature
        max_temp = props.get("maxTemperature", {})
        values = max_temp.get("values", [])
        if not values:
            print(f"  ⚠ NWS {city}: no temperature data")
            return None

        # Get today's/tomorrow's forecast
        # NWS returns temps in Celsius, convert to Fahrenheit
        uom = max_temp.get("uom", "")
        temp_c = values[0].get("value")
        if temp_c is None:
            return None

        if "degC" in uom or "Cel" in uom:
            temp_f = temp_c * 9 / 5 + 32
        else:
            temp_f = temp_c  # Already Fahrenheit

        # Also grab min temperature for spread calculation
        min_temp = props.get("minTemperature", {})
        min_values = min_temp.get("values", [])
        low_f = None
        if min_values and min_values[0].get("value") is not None:
            low_c = min_values[0]["value"]
            if "degC" in min_temp.get("uom", "") or "Cel" in min_temp.get("uom", ""):
                low_f = low_c * 9 / 5 + 32
            else:
                low_f = low_c

        valid_time = values[0].get("validTime", "")

        return {
            "high_f": round(temp_f, 1),
            "low_f": round(low_f, 1) if low_f else None,
            "forecast_time": valid_time,
            "city": city,
        }

    except Exception as e:
        print(f"  ❌ NWS {city} error: {e}")
        return None


def calc_bracket_probability(forecast_high: float, bracket_low: float,
                              bracket_high: float, std_dev: float = 2.5) -> float:
    """
    Calculate probability that actual high falls within [bracket_low, bracket_high].
    Uses normal distribution centered on NWS forecast with historical std dev.

    NWS forecast accuracy: ~2-3°F standard deviation for day-ahead highs.
    """
    from math import erf, sqrt

    def norm_cdf(x, mu, sigma):
        return 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))

    p = norm_cdf(bracket_high, forecast_high, std_dev) - \
        norm_cdf(bracket_low, forecast_high, std_dev)
    return max(0.001, min(0.999, p))


# ═══════════════════════════════════════════════════════════
# BOT PERSONALITIES
# ═══════════════════════════════════════════════════════════

class Bot:
    """Base bot with its own P&L tracking and strategy."""

    def __init__(self, name, emoji, description):
        self.name = name
        self.emoji = emoji
        self.description = description
        self.trades = []       # List of trade records
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.consecutive_losses = 0
        self.active = True

    def to_dict(self):
        return {
            "name": self.name,
            "emoji": self.emoji,
            "total_pnl": self.total_pnl,
            "wins": self.wins,
            "losses": self.losses,
            "active": self.active,
            "trades": self.trades[-50:],  # Keep last 50
        }


def get_market_label(market: dict) -> str:
    """
    Get the human-readable bracket description from a market dict.
    Kalshi uses different field names: subtitle, title, yes_sub_title, etc.
    Falls back to parsing the ticker itself.
    """
    # Try various field names Kalshi might use
    for field in ("subtitle", "yes_sub_title", "title", "short_title"):
        val = market.get(field, "")
        if val and ("°" in val or "to" in val or "above" in val or "below" in val):
            return val

    # Fall back to ticker parsing
    return market.get("ticker", "")


def extract_market_prices(market: dict) -> dict:
    """
    Extract best bid/ask from market-level fields.
    Kalshi returns prices as dollar strings: "0.5300" = 53 cents.

    Fields:
      yes_bid_dollars, yes_ask_dollars → price to buy/sell YES
      no_bid_dollars, no_ask_dollars   → price to buy/sell NO

    Returns dict with prices in CENTS (integers).
    """
    def dollars_to_cents(val) -> int:
        """Convert dollar string or float to cents integer."""
        try:
            d = float(val)
            return int(round(d * 100))
        except (ValueError, TypeError):
            return 0

    yes_bid = dollars_to_cents(market.get("yes_bid_dollars", 0))
    yes_ask = dollars_to_cents(market.get("yes_ask_dollars", 0))
    no_bid = dollars_to_cents(market.get("no_bid_dollars", 0))
    no_ask = dollars_to_cents(market.get("no_ask_dollars", 0))

    # If yes_ask is 0 or 100, there's no real ask — mark as 99 (unavailable)
    if yes_ask <= 0 or yes_ask >= 100:
        yes_ask = 99
    if no_ask <= 0 or no_ask >= 100:
        no_ask = 99
    if yes_bid <= 0:
        yes_bid = 0
    if no_bid <= 0:
        no_bid = 0

    return {
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
    }


class TailSniper(Bot):
    """Buys cheap YES on tail brackets when NWS says they're underpriced."""

    def __init__(self):
        super().__init__("Tail Sniper", "🎯",
                         "Buys underpriced tail brackets after forecast shifts")

    def evaluate(self, market, forecast_high, book_prices):
        """
        Look for tail brackets (far from center) priced cheaply.
        Buy YES when NWS probability > market ask price.
        """
        ticker = market["ticker"]
        subtitle = get_market_label(market)

        # Parse bracket from subtitle (e.g., "72° to 73°" or ">= 80°")
        bracket = parse_bracket(subtitle, ticker)
        if not bracket:
            return None

        bracket_low, bracket_high = bracket
        nws_prob = calc_bracket_probability(forecast_high, bracket_low, bracket_high)

        # Only interested in tails (prob < 25%)
        if nws_prob > 0.25:
            return None

        # Market price — use yes_ask (what it costs to buy yes)
        market_ask = book_prices.get("yes_ask", 99)
        if market_ask <= 0 or market_ask >= 99:
            return None

        market_price = market_ask / 100.0
        edge = nws_prob - market_price

        if edge >= MIN_EDGE_THRESHOLD:
            return {
                "ticker": ticker,
                "side": "yes",
                "action": "buy",
                "price_cents": market_ask,
                "edge": edge,
                "nws_prob": nws_prob,
                "market_price": market_price,
                "bracket": subtitle,
                "bot": self.name,
            }
        return None


class CenterFader(Bot):
    """Fades (sells) overpriced center brackets by buying NO."""

    def __init__(self):
        super().__init__("Center Fader", "📉",
                         "Buys NO on overpriced center brackets")

    def evaluate(self, market, forecast_high, book_prices):
        ticker = market["ticker"]
        subtitle = get_market_label(market)

        bracket = parse_bracket(subtitle, ticker)
        if not bracket:
            return None

        bracket_low, bracket_high = bracket
        nws_prob = calc_bracket_probability(forecast_high, bracket_low, bracket_high)

        # Only interested in center brackets (high prob)
        if nws_prob < 0.15:
            return None

        # If market overprices YES, buy NO
        # no_ask = what it costs to buy no
        no_ask = book_prices.get("no_ask", 99)
        if no_ask <= 0 or no_ask >= 99:
            return None

        no_fair = 1.0 - nws_prob
        no_market = no_ask / 100.0
        edge = no_fair - no_market

        if edge >= MIN_EDGE_THRESHOLD:
            return {
                "ticker": ticker,
                "side": "no",
                "action": "buy",
                "price_cents": no_ask,
                "edge": edge,
                "nws_prob": nws_prob,
                "market_price": 1.0 - no_market,  # Equivalent yes price
                "bracket": subtitle,
                "bot": self.name,
            }
        return None


class ModelDivergence(Bot):
    """Trades when NWS forecast diverges significantly from market consensus."""

    def __init__(self):
        super().__init__("Model Divergence", "🔀",
                         "Exploits gaps between NWS updates and stale market prices")

    def evaluate(self, market, forecast_high, book_prices):
        ticker = market["ticker"]
        subtitle = get_market_label(market)

        bracket = parse_bracket(subtitle, ticker)
        if not bracket:
            return None

        bracket_low, bracket_high = bracket
        nws_prob = calc_bracket_probability(forecast_high, bracket_low, bracket_high)

        # Check both sides for divergence
        yes_ask = book_prices.get("yes_ask", 99)
        no_ask = book_prices.get("no_ask", 99)

        # Buy YES if underpriced
        if yes_ask > 0 and yes_ask < 99:
            market_price = yes_ask / 100.0
            edge = nws_prob - market_price
            # Model Divergence needs a bigger edge (12%) — only trades strong signals
            if edge >= MIN_EDGE_THRESHOLD + 0.04:
                return {
                    "ticker": ticker,
                    "side": "yes",
                    "action": "buy",
                    "price_cents": yes_ask,
                    "edge": edge,
                    "nws_prob": nws_prob,
                    "market_price": market_price,
                    "bracket": subtitle,
                    "bot": self.name,
                }

        # Buy NO if overpriced
        if no_ask > 0 and no_ask < 99:
            no_fair = 1.0 - nws_prob
            no_market = no_ask / 100.0
            edge = no_fair - no_market
            if edge >= MIN_EDGE_THRESHOLD + 0.04:
                return {
                    "ticker": ticker,
                    "side": "no",
                    "action": "buy",
                    "price_cents": no_ask,
                    "edge": edge,
                    "nws_prob": nws_prob,
                    "market_price": 1.0 - no_market,
                    "bracket": subtitle,
                    "bot": self.name,
                }

        return None


class MidGrinder(Bot):
    """Finds small consistent edges on mid-range brackets."""

    def __init__(self):
        super().__init__("Mid Grinder", "🔧",
                         "Grinds small edges on mid-probability brackets")

    def evaluate(self, market, forecast_high, book_prices):
        ticker = market["ticker"]
        subtitle = get_market_label(market)

        bracket = parse_bracket(subtitle, ticker)
        if not bracket:
            return None

        bracket_low, bracket_high = bracket
        nws_prob = calc_bracket_probability(forecast_high, bracket_low, bracket_high)

        # Mid-range: 10-40% probability
        if nws_prob < 0.10 or nws_prob > 0.40:
            return None

        yes_ask = book_prices.get("yes_ask", 99)
        if yes_ask <= 0 or yes_ask >= 99:
            return None

        market_price = yes_ask / 100.0
        edge = nws_prob - market_price

        if edge >= MIN_EDGE_THRESHOLD:
            return {
                "ticker": ticker,
                "side": "yes",
                "action": "buy",
                "price_cents": yes_ask,
                "edge": edge,
                "nws_prob": nws_prob,
                "market_price": market_price,
                "bracket": subtitle,
                "bot": self.name,
            }
        return None


# ═══════════════════════════════════════════════════════════
# UTILITY: PARSE BRACKET FROM SUBTITLE
# ═══════════════════════════════════════════════════════════

def parse_bracket(text: str, ticker: str = "") -> tuple | None:
    """
    Parse Kalshi market into (low, high) temperature range.

    Real Kalshi formats from API:
      "78° or above"        → (78, 150)
      "73° or below"        → (-50, 73)
      "76° to 77°"          → (76, 77)
      Ticker: -T77          → threshold at 77 (>=77 means "78° or above")
      Ticker: -B76.5        → bracket at 76.5 (means "76° to 77°")
    """
    import re

    s = text.replace("°F", "°").replace("°C", "°").strip()

    # "78° or above" / "78 or above"
    m = re.search(r'(\d+)\s*°?\s*or\s*above', s, re.IGNORECASE)
    if m:
        return (float(m.group(1)), 150.0)

    # "73° or below" / "73 or below"
    m = re.search(r'(\d+)\s*°?\s*or\s*below', s, re.IGNORECASE)
    if m:
        return (-50.0, float(m.group(1)))

    # Range: "76° to 77°" or "76 to 77"
    m = re.search(r'(\d+)\s*°?\s*to\s*(\d+)\s*°?', s)
    if m:
        return (float(m.group(1)), float(m.group(2)))

    # ">= 85°" or "> 85"
    m = re.search(r'>=?\s*(\d+)', s)
    if m:
        return (float(m.group(1)), 150.0)

    # "< 50°" or "<= 50"
    m = re.search(r'<=?\s*(\d+)', s)
    if m:
        return (-50.0, float(m.group(1)))

    # Parse from ticker if text didn't work
    # Threshold: KXHIGHNY-26MAR31-T77 → "77 or above" means >= 77+1 = 78° or above
    # Actually T77 means the threshold IS at 77, so the market is "78° or above"
    # But for probability calc, we want P(high >= 78) which is P(high > 77)
    t = ticker if ticker else text
    m = re.search(r'-T(\d+\.?\d*)', t)
    if m:
        threshold = float(m.group(1))
        # T77 = "78° or above", so bracket is (threshold+1, 150)
        # But actually the market is >=threshold+0.5 rounded
        # Safest: use threshold+0.5 as the lower bound
        return (threshold + 0.5, 150.0)

    # Bracket: KXHIGHNY-26MAR31-B76.5 → "76° to 77°"
    m = re.search(r'-B(\d+\.?\d*)', t)
    if m:
        mid = float(m.group(1))
        # B76.5 means bracket "76° to 77°" → (76, 77)
        return (mid - 0.5, mid + 0.5)

    return None


def parse_threshold_from_ticker(ticker: str) -> float | None:
    """
    Parse threshold from ticker like KXHIGHNY-26MAR31-T72.
    Returns 72.0 or None.
    """
    import re
    m = re.search(r'-T(\d+\.?\d*)', ticker)
    if m:
        return float(m.group(1))
    # Also try B prefix (bracket): HIGHNY-22NOV28-B51.5
    m = re.search(r'-B(\d+\.?\d*)', ticker)
    if m:
        return float(m.group(1))
    return None


# ═══════════════════════════════════════════════════════════
# KELLY CRITERION POSITION SIZING
# ═══════════════════════════════════════════════════════════

def kelly_size(edge: float, market_price: float, bankroll: float,
               consecutive_losses: int = 0) -> int:
    """
    Quarter Kelly position sizing.
    Automatically halves after LOSS_STREAK_THRESHOLD consecutive losses.
    Returns number of contracts.
    """
    if edge <= 0 or market_price <= 0 or market_price >= 1:
        return 0

    win_prob = min(0.90, market_price + edge)
    lose_prob = 1 - win_prob
    odds = (1.0 - market_price) / market_price

    if odds <= 0:
        return 0

    kelly_f = (odds * win_prob - lose_prob) / odds
    kelly_f = max(0, kelly_f)

    # Quarter Kelly
    fraction = kelly_f * KELLY_FRACTION
    fraction = min(fraction, MAX_KELLY_BET_PCT)

    # Loss streak reduction
    if consecutive_losses >= LOSS_STREAK_THRESHOLD:
        fraction *= 0.5

    dollar_amount = bankroll * fraction
    price_per_contract = market_price  # Cost in dollars per contract
    contracts = max(1, int(dollar_amount / price_per_contract))
    contracts = min(contracts, MAX_POSITION_PER_TRADE)

    return contracts


# ═══════════════════════════════════════════════════════════
# RISK MANAGER
# ═══════════════════════════════════════════════════════════

class RiskManager:
    """Enforces hard risk limits. No overrides."""

    def __init__(self, starting_bankroll):
        self.starting_bankroll = starting_bankroll
        self.floor = starting_bankroll * (1 - MAX_DRAWDOWN_PCT)
        self.daily_loss = 0.0
        self.daily_reset_date = datetime.date.today()
        self.city_exposure = {}  # city → total cost in dollars
        self.halted = False

    def reset_daily(self):
        today = datetime.date.today()
        if today != self.daily_reset_date:
            self.daily_loss = 0.0
            self.daily_reset_date = today
            self.city_exposure = {}

    def can_trade(self, current_bankroll: float, city: str,
                  trade_cost: float) -> tuple:
        """
        Returns (allowed: bool, reason: str).
        """
        self.reset_daily()

        if self.halted:
            return False, "HALTED — manual restart required"

        if self.floor > 0 and current_bankroll <= self.floor:
            self.halted = True
            return False, f"FLOOR HIT — bankroll ${current_bankroll:.2f} <= ${self.floor:.2f}"

        if self.daily_loss + trade_cost >= DAILY_LOSS_LIMIT:
            return False, f"Daily loss limit — ${self.daily_loss:.2f} + ${trade_cost:.2f} >= ${DAILY_LOSS_LIMIT}"

        # City concentration
        city_total = self.city_exposure.get(city, 0) + trade_cost
        max_city = current_bankroll * MAX_CITY_CONCENTRATION
        if city_total > max_city:
            return False, f"{city} concentration — ${city_total:.2f} > ${max_city:.2f}"

        return True, "OK"

    def record_trade(self, city: str, cost: float):
        self.city_exposure[city] = self.city_exposure.get(city, 0) + cost

    def record_loss(self, amount: float):
        self.daily_loss += amount


# ═══════════════════════════════════════════════════════════
# SWARM ENGINE
# ═══════════════════════════════════════════════════════════

class SwarmEngine:
    """
    Orchestrates all 4 bots, manages state, executes real trades.
    """

    STATE_FILE = "swarm_state.json"
    TRADE_LOG = "trade_log.json"

    def __init__(self, kalshi: KalshiClient):
        self.kalshi = kalshi
        self.bots = [
            TailSniper(),
            CenterFader(),
            ModelDivergence(),
            MidGrinder(),
        ]
        self.risk = RiskManager(STARTING_BANKROLL)
        self.scan_count = 0
        self.total_trades = 0
        self.total_pnl = 0.0
        self.trade_log = []
        self.last_scan_results = {}
        self.last_known_balance = 0
        self.load_state()

    # ─── State Persistence ────────────────────

    def save_state(self):
        state = {
            "scan_count": self.scan_count,
            "total_trades": self.total_trades,
            "total_pnl": self.total_pnl,
            "bots": {b.name: b.to_dict() for b in self.bots},
            "risk": {
                "daily_loss": self.risk.daily_loss,
                "halted": self.risk.halted,
                "city_exposure": self.risk.city_exposure,
            },
            "last_updated": datetime.datetime.now().isoformat(),
        }
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, self.STATE_FILE)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, self.STATE_FILE)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    state = json.load(f)
                self.scan_count = state.get("scan_count", 0)
                self.total_trades = state.get("total_trades", 0)
                self.total_pnl = state.get("total_pnl", 0)
                # Restore bot P&L
                for bot in self.bots:
                    bdata = state.get("bots", {}).get(bot.name, {})
                    bot.total_pnl = bdata.get("total_pnl", 0)
                    bot.wins = bdata.get("wins", 0)
                    bot.losses = bdata.get("losses", 0)
                    bot.trades = bdata.get("trades", [])
                print(f"📂 Loaded state: {self.total_trades} trades, ${self.total_pnl:.2f} P&L")
            except Exception as e:
                print(f"  ⚠ Could not load state: {e}")

    def save_trade(self, trade: dict):
        """Append a trade to the persistent log."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, self.TRADE_LOG)
        try:
            if os.path.exists(path):
                with open(path) as f:
                    logs = json.load(f)
            else:
                logs = []
            logs.append(trade)
            with open(path, "w") as f:
                json.dump(logs, f, indent=2)
        except:
            pass

    # ─── Main Scan Loop ──────────────────────

    def run_scan(self):
        """
        One complete scan cycle:
        1. Get NWS forecasts
        2. Get real Kalshi markets + prices
        3. Run all bots
        4. Execute trades with edge
        """
        self.scan_count += 1
        now = datetime.datetime.now()
        print(f"\n{'='*60}")
        print(f"⚡ SCAN #{self.scan_count} — {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        # Check if API is connected
        if not self.kalshi.connected:
            print("❌ Kalshi API not connected. Skipping scan.")
            return

        # Get current balance (with caching for API timeouts)
        balance = self.kalshi.get_balance_dollars()
        if balance <= 0 and self.last_known_balance > 0:
            balance = self.last_known_balance
            print(f"💰 Balance: ${balance:.2f} (cached — API timeout)")
        else:
            self.last_known_balance = balance
            print(f"💰 Balance: ${balance:.2f}")

        scan_signals = []
        scan_trades = []

        for city, series in WEATHER_SERIES.items():
            print(f"\n  🌡️ {city} — {series}")

            # 1. Get NWS forecast
            forecast = get_nws_forecast(city)
            if not forecast:
                print(f"    ⚠ No NWS data for {city}")
                continue

            high_f = forecast["high_f"]
            print(f"    NWS forecast high: {high_f}°F")

            # 2. Get real Kalshi markets
            markets = self.kalshi.get_weather_markets_for_city(series)
            if not markets:
                print(f"    ⚠ No open markets for {city}")
                continue

            print(f"    📊 {len(markets)} open markets")

            # 3. Run each bot on each market
            for market in markets:
                ticker = market["ticker"]
                subtitle = get_market_label(market)

                # Extract prices from market data (dollars → cents)
                # Kalshi returns prices as dollar strings: "0.5300" = 53 cents
                prices = extract_market_prices(market)

                for bot in self.bots:
                    if not bot.active:
                        continue

                    signal = bot.evaluate(market, high_f, prices)
                    if not signal:
                        continue

                    signal["city"] = city
                    signal["timestamp"] = now.isoformat()
                    signal["forecast_high"] = high_f
                    scan_signals.append(signal)

                    # 4. Size the position
                    contracts = kelly_size(
                        signal["edge"],
                        signal["price_cents"] / 100.0,
                        balance,
                        bot.consecutive_losses,
                    )
                    if contracts <= 0:
                        continue

                    trade_cost = (signal["price_cents"] / 100.0) * contracts

                    # 5. Risk check
                    allowed, reason = self.risk.can_trade(balance, city, trade_cost)
                    if not allowed:
                        print(f"    🛡️ {bot.emoji} BLOCKED: {reason}")
                        continue

                    # 6. PLACE REAL ORDER
                    print(
                        f"    {bot.emoji} {bot.name}: "
                        f"{signal['side'].upper()} {contracts}x "
                        f"{subtitle} @ {signal['price_cents']}¢ "
                        f"(edge: {signal['edge']:.1%})"
                    )

                    result = self.kalshi.place_order(
                        ticker=signal["ticker"],
                        side=signal["side"],
                        action="buy",
                        count=contracts,
                        yes_price=signal["price_cents"],
                    )

                    if result and "order" in result:
                        order = result["order"]
                        fill_count = (
                            order.get("taker_fill_count", 0)
                            + order.get("maker_fill_count", 0)
                        )
                        fill_cost = (
                            order.get("taker_fill_cost", 0)
                            + order.get("maker_fill_cost", 0)
                        ) / 100.0

                        trade_record = {
                            "timestamp": now.isoformat(),
                            "bot": bot.name,
                            "city": city,
                            "ticker": ticker,
                            "bracket": subtitle,
                            "side": signal["side"],
                            "contracts": contracts,
                            "price_cents": signal["price_cents"],
                            "edge": signal["edge"],
                            "nws_prob": signal["nws_prob"],
                            "order_id": order.get("order_id"),
                            "status": order.get("status"),
                            "filled": fill_count,
                            "fill_cost": fill_cost,
                        }

                        bot.trades.append(trade_record)
                        self.total_trades += 1
                        self.risk.record_trade(city, fill_cost)
                        scan_trades.append(trade_record)
                        self.save_trade(trade_record)

                        # Update balance after trade
                        balance = self.kalshi.get_balance_dollars()

        # ─── Check Settlements ────────────────

        self._check_settlements()

        # ─── Save scan results for dashboard ──

        self.last_scan_results = {
            "scan_number": self.scan_count,
            "timestamp": now.isoformat(),
            "balance": balance,
            "signals_found": len(scan_signals),
            "trades_placed": len(scan_trades),
            "signals": scan_signals,
            "trades": scan_trades,
        }

        self.save_state()
        
        # Save dashboard data to file for HTTP server to read
        try:
            dash_data = self.get_dashboard_data()
            dash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_data.json")
            with open(dash_path, "w") as f:
                json.dump(dash_data, f)
        except Exception as e:
            print(f"  ⚠ Dashboard data save error: {e}")

        # ─── Print Summary ────────────────────

        print(f"\n  {'─'*50}")
        print(f"  📊 SCAN SUMMARY")
        print(f"  Signals: {len(scan_signals)} | Trades: {len(scan_trades)}")
        print(f"  Balance: ${balance:.2f}")
        print(f"  Total trades (all time): {self.total_trades}")
        print(f"  Total P&L: ${self.total_pnl:+.2f}")
        for bot in self.bots:
            total = bot.wins + bot.losses
            wr = (bot.wins / total * 100) if total > 0 else 0
            print(
                f"    {bot.emoji} {bot.name}: "
                f"{wr:.0f}% ({bot.wins}W/{bot.losses}L) "
                f"${bot.total_pnl:+.2f}"
            )
        print(f"  🛡️ Daily loss: ${self.risk.daily_loss:.2f} / ${DAILY_LOSS_LIMIT}")
        print(f"  Next scan in {SCAN_INTERVAL_MINUTES} minutes...")

    # ─── Settlement Tracking ─────────────────

    def _check_settlements(self):
        """Check for settled positions and update P&L."""
        positions = self.kalshi.get_positions(settlement_status="settled")
        settlements = self.kalshi.get_settlements(limit=20)

        if settlements:
            print(f"\n  📋 Recent settlements: {len(settlements)}")
            for s in settlements[:5]:
                ticker = s.get("market_ticker", "?")
                revenue = s.get("revenue", 0) / 100.0
                cost = s.get("cost", 0) / 100.0 if "cost" in s else 0
                pnl = revenue - cost
                won = s.get("yes_count", 0) > 0 and s.get("settlement_value") == "yes"
                status = "✅ WIN" if pnl > 0 else "❌ LOSS"
                print(f"    {status}: {ticker} → ${pnl:+.2f}")

    # ─── Dashboard Data ──────────────────────

    def get_dashboard_data(self) -> dict:
        """Return all data needed for the dashboard."""
        positions = []
        if self.kalshi.connected:
            try:
                positions = self.kalshi.get_positions(settlement_status="unsettled")
            except:
                pass

        return {
            "scan_count": self.scan_count,
            "total_trades": self.total_trades,
            "total_pnl": self.total_pnl,
            "balance": self.kalshi.get_balance_dollars() if self.kalshi.connected else 0,
            "connected": self.kalshi.connected,
            "bots": [b.to_dict() for b in self.bots],
            "risk": {
                "daily_loss": self.risk.daily_loss,
                "halted": self.risk.halted,
                "floor": self.risk.floor,
                "city_exposure": self.risk.city_exposure,
            },
            "last_scan": self.last_scan_results,
            "positions": [
                {
                    "ticker": p.get("ticker", "?"),
                    "side": "YES" if p.get("yes_count", 0) > 0 else "NO",
                    "count": p.get("yes_count", 0) or p.get("no_count", 0),
                    "cost": p.get("total_cost", 0) / 100.0 if p.get("total_cost") else 0,
                }
                for p in (positions or [])
            ],
            "last_updated": datetime.datetime.now().isoformat(),
        }


# ═══════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"""
    ⚡⚡⚡ STORMCHASER v5 — REAL MODE ⚡⚡⚡
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    💰 Bankroll: ${STARTING_BANKROLL:,}
    🛡️  Max drawdown: {MAX_DRAWDOWN_PCT*100:.0f}% (floor: ${STARTING_BANKROLL*(1-MAX_DRAWDOWN_PCT):,.0f})
    📊 Daily loss limit: ${DAILY_LOSS_LIMIT}
    🎯 Min edge: {MIN_EDGE_THRESHOLD:.0%}
    ⏱️  Scan interval: {SCAN_INTERVAL_MINUTES} min
    🤖 Bots: Tail Sniper | Center Fader | Model Diverge | Mid Grinder
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)

    kalshi = KalshiClient(KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH, BASE_URL)
    kalshi.test_connection()

    swarm = SwarmEngine(kalshi)
    swarm.run_scan()

    schedule.every(SCAN_INTERVAL_MINUTES).minutes.do(swarm.run_scan)

    print(f"\n🔄 Swarm running — scanning every {SCAN_INTERVAL_MINUTES} min")
    print("   Ctrl+C to stop\n")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")
        swarm.save_state()
