"""
Stormchaser v5.2 Configuration
"""

# ═══════════════════════════════════════════════════════════════
# KALSHI API CREDENTIALS
# ═══════════════════════════════════════════════════════════════
KALSHI_API_KEY_ID = "cfecfcd9-d9da-40f4-b297-178886c79350"
KALSHI_PRIVATE_KEY_PATH = "/Users/hudson/stormchaser/private_key.pem"

# Demo environment (paper money) — change to production URL when ready
BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"

# ═══════════════════════════════════════════════════════════════
# BANKROLL & RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════
STARTING_BANKROLL = 385.00       # Current demo balance
MAX_DRAWDOWN_PCT = 0.80      # 15% = hard stop at ~$327
DAILY_LOSS_LIMIT = 150        # Tighter for smaller bankroll
MAX_RISK_PER_TRADE = 0.025       # 2.5% per trade = ~$9.60 max
LOSS_STREAK_THRESHOLD = 3        # Halve sizes after 3 consecutive losses
BOT_BENCH_THRESHOLD = 0.50       # Bench bot if win rate < 50% after 15 trades
BOT_BENCH_MIN_TRADES = 15        # Min trades before benching applies

# ═══════════════════════════════════════════════════════════════
# SIGNAL FILTERS — v5.2 is PICKY
# ═══════════════════════════════════════════════════════════════
MIN_EDGE_THRESHOLD = 0.12        # 12% minimum edge (was 10% in v5)
MIN_CONFIDENCE = 0.65            # 65% minimum ensemble agreement
MAX_CONTRACTS_PER_TRADE = 20     # Cap per-trade contract count
CITY_CONCENTRATION_LIMIT = 0.35  # Max 35% of bankroll in one city

# ═══════════════════════════════════════════════════════════════
# SCANNING
# ═══════════════════════════════════════════════════════════════
SCAN_INTERVAL_SECONDS = 300      # 5-minute scan cycle
CITIES = {
    "NYC": {"lat": 40.7128, "lon": -74.0060, "nws_station": "KNYC",
            "series": "KXHIGHNY", "label": "New York"},
    "CHI": {"lat": 41.8781, "lon": -87.6298, "nws_station": "KORD",
            "series": "KXHIGHCHI", "label": "Chicago"},
    "MIA": {"lat": 25.7617, "lon": -80.1918, "nws_station": "KMIA",
            "series": "KXHIGHMIA", "label": "Miami"},
    "AUS": {"lat": 30.2672, "lon": -97.7431, "nws_station": "KAUS",
            "series": "KXHIGHAUS", "label": "Austin"},
}

# ═══════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════
DASHBOARD_PORT = 8000
