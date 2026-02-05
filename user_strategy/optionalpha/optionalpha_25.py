#!/usr/bin/env python
"""
Options Alpha 25 - Simplified Version (WebSocket)
==================================================

STRATEGY OVERVIEW:
------------------
A simplified momentum-based options buying strategy that trades ATM options
using a fixed 25-point target with breakeven trailing stop loss. The strategy
captures directional moves in the first hour of trading with simple profit
protection logic.

CORE CONCEPTS:
--------------
1. **Simple Timeframe Approach**:
   - 15-min candle (9:15-9:30): Determine ATM strike and entry levels

2. **Entry Logic**:
   - Waits for price to break above entry level with confirmation
   - Entry level for PE: Average(PE 15min low, CE 15min high)
   - Entry level for CE: Average(CE 15min low, CE 15min high)
   - Confirmation: Current 1min candle open AND close above entry level
   - Crossover: Previous 1min candle open OR close below entry level

3. **Fixed Target & Breakeven Stop Loss**:
   - Initial SL: 5% below entry level
   - Target: Entry price + 25 points (fixed)
   - Trailing Logic: When profit reaches 20 points, move SL to cost (breakeven)
   - SL only moves UP to breakeven, never down
   - Simple and predictable profit protection

TIMING & SCHEDULE:
------------------
09:15 - Market opens
09:30 - First 15-min candle closes → Calculate ATM strike & entry levels
09:31 - Strategy starts monitoring for entry conditions
14:15 - No new entries allowed after this time
15:00 - Force exit all positions
15:30 - Market closes

RISK MANAGEMENT:
----------------
- Capital: 80% of available funds per trade
- Max Trades: 2 completed trades per day (default)
- Stop Loss: 5% below entry level initially, moves to cost when profit reaches 20 points
- Target: Entry price + 25 points (fixed)
- Position Sizing: Calculated based on capital and lot size
- Only one position at a time (CE or PE, whichever breaks out first)

TECHNICAL FEATURES:
-------------------
- WebSocket-first architecture with HTTP fallback for price updates
- Auto-reconnection with exponential backoff (up to 10 attempts)
- Health monitoring every 30 seconds
- Stale data detection and automatic recovery
- Crash recovery with up to 3 restart attempts
- Order verification with retry logic (3 attempts)
- Thread-safe tick data management
- 1-minute candle building from WebSocket ticks

EXAMPLE TRADE FLOW:
-------------------
09:30 AM:
  - NIFTY 15min close: 22,510
  - ATM Strike: 22,500 (auto-calculated via optionsymbol API)
  - PE Symbol: NIFTY10FEB2622500PE
  - CE Symbol: NIFTY10FEB2622500CE
  - PE 15min: Low=140, High=160
  - CE 15min: Low=145, High=165
  - PE Entry Level: (140 + 165) / 2 = 152.50
  - CE Entry Level: (145 + 165) / 2 = 155.00

09:42 AM:
  - PE LTP crosses 152.50
  - Current 1min candle: Open=153, Close=154 (both > 152.50) ✓
  - Previous 1min candle: Open=151, Close=152 (at least one < 152.50) ✓
  - Entry confirmed! Buy PE @ 154.00
  - Initial SL: 152.50 × 0.95 = 144.88
  - Target: 154.00 + 25 = 179.00

10:05 AM:
  - PE LTP reaches 174.00
  - Profit: 174.00 - 154.00 = 20 points ✓
  - SL moves to COST: 154.00 (breakeven protection activated)
  - Target still: 179.00

10:20 AM:
  - PE LTP reaches 179.50
  - Target hit at 179.00
  - Exit @ 179.00
  - PnL: (179.00 - 154.00) × qty = 25 points × qty
  - Trade closed with full 25-point profit!

CONFIGURATION OPTIONS:
----------------------
Edit these constants at the top of the file:

INDEX = "NIFTY"                    # NIFTY, BANKNIFTY, SENSEX
EXPIRY_WEEK = 1                    # 1=current, 2=next week
CAPITAL_PERCENT = 0.80             # Use 80% of available capital
SL_PERCENT = 0.05                  # 5% initial stop loss
TARGET_POINTS = 25                 # Fixed 25-point profit target
BREAKEVEN_POINTS = 20              # Move SL to cost when profit reaches 20 points
MAX_COMPLETED_TRADES = 2           # Max trades per day
NO_NEW_ENTRY_TIME = "14:15"        # Last entry time
FORCE_EXIT_TIME = "14:59"          # Square off time

REQUIREMENTS:
-------------
- OpenAlgo API running on http://127.0.0.1:5003
- WebSocket server running on ws://127.0.0.1:8765
- API key configured in ~/.config/openalgo/config.json
- Live broker connection with sufficient margin
- Market data subscription for selected index

USAGE:
------
uv run python3 nifty_optionsalpha.py

LOGS:
-----
- Console: INFO level messages
- File: logs/strategy_YYYYMMDD_HHMMSS.log (if LOG_TO_FILE=True)

Author: Generated with Claude
Version: 25.0 (Simplified)
Last Updated: 2026-02-05
Base: Options Alpha Strategy v3.0
"""

import time
import threading
import logging
import sys
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, List, Callable
from collections import deque
from enum import Enum
import traceback
from pathlib import Path
import pandas as pd
from openalgo import api


# =============================================================================
# LOGGING SETUP
# =============================================================================

class StrategyLogger:
    """Custom logger with console and optional file output"""

    def __init__(self, name: str, log_to_file: bool = False, log_dir: str = "logs"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers

        # Console handler (always enabled)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler (optional)
        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(
                log_dir,
                f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '[%(asctime)s] %(levelname)s [%(funcName)s:%(lineno)d]: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
            self.logger.info(f"Logging to file: {log_file}")

    def info(self, msg): self.logger.info(msg)
    def debug(self, msg): self.logger.debug(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)
    def critical(self, msg): self.logger.critical(msg)


# =============================================================================
# CONNECTION STATE
# =============================================================================

class ConnectionState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


# =============================================================================
# USER CONFIGURATION - EDIT THESE VALUES
# =============================================================================

# OpenAlgo API Key - loaded from central config
from pathlib import Path
sys.path.insert(0, str(Path.home() / ".config/openalgo"))
try:
    from client import API_KEY
except ImportError:
    API_KEY = "0a7ea96180c4bffa62c7401005779eccbc3d3a0a26676bcd241f182979d6bd01"  # Fallback - edit ~/.config/openalgo/config.json

# OpenAlgo Server
HOST = "http://127.0.0.1:5003"
WS_URL = "ws://127.0.0.1:8765"

# Index to trade
INDEX = "NIFTY"                    # NIFTY, BANKNIFTY, SENSEX

# Expiry week (1 = current week, 2 = next week, etc.)
EXPIRY_WEEK = 1

# Capital & Position Sizing
CAPITAL_PERCENT = 0.90             # Use 90% of available capital

# Risk Management
SL_PERCENT = 0.05                  # 5% stop loss below entry level
TARGET_POINTS = 25                 # Fixed 25-point profit target
BREAKEVEN_POINTS = 20              # Move SL to cost when profit reaches this

# Trading Limits
MAX_COMPLETED_TRADES = 2           # Max completed (exited) trades per day
NO_NEW_ENTRY_TIME = "14:15"            # No new entries after this time
FORCE_EXIT_TIME = "14:59"              # Force exit all positions at this time

# Robustness Settings
LOG_TO_FILE = False                    # Enable file logging
LOG_DIR = "logs"                       # Log directory
MAX_RECONNECT_ATTEMPTS = 10            # Max WebSocket reconnection attempts
RECONNECT_BASE_DELAY = 2               # Initial reconnect delay (seconds)
RECONNECT_MAX_DELAY = 60               # Max reconnect delay (seconds)
HEALTH_CHECK_INTERVAL = 30             # Check connection health every N seconds
STALE_DATA_THRESHOLD = 60              # Data older than N seconds = stale
AUTO_RESTART_ON_CRASH = True           # Auto-restart strategy on crash
MAX_CRASH_RESTARTS = 3                 # Max restarts before giving up

# Excel Performance Tracking
EXCEL_LOG_FILE = "optionalpha_performance.xlsx"  # Excel file for trade history

# =============================================================================


@dataclass
class Config:
    """Strategy configuration - uses top-level constants as defaults"""

    # Index Settings
    INDEX: str = INDEX
    INDEX_EXCHANGE: str = "NSE_INDEX"       # Auto-set based on INDEX
    OPTIONS_EXCHANGE: str = "NFO"           # Auto-set based on INDEX
    STRIKE_INTERVAL: int = 50               # Auto-set based on INDEX

    # Expiry & Capital (from top-level config)
    EXPIRY_WEEK: int = EXPIRY_WEEK
    CAPITAL_PERCENT: float = CAPITAL_PERCENT

    # Risk Management (from top-level config)
    SL_PERCENT: float = SL_PERCENT
    TARGET_POINTS: float = TARGET_POINTS
    BREAKEVEN_POINTS: float = BREAKEVEN_POINTS
    MAX_COMPLETED_TRADES: int = MAX_COMPLETED_TRADES

    # Timing (IST) - usually don't need to change
    MARKET_OPEN: str = "09:15"
    FIRST_CANDLE_CLOSE: str = "09:30"
    MARKET_CLOSE: str = "15:30"
    NO_NEW_ENTRY_TIME: str = NO_NEW_ENTRY_TIME
    FORCE_EXIT_TIME: str = FORCE_EXIT_TIME

    # Strategy Name
    STRATEGY_NAME: str = "OptionsAlpha"

    # 1-min candle tracking (number of candles to keep)
    CANDLE_HISTORY_SIZE: int = 5


# Index-specific configurations (for scalability)
INDEX_CONFIG = {
    "NIFTY": {
        "index_exchange": "NSE_INDEX",
        "options_exchange": "NFO",
        "strike_interval": 50,
        "lot_size": 65  # Fallback, actual fetched from API
    },
    "BANKNIFTY": {
        "index_exchange": "NSE_INDEX",
        "options_exchange": "NFO",
        "strike_interval": 100,
        "lot_size": 30
    },
    "SENSEX": {
        "index_exchange": "BSE_INDEX",
        "options_exchange": "BFO",
        "strike_interval": 100,
        "lot_size": 20
    }
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EntryLevels:
    """Calculated entry levels for the day"""
    atm_strike: int
    pe_symbol: str
    ce_symbol: str
    pe_entry_level: float
    ce_entry_level: float
    expiry_date: str
    lot_size: int


@dataclass
class Position:
    """Active position tracking - tracks only positions created by THIS strategy instance"""
    option_type: Literal["CE", "PE"]
    symbol: str
    entry_price: float
    quantity: int
    sl_price: float                    # Current SL (can change with breakeven)
    target_price: float
    entry_time: datetime
    order_id: str                      # Entry order ID (unique identifier)
    cost_basis: float                  # Total cost for PnL calculation
    breakeven_activated: bool = False  # Track if breakeven SL has been applied
    strategy_tag: str = ""             # Strategy name that created this position
    initial_sl: float = 0.0            # Original SL at entry (doesn't change)


@dataclass
class Candle:
    """1-minute candle structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class TickData:
    """Real-time tick data for a symbol"""
    symbol: str
    ltp: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    # 1-min candle tracking
    current_candle: Optional[Candle] = None
    candle_history: deque = field(default_factory=lambda: deque(maxlen=5))


# =============================================================================
# REAL-TIME TICK MANAGER (WebSocket) - WITH ROBUST RECONNECTION
# =============================================================================

class TickManager:
    """Manages real-time tick data via WebSocket with auto-reconnection"""

    def __init__(self, client: api, config: Config, logger: StrategyLogger):
        self.client = client
        self.config = config
        self.logger = logger
        self.ticks: Dict[str, TickData] = {}
        self.lock = threading.Lock()
        self._current_minute: Dict[str, int] = {}

        # Connection state management
        self.state = ConnectionState.DISCONNECTED
        self.subscribed_symbols: List[dict] = []
        self.reconnect_attempts = 0
        self.last_tick_time: Optional[datetime] = None

        # Background threads
        self._health_check_thread: Optional[threading.Thread] = None
        self._stop_health_check = threading.Event()

        # Callbacks
        self.on_reconnect_callback: Optional[Callable] = None
        self.on_disconnect_callback: Optional[Callable] = None

    def on_tick(self, data: dict):
        """Callback for WebSocket tick updates"""
        try:
            ltp_data = data.get("ltp", {})
            for _exchange, symbols in ltp_data.items():
                for symbol, info in symbols.items():
                    ltp = float(info.get("ltp", 0))
                    if ltp > 0:
                        self._update_tick(symbol, ltp)
                        self.last_tick_time = datetime.now()
        except Exception as e:
            self.logger.error(f"[TickManager] Error processing tick: {e}")

    def _update_tick(self, symbol: str, ltp: float):
        """Update tick data and build 1-min candles"""
        now = datetime.now()
        current_minute = now.minute

        with self.lock:
            if symbol not in self.ticks:
                self.ticks[symbol] = TickData(symbol=symbol)

            tick = self.ticks[symbol]
            tick.ltp = ltp
            tick.timestamp = now

            # Check if we need to start a new candle
            prev_minute = self._current_minute.get(symbol, -1)

            if current_minute != prev_minute:
                # Save previous candle if exists
                if tick.current_candle is not None:
                    tick.candle_history.append(tick.current_candle)

                # Start new candle
                tick.current_candle = Candle(
                    timestamp=now.replace(second=0, microsecond=0),
                    open=ltp,
                    high=ltp,
                    low=ltp,
                    close=ltp
                )
                self._current_minute[symbol] = current_minute
            else:
                # Update current candle
                if tick.current_candle:
                    tick.current_candle.high = max(tick.current_candle.high, ltp)
                    tick.current_candle.low = min(tick.current_candle.low, ltp)
                    tick.current_candle.close = ltp

    def get_ltp(self, symbol: str) -> float:
        """Get current LTP for a symbol"""
        with self.lock:
            tick = self.ticks.get(symbol)
            return tick.ltp if tick else 0.0

    def get_tick_age(self, symbol: str) -> float:
        """Get age of last tick in seconds"""
        with self.lock:
            tick = self.ticks.get(symbol)
            if tick and tick.timestamp:
                return (datetime.now() - tick.timestamp).total_seconds()
            return float('inf')

    def get_last_candles(self, symbol: str, count: int = 2) -> List[Candle]:
        """Get last N completed 1-min candles"""
        with self.lock:
            tick = self.ticks.get(symbol)
            if not tick:
                return []
            return list(tick.candle_history)[-count:]

    def get_current_candle(self, symbol: str) -> Optional[Candle]:
        """Get current (incomplete) 1-min candle"""
        with self.lock:
            tick = self.ticks.get(symbol)
            return tick.current_candle if tick else None

    def is_data_stale(self) -> bool:
        """Check if data is stale (no recent ticks)"""
        if not self.last_tick_time:
            return True
        age = (datetime.now() - self.last_tick_time).total_seconds()
        return age > STALE_DATA_THRESHOLD

    def subscribe(self, symbols: List[dict]) -> bool:
        """Subscribe to symbols via WebSocket with retry logic"""
        self.subscribed_symbols = symbols
        return self._connect_and_subscribe()

    def _connect_and_subscribe(self) -> bool:
        """Internal method to connect and subscribe"""
        self.state = ConnectionState.CONNECTING
        try:
            self.client.connect()
            self.client.subscribe_ltp(
                self.subscribed_symbols,
                on_data_received=self.on_tick
            )
            self.state = ConnectionState.CONNECTED
            self.reconnect_attempts = 0
            self.logger.info(f"[TickManager] Connected and subscribed to {len(self.subscribed_symbols)} symbols")

            # Start health check thread
            self._start_health_check()
            return True

        except Exception as e:
            self.logger.error(f"[TickManager] Connection error: {e}")
            self.state = ConnectionState.DISCONNECTED
            return False

    def reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff (iterative, not recursive)"""
        if self.state == ConnectionState.RECONNECTING:
            return False

        self.state = ConnectionState.RECONNECTING

        # Iterative reconnection to avoid stack overflow
        while self.reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
            self.reconnect_attempts += 1

            # Calculate delay with exponential backoff
            delay = min(
                RECONNECT_BASE_DELAY * (2 ** (self.reconnect_attempts - 1)),
                RECONNECT_MAX_DELAY
            )

            self.logger.warning(
                f"[TickManager] Reconnecting in {delay}s (attempt {self.reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS})"
            )
            time.sleep(delay)

            # Try to disconnect cleanly first
            try:
                self.client.disconnect()
            except Exception:
                pass

            # Attempt reconnection
            if self._connect_and_subscribe():
                self.logger.info("[TickManager] Reconnection successful!")
                if self.on_reconnect_callback:
                    self.on_reconnect_callback()
                return True
            else:
                self.logger.error(f"[TickManager] Reconnection attempt {self.reconnect_attempts} failed")

        # All attempts exhausted
        self.logger.critical(f"[TickManager] Max reconnection attempts ({MAX_RECONNECT_ATTEMPTS}) exceeded")
        self.state = ConnectionState.FAILED
        return False

    def _start_health_check(self):
        """Start background health check thread (only if not already running)"""
        # Prevent multiple health check threads
        if self._health_check_thread and self._health_check_thread.is_alive():
            return

        self._stop_health_check.clear()
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_check_thread.start()

    def _health_check_loop(self):
        """Background loop to check connection health"""
        while not self._stop_health_check.is_set():
            time.sleep(HEALTH_CHECK_INTERVAL)

            if self._stop_health_check.is_set():
                break

            # Check if data is stale
            if self.state == ConnectionState.CONNECTED and self.is_data_stale():
                self.logger.warning("[TickManager] Stale data detected, triggering reconnection")
                if self.on_disconnect_callback:
                    self.on_disconnect_callback()
                self.reconnect()

    def unsubscribe(self, symbols: List[dict]):
        """Unsubscribe from symbols"""
        try:
            self.client.unsubscribe_ltp(symbols)
        except Exception as e:
            self.logger.error(f"[TickManager] Unsubscribe error: {e}")

    def disconnect(self):
        """Disconnect WebSocket and cleanup"""
        self._stop_health_check.set()

        try:
            self.client.disconnect()
            self.state = ConnectionState.DISCONNECTED
            self.logger.info("[TickManager] Disconnected")
        except Exception as e:
            self.logger.error(f"[TickManager] Disconnect error: {e}")

    @property
    def connected(self) -> bool:
        """Check if currently connected"""
        return self.state == ConnectionState.CONNECTED


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_trading_day(client: api) -> tuple[bool, str]:
    """
    Check if today is a trading day.
    Returns: (is_trading: bool, message: str)

    Note: Checks market timings API first to handle special trading days
    (e.g., Union Budget day on weekends). Weekend check is only used as fallback.
    """
    today = datetime.now()
    is_weekend = today.weekday() >= 5  # Saturday = 5, Sunday = 6

    # Check market timings API first (handles special trading days like Budget day)
    try:
        result = client.timings(date=today.strftime("%Y-%m-%d"))
        if result.get("status") == "success":
            data = result.get("data", [])
            if not data:
                # No exchange data = holiday or market closed
                if is_weekend:
                    return False, "Weekend - Market closed"
                return False, "Holiday - Market closed"

            # Check if NSE/NFO is open
            for exchange in data:
                if exchange.get("exchange") in ["NSE", "NFO"]:
                    if is_weekend:
                        return True, "Special trading day (market open on weekend)"
                    return True, "Trading day"

            return False, "NSE/NFO not open today"
    except Exception as e:
        # If API fails, fail-safe: don't trade on uncertain days
        print(f"Warning: Could not check market timings: {e}")
        if is_weekend:
            return False, "Weekend - Market closed (API check failed)"
        # FAIL-SAFE: Don't trade if we can't confirm it's a trading day
        return False, "Cannot confirm trading day (API check failed) - not trading to be safe"

    return True, "Trading day"


def round_to_tick(price: float, tick_size: float = 0.05) -> float:
    """
    Round price to nearest tick size (default 0.05 for options).

    Examples:
        round_to_tick(245.23) → 245.25
        round_to_tick(245.21) → 245.20
        round_to_tick(245.00) → 245.00
    """
    return round(round(price / tick_size) * tick_size, 2)


def get_expiry_date(client: api, index: str, exchange: str, expiry_week: int) -> str:
    """
    Get expiry date based on expiry_week parameter.
    expiry_week: 1 = current/nearest, 2 = next week, etc.
    Returns date in DDMMMYY format (e.g., 28NOV25)
    """
    result = client.expiry(
        symbol=index,
        exchange=exchange,
        instrumenttype="options"
    )

    if result.get("status") != "success":
        raise Exception(f"Failed to fetch expiry dates: {result.get('message')}")

    expiry_dates = result.get("data", [])
    if not expiry_dates:
        raise Exception("No expiry dates available")

    # expiry_dates are sorted, pick based on expiry_week
    idx = min(expiry_week - 1, len(expiry_dates) - 1)
    return expiry_dates[idx]


def build_option_symbol(index: str, expiry_date: str, strike: int, option_type: str) -> str:
    """
    Build option symbol in OpenAlgo format.
    Example: NIFTY28NOV2526000CE

    Note: This is only used for resistance level calculation where we need to
    build multiple symbols (20+ API calls would be slow). For main entry level
    calculation, use optionsymbol() API instead.
    """
    return f"{index}{expiry_date}{strike}{option_type}"


def get_5min_candle_close(client: api, symbol: str, exchange: str) -> float:
    """
    Fetch the first 5-minute candle close (9:15-9:20).
    Returns: close price
    """
    today = datetime.now().strftime("%Y-%m-%d")

    df = client.history(
        symbol=symbol,
        exchange=exchange,
        interval="5m",
        start_date=today,
        end_date=today
    )

    if isinstance(df, dict) and df.get("status") == "error":
        raise Exception(f"Failed to fetch 5min candle: {df.get('message')}")

    if df is None or df.empty:
        raise Exception(f"No 5min candle data for {symbol}")

    # Get the first candle (9:15-9:20)
    first_candle = df.iloc[0]
    return float(first_candle["close"])


# REMOVED: Resistance levels calculation not needed in simplified version
# Original function calculated R1-R6 levels using straddle/strangle pricing
# Simplified version uses fixed 25-point target with breakeven SL instead


def get_15min_candle(client: api, symbol: str, exchange: str) -> dict:
    """
    Fetch the first 15-minute candle (9:15-9:30).
    Returns: {open, high, low, close}
    """
    today = datetime.now().strftime("%Y-%m-%d")

    df = client.history(
        symbol=symbol,
        exchange=exchange,
        interval="15m",
        start_date=today,
        end_date=today
    )

    if isinstance(df, dict) and df.get("status") == "error":
        raise Exception(f"Failed to fetch 15min candle: {df.get('message')}")

    if df.empty:
        raise Exception(f"No 15min candle data for {symbol}")

    # Get the first candle (9:15-9:30)
    first_candle = df.iloc[0]
    return {
        "open": float(first_candle["open"]),
        "high": float(first_candle["high"]),
        "low": float(first_candle["low"]),
        "close": float(first_candle["close"])
    }


def calculate_quantity(capital: float, price: float, lot_size: int) -> int:
    """
    Calculate quantity based on capital and lot size.

    Returns 0 if insufficient capital or invalid inputs.
    """
    if price <= 0 or lot_size <= 0 or capital <= 0:
        return 0

    max_lots = int(capital / (price * lot_size))

    if max_lots < 1:
        return 0  # Insufficient capital for even 1 lot

    return max_lots * lot_size


def check_entry_condition_ws(
    tick_manager: TickManager,
    symbol: str,
    entry_level: float
) -> bool:
    """
    Check entry condition using WebSocket data:
    1. LTP > entry_level
    2. Current 1min candle: open AND close > entry_level
    3. Previous 1min candle: open OR close < entry_level (crossover)
    """
    # Get LTP and validate it's not zero/stale
    ltp = tick_manager.get_ltp(symbol)
    if ltp <= 0 or ltp <= entry_level:
        return False

    # Get current candle
    curr_candle = tick_manager.get_current_candle(symbol)
    if not curr_candle:
        return False

    # Current candle: both open and close above entry level
    if not (curr_candle.open > entry_level and curr_candle.close > entry_level):
        return False

    # Get previous candle
    prev_candles = tick_manager.get_last_candles(symbol, count=1)
    if not prev_candles:
        return False

    prev_candle = prev_candles[-1]

    # Previous candle: open OR close below entry level (crossover confirmation)
    if not (prev_candle.open < entry_level or prev_candle.close < entry_level):
        return False

    return True


# =============================================================================
# MAIN STRATEGY CLASS
# =============================================================================

class OptionsAlphaStrategy:
    """Main strategy class with WebSocket support and robust error handling"""

    def __init__(self, config: Config):
        self.config = config

        # Setup logger first
        self.logger = StrategyLogger(
            name=config.STRATEGY_NAME,
            log_to_file=LOG_TO_FILE,
            log_dir=LOG_DIR
        )

        self.client = api(
            api_key=API_KEY,
            host=HOST,
            ws_url=WS_URL
        )

        # Load index-specific config
        idx_config = INDEX_CONFIG.get(config.INDEX, INDEX_CONFIG["NIFTY"])
        self.index_exchange = idx_config["index_exchange"]
        self.options_exchange = idx_config["options_exchange"]
        self.strike_interval = idx_config["strike_interval"]

        # Tick manager for WebSocket (with logger)
        self.tick_manager = TickManager(self.client, config, self.logger)

        # Set reconnection callbacks
        self.tick_manager.on_reconnect_callback = self._on_ws_reconnect
        self.tick_manager.on_disconnect_callback = self._on_ws_disconnect

        # State
        self.entry_levels: Optional[EntryLevels] = None
        self.position: Optional[Position] = None
        self.completed_trades: int = 0      # Tracks COMPLETED (exited) trades
        self.can_trade: bool = True          # Flag to control new entries
        self.daily_pnl: float = 0.0
        self.trades_dict: dict = {}          # Key: order_id, Value: complete trade info
        self.subscribed_symbols: List[dict] = []
        self.allocated_capital: float = 0.0  # Capital allocated for trading (capped at 2L)

        # Crash recovery state
        self.crash_count: int = 0
        self.running: bool = False

    def log(self, message: str):
        """Log message with timestamp"""
        self.logger.info(message)

    def log_trade_to_excel(self, trade_data: dict):
        """
        Log trade details to Excel file for long-term performance tracking.
        Creates file if it doesn't exist, appends new row otherwise.

        Args:
            trade_data: Dictionary with complete trade information
        """
        try:
            excel_file = Path(EXCEL_LOG_FILE)

            # Prepare row data with 9:30 calculated levels + trade execution details
            row_data = {
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "ATM Strike": self.entry_levels.atm_strike if self.entry_levels else "",
                "PE Symbol": self.entry_levels.pe_symbol if self.entry_levels else "",
                "PE Entry Level": self.entry_levels.pe_entry_level if self.entry_levels else "",
                "CE Symbol": self.entry_levels.ce_symbol if self.entry_levels else "",
                "CE Entry Level": self.entry_levels.ce_entry_level if self.entry_levels else "",
                "Expiry": self.entry_levels.expiry_date if self.entry_levels else "",
                "Entry Order ID": trade_data.get("entry_order_id", ""),
                "Exit Order ID": trade_data.get("exit_order_id", ""),
                "Trade Type": trade_data.get("option_type", ""),
                "Traded Symbol": trade_data.get("symbol", ""),
                "Entry Time": trade_data.get("entry_time", ""),
                "Entry Price": trade_data.get("entry_price", 0.0),
                "Initial SL": trade_data.get("initial_sl", 0.0),
                "Target Price": trade_data.get("target_price", 0.0),
                "Exit Time": trade_data.get("exit_time", ""),
                "Exit Price": trade_data.get("exit_price", 0.0),
                "Final SL": trade_data.get("final_sl", 0.0),
                "Quantity": trade_data.get("quantity", 0),
                "PnL": trade_data.get("pnl", 0.0),
                "Exit Reason": trade_data.get("exit_reason", ""),
                "Breakeven Activated": "Yes" if trade_data.get("breakeven_activated", False) else "No"
            }

            # Convert to DataFrame
            new_row_df = pd.DataFrame([row_data])

            # Append to existing file or create new one
            if excel_file.exists():
                # Read existing data
                existing_df = pd.read_excel(excel_file, engine='openpyxl')
                # Append new row
                updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            else:
                # Create new file
                updated_df = new_row_df

            # Write to Excel
            updated_df.to_excel(excel_file, index=False, engine='openpyxl')
            self.logger.debug(f"Trade logged to Excel: {excel_file}")

        except Exception as e:
            self.logger.error(f"Failed to log trade to Excel: {e}")
            # Don't raise - Excel logging failure shouldn't stop the strategy

    def _on_ws_reconnect(self):
        """Callback when WebSocket reconnects"""
        self.log("WebSocket reconnected - resuming strategy")

    def _on_ws_disconnect(self):
        """Callback when WebSocket disconnects"""
        self.log("WebSocket disconnected - will attempt reconnection")

    def is_market_hours(self) -> tuple[bool, str]:
        """
        Check if we're within tradeable market hours.
        Returns: (can_trade: bool, message: str)
        """
        now = datetime.now()
        today_str = now.strftime('%Y-%m-%d')

        market_open = datetime.strptime(f"{today_str} {self.config.MARKET_OPEN}", "%Y-%m-%d %H:%M")
        market_close = datetime.strptime(f"{today_str} {self.config.MARKET_CLOSE}", "%Y-%m-%d %H:%M")

        if now > market_close:
            return False, f"Market is closed for today (closes at {self.config.MARKET_CLOSE})"
        elif now < market_open:
            return True, f"Market not yet open (opens at {self.config.MARKET_OPEN})"
        else:
            return True, "Market is open"

    def wait_for_market_open(self):
        """Wait until market opens (called only if market not yet closed)"""
        self.log(f"Waiting for market open at {self.config.MARKET_OPEN}...")

        while True:
            now = datetime.now()
            market_open = datetime.strptime(
                f"{now.strftime('%Y-%m-%d')} {self.config.MARKET_OPEN}",
                "%Y-%m-%d %H:%M"
            )

            if now >= market_open:
                break

            time.sleep(10)

        self.log("Market is open!")

    # REMOVED: Not needed in simplified version - no resistance levels
    # def wait_for_5min_candle_and_calculate_levels(self):
    #     """Wait for first 5-min candle (9:20) and calculate R1-R6 levels"""
    #     pass

    def wait_for_first_candle(self):
        """Wait for first 15-min candle to complete (9:30)"""
        self.log(f"Waiting for 9:30...")

        while True:
            now = datetime.now()
            candle_close = datetime.strptime(
                f"{now.strftime('%Y-%m-%d')} {self.config.FIRST_CANDLE_CLOSE}",
                "%Y-%m-%d %H:%M"
            )

            if now >= candle_close:
                break

            time.sleep(5)

        time.sleep(10)
        self.log("9:30 candle closed")

    def calculate_entry_levels(self) -> EntryLevels:
        """Calculate ATM strike and entry levels using OpenAlgo optionsymbol API"""
        # Get expiry date
        expiry_date = get_expiry_date(
            self.client,
            self.config.INDEX,
            self.options_exchange,
            self.config.EXPIRY_WEEK
        )

        # Use optionsymbol API to get ATM PE (auto-calculates ATM strike)
        pe_result = self.client.optionsymbol(
            underlying=self.config.INDEX,
            exchange=self.index_exchange,
            expiry_date=expiry_date,
            offset="ATM",
            option_type="PE"
        )

        if pe_result.get("status") != "success":
            raise Exception(f"Failed to get PE symbol: {pe_result.get('message')}")

        pe_data = pe_result.get("data", {})
        pe_symbol = pe_data.get("symbol")
        lot_size = pe_data.get("lotsize")
        underlying_ltp = pe_data.get("underlying_ltp")

        # Use optionsymbol API to get ATM CE
        ce_result = self.client.optionsymbol(
            underlying=self.config.INDEX,
            exchange=self.index_exchange,
            expiry_date=expiry_date,
            offset="ATM",
            option_type="CE"
        )

        if ce_result.get("status") != "success":
            raise Exception(f"Failed to get CE symbol: {ce_result.get('message')}")

        ce_data = ce_result.get("data", {})
        ce_symbol = ce_data.get("symbol")

        # Extract ATM strike from symbol with error handling
        try:
            # Format: NIFTY28NOV2522500PE -> remove last 2 chars (PE), split by expiry, get strike
            symbol_parts = pe_symbol[:-2].split(expiry_date)
            if len(symbol_parts) < 2:
                raise ValueError(f"Invalid PE symbol format: {pe_symbol}")
            atm_strike = int(symbol_parts[1])
        except (ValueError, IndexError) as e:
            raise Exception(f"Failed to extract ATM strike from symbol {pe_symbol}: {e}")

        # Get first 15-min candle for PE and CE
        pe_candle = get_15min_candle(self.client, pe_symbol, self.options_exchange)
        ce_candle = get_15min_candle(self.client, ce_symbol, self.options_exchange)

        # Calculate entry levels
        pe_entry_level = (pe_candle["low"] + ce_candle["high"]) / 2
        ce_entry_level = (ce_candle["low"] + ce_candle["high"]) / 2

        self.log(f"Setup: {expiry_date} | ATM: {atm_strike} | Spot: {underlying_ltp:.2f}")
        self.log(f"Entry Levels → PE: {pe_entry_level:.2f} | CE: {ce_entry_level:.2f}")

        return EntryLevels(
            atm_strike=atm_strike,
            pe_symbol=pe_symbol,
            ce_symbol=ce_symbol,
            pe_entry_level=pe_entry_level,
            ce_entry_level=ce_entry_level,
            expiry_date=expiry_date,
            lot_size=lot_size
        )

    def setup_websocket(self):
        """Setup WebSocket subscription for PE and CE symbols"""
        self.subscribed_symbols = [
            {"exchange": self.options_exchange, "symbol": self.entry_levels.pe_symbol},
            {"exchange": self.options_exchange, "symbol": self.entry_levels.ce_symbol}
        ]

        self.tick_manager.subscribe(self.subscribed_symbols)
        time.sleep(2)  # Wait for initial ticks

        # Verify data
        pe_ltp = self.tick_manager.get_ltp(self.entry_levels.pe_symbol)
        ce_ltp = self.tick_manager.get_ltp(self.entry_levels.ce_symbol)
        self.log(f"WebSocket ready | PE: {pe_ltp:.2f} | CE: {ce_ltp:.2f}")

    def cleanup_websocket(self):
        """Cleanup WebSocket connection"""
        if self.subscribed_symbols:
            self.tick_manager.unsubscribe(self.subscribed_symbols)
        self.tick_manager.disconnect()

    def initialize_capital(self):
        """
        Initialize capital with 2L cap - called once at start.

        If available capital > 2.5L, cap at 2L.
        Otherwise use configured percentage of available capital.
        """
        result = self.client.funds()
        if result.get("status") == "success":
            available = float(result.get("data", {}).get("availablecash", 0))
            capital_with_percent = available * self.config.CAPITAL_PERCENT

            # If capital > 2.5L, cap at 2L
            if capital_with_percent > 250000:
                self.allocated_capital = 200000
                self.log(f"Capital capped: Available ₹{available:,.0f} → Using ₹2,00,000")
            else:
                self.allocated_capital = capital_with_percent
                self.log(f"Capital allocated: ₹{self.allocated_capital:,.0f} ({self.config.CAPITAL_PERCENT*100:.0f}% of ₹{available:,.0f})")
        else:
            self.allocated_capital = 0.0
            self.log("Failed to fetch capital - cannot trade")

    def get_available_capital(self) -> float:
        """Get allocated capital (already capped and set at start)"""
        return self.allocated_capital

    def get_ltp_http(self, symbol: str, exchange: str) -> float:
        """
        Get LTP via HTTP API (fallback when WebSocket unavailable).

        Args:
            symbol: Trading symbol
            exchange: Exchange (NFO, NSE, etc.)

        Returns:
            LTP price or 0.0 if failed
        """
        try:
            result = self.client.quotes(symbol=symbol, exchange=exchange)
            if result.get("status") == "success":
                ltp = float(result.get("data", {}).get("ltp", 0))
                return ltp
        except Exception as e:
            self.logger.error(f"HTTP LTP fetch failed for {symbol}: {e}")
        return 0.0

    def get_ltp_with_fallback(self, symbol: str, exchange: str = None) -> float:
        """
        Get LTP with WebSocket primary and HTTP fallback.

        Priority:
        1. WebSocket (if connected and data fresh)
        2. HTTP API (fallback)

        Args:
            symbol: Trading symbol
            exchange: Exchange (defaults to options_exchange)

        Returns:
            LTP price or 0.0 if all methods fail
        """
        exchange = exchange or self.options_exchange

        # Try WebSocket first (if connected and data is fresh)
        if self.tick_manager.connected:
            ws_ltp = self.tick_manager.get_ltp(symbol)
            tick_age = self.tick_manager.get_tick_age(symbol)

            # Use WebSocket data if fresh (< 5 seconds old)
            if ws_ltp > 0 and tick_age < 5:
                return ws_ltp

            # WebSocket data is stale, log warning
            if ws_ltp > 0:
                self.logger.warning(f"WebSocket data stale ({tick_age:.1f}s), using HTTP fallback")

        # Fallback to HTTP
        http_ltp = self.get_ltp_http(symbol, exchange)
        if http_ltp > 0:
            self.logger.debug(f"HTTP LTP for {symbol}: {http_ltp}")
            return http_ltp

        # Last resort: return stale WebSocket data if available
        if self.tick_manager.connected:
            ws_ltp = self.tick_manager.get_ltp(symbol)
            if ws_ltp > 0:
                self.logger.warning(f"Using stale WebSocket data for {symbol}: {ws_ltp}")
                return ws_ltp

        self.logger.error(f"Failed to get LTP for {symbol} (both WS and HTTP failed)")
        return 0.0

    def get_fill_price_from_tradebook(self, order_id: str) -> tuple[float, int]:
        """
        Get actual fill price and quantity from tradebook.

        Args:
            order_id: Order ID to lookup

        Returns:
            tuple: (avg_price, filled_qty)
        """
        try:
            result = self.client.tradebook()
            if result.get("status") == "success":
                trades = result.get("data", [])

                # Find all trades matching this order_id
                matching_trades = [t for t in trades if t.get("orderid") == order_id]

                if matching_trades:
                    # Calculate weighted average price
                    total_value = 0
                    total_qty = 0

                    for trade in matching_trades:
                        qty = int(trade.get("quantity", 0))
                        price = float(trade.get("price", 0))
                        total_value += price * qty
                        total_qty += qty

                    if total_qty > 0:
                        avg_price = total_value / total_qty
                        return round_to_tick(avg_price), total_qty

        except Exception as e:
            self.logger.error(f"Error fetching tradebook: {e}")

        return 0.0, 0

    def verify_order_executed(
        self,
        order_id: str,
        expected_action: str,
        max_wait_seconds: int = 10,
        check_interval: float = 1.0
    ) -> dict:
        """
        Verify order execution status and fetch actual fill price from tradebook.

        Args:
            order_id: Order ID to check
            expected_action: Expected action (BUY/SELL)
            max_wait_seconds: Maximum time to wait for execution
            check_interval: Time between status checks (default 1 second)

        Returns:
            dict with keys:
                - executed: bool
                - status: str (order status)
                - filled_qty: int
                - avg_price: float (from tradebook)
                - message: str
        """
        # Initial wait for order to hit exchange
        time.sleep(1)

        start_time = time.time()
        last_status = "UNKNOWN"
        verification_count = 0

        self.log(f"Verifying order {order_id}...")

        while (time.time() - start_time) < max_wait_seconds:
            verification_count += 1
            try:
                result = self.client.orderstatus(order_id=order_id)

                if result.get("status") == "success":
                    order_data = result.get("data", {})
                    order_status = order_data.get("order_status", "").upper()
                    last_status = order_status

                    self.logger.debug(f"Verification #{verification_count}: {order_status}")

                    # Check for completed/executed status
                    if order_status in ["COMPLETE", "COMPLETED", "FILLED", "EXECUTED"]:
                        # Get actual fill price from tradebook
                        avg_price, filled_qty = self.get_fill_price_from_tradebook(order_id)

                        # Fallback to orderstatus if tradebook lookup fails
                        if avg_price <= 0:
                            filled_qty = int(order_data.get("filled_quantity", order_data.get("quantity", 0)))
                            avg_price = float(order_data.get("average_price", order_data.get("price", 0)))
                            avg_price = round_to_tick(avg_price)

                        self.log(f"Order EXECUTED (attempt {verification_count}): Qty={filled_qty}, Avg Price={avg_price:.2f}")
                        return {
                            "executed": True,
                            "status": order_status,
                            "filled_qty": filled_qty,
                            "avg_price": avg_price,
                            "message": "Order executed successfully"
                        }

                    # Check for rejected/cancelled
                    elif order_status in ["REJECTED", "CANCELLED", "CANCELED", "FAILED"]:
                        reject_reason = order_data.get("reject_reason", order_data.get("message", "Unknown"))
                        self.logger.error(f"Order {order_status} (attempt {verification_count}): {reject_reason}")
                        return {
                            "executed": False,
                            "status": order_status,
                            "filled_qty": 0,
                            "avg_price": 0.0,
                            "message": f"Order {order_status}: {reject_reason}"
                        }

                    # Still pending - continue waiting
                    self.logger.debug(f"Order status: {order_status}, waiting...")

                else:
                    self.logger.warning(f"orderstatus API error (attempt {verification_count}): {result.get('message')}")

            except Exception as e:
                self.logger.error(f"Error checking order status (attempt {verification_count}): {e}")

            time.sleep(check_interval)

        # Timeout
        self.logger.warning(f"Order verification timeout after {verification_count} attempts. Last status: {last_status}")
        return {
            "executed": False,
            "status": last_status,
            "filled_qty": 0,
            "avg_price": 0.0,
            "message": f"Timeout waiting for order execution after {verification_count} attempts. Last status: {last_status}"
        }

    def place_order(self, option_type: Literal["CE", "PE"], entry_price: float) -> Optional[Position]:
        """Place buy order and create position"""
        symbol = self.entry_levels.ce_symbol if option_type == "CE" else self.entry_levels.pe_symbol
        entry_level = self.entry_levels.ce_entry_level if option_type == "CE" else self.entry_levels.pe_entry_level

        # Calculate quantity
        capital = self.get_available_capital()
        quantity = calculate_quantity(capital, entry_price, self.entry_levels.lot_size)

        if quantity <= 0:
            self.log("Insufficient capital for order")
            return None

        # Order placement (silent for speed)

        # Place order
        result = self.client.placeorder(
            strategy=self.config.STRATEGY_NAME,
            symbol=symbol,
            exchange=self.options_exchange,
            action="BUY",
            quantity=quantity,
            price_type="MARKET",
            product="MIS"
        )

        if result.get("status") != "success":
            self.log(f"Order placement failed: {result.get('message')}")
            return None

        order_id = result.get("orderid", "")
        self.log(f"Order placed! Order ID: {order_id}")

        # Verify order execution
        verification = self.verify_order_executed(order_id, "BUY")

        if not verification["executed"]:
            self.logger.error(f"BUY order not executed: {verification['message']}")
            return None

        # Use verified fill price (fallback to WebSocket LTP if not available)
        actual_price = verification["avg_price"]
        if actual_price <= 0:
            actual_price = self.tick_manager.get_ltp(symbol)
        if actual_price <= 0:
            actual_price = entry_price

        actual_price = round_to_tick(actual_price)
        filled_qty = verification["filled_qty"] if verification["filled_qty"] > 0 else quantity

        # Calculate SL and Target (rounded to tick size)
        sl_price = round_to_tick(entry_level * (1 - self.config.SL_PERCENT))
        target_price = round_to_tick(actual_price + self.config.TARGET_POINTS)

        position = Position(
            option_type=option_type,
            symbol=symbol,
            entry_price=actual_price,
            quantity=filled_qty,
            sl_price=sl_price,
            target_price=target_price,
            entry_time=datetime.now(),
            order_id=order_id,
            cost_basis=actual_price * filled_qty,
            breakeven_activated=False,
            strategy_tag=self.config.STRATEGY_NAME,
            initial_sl=sl_price  # Store initial SL for trade record
        )

        self.log(f"✓ POSITION: {option_type} {symbol} | Entry: {actual_price:.2f} | Qty: {filled_qty} | SL: {sl_price:.2f} | Tgt: {target_price:.2f}")
        return position

    # REMOVED: verify_position_exists() - too slow for scalping
    # Trust our internal tracking for speed

    def manage_position(self) -> bool:
        """
        Manage active position with breakeven trailing SL using WebSocket data.

        Trailing Logic:
        - When profit reaches 20 points → move SL to cost (breakeven)

        Returns True if position is closed, False otherwise.
        """
        if not self.position:
            return False

        # Get LTP with WebSocket + HTTP fallback (critical for SL/target)
        ltp = self.get_ltp_with_fallback(self.position.symbol)
        if ltp <= 0:
            self.logger.warning("Could not get LTP for position management")
            return False

        current_value = ltp * self.position.quantity
        pnl = current_value - self.position.cost_basis
        pnl_percent = (pnl / self.position.cost_basis) * 100
        pnl_points = ltp - self.position.entry_price

        # Check target hit
        if ltp >= self.position.target_price:
            self.log(f"✓ TARGET HIT @ {ltp:.2f} (+{pnl_points:.1f}pts)")
            return self.close_position("TARGET")

        # Check SL hit
        if ltp <= self.position.sl_price:
            self.log(f"✗ SL HIT @ {ltp:.2f} ({pnl_points:.1f}pts)")
            return self.close_position("STOPLOSS")

        # Breakeven Trailing SL logic
        if not self.position.breakeven_activated and pnl_points >= self.config.BREAKEVEN_POINTS:
            new_sl = round_to_tick(self.position.entry_price)

            if new_sl > self.position.sl_price:
                self.position.sl_price = new_sl
                self.position.breakeven_activated = True
                self.log(f"⚡ BREAKEVEN @ +{pnl_points:.1f}pts | SL→{new_sl:.2f}")

        return False

    def close_position(self, reason: str, max_retries: int = 3) -> bool:
        """
        Robust position close method with retries.

        IMPORTANT: This closes the position by placing an opposite order (SELL if we're LONG).
        The broker will close the position using FIFO logic. Since this strategy only
        creates ONE position at a time and tracks it by order_id, this is safe.
        The verify_position_exists() check ensures we don't try to close manually-closed positions.

        Args:
            reason: Exit reason (STOPLOSS, TARGET, TIME_EXIT, MANUAL_EXIT, etc.)
            max_retries: Number of retry attempts for failed orders

        Returns:
            True if position closed successfully, False otherwise
        """
        if not self.position:
            return False

        self.log(f"Closing {self.position.option_type}: {reason}")

        # Store position details before closing
        symbol = self.position.symbol
        quantity = self.position.quantity
        entry_price = self.position.entry_price
        option_type = self.position.option_type
        entry_time = self.position.entry_time
        entry_order_id = self.position.order_id  # Store entry order ID (don't overwrite!)
        initial_sl = self.position.initial_sl
        target_price = self.position.target_price
        breakeven_activated = self.position.breakeven_activated
        final_sl = self.position.sl_price

        # Attempt to close with retries
        order_success = False
        exit_price = 0.0
        exit_order_id = ""

        for attempt in range(1, max_retries + 1):
            self.log(f"Exit attempt {attempt}/{max_retries}...")

            try:
                result = self.client.placeorder(
                    strategy=self.config.STRATEGY_NAME,
                    symbol=symbol,
                    exchange=self.options_exchange,
                    action="SELL",
                    quantity=quantity,
                    price_type="MARKET",
                    product="MIS"
                )

                if result.get("status") == "success":
                    exit_order_id = result.get("orderid", "")
                    self.log(f"Exit order placed! Order ID: {exit_order_id}")

                    # Verify order execution
                    verification = self.verify_order_executed(exit_order_id, "SELL")

                    if verification["executed"]:
                        order_success = True
                        exit_price = verification["avg_price"]

                        # Fallback if avg_price not available
                        if exit_price <= 0:
                            exit_price = self.tick_manager.get_ltp(symbol)
                        if exit_price <= 0:
                            if reason == "STOPLOSS":
                                exit_price = self.position.sl_price
                            elif reason == "TARGET":
                                exit_price = self.position.target_price
                            else:
                                exit_price = entry_price

                        exit_price = round_to_tick(exit_price)
                        self.log(f"Exit CONFIRMED at {exit_price:.2f}")
                        break
                    else:
                        self.logger.warning(f"Exit order not confirmed: {verification['message']}")
                        if attempt < max_retries:
                            time.sleep(2)

                else:
                    self.log(f"Exit order placement failed: {result.get('message')}")
                    if attempt < max_retries:
                        time.sleep(2)  # Wait before retry

            except Exception as e:
                self.logger.error(f"Exit order exception: {e}")
                if attempt < max_retries:
                    time.sleep(2)

        if not order_success:
            self.logger.critical(f"FAILED TO CLOSE POSITION after {max_retries} attempts!")
            self.logger.critical(f"MANUAL INTERVENTION REQUIRED: {symbol} x {quantity}")
            return False

        # Calculate PnL
        pnl = (exit_price - entry_price) * quantity
        self.daily_pnl += pnl

        # Record trade in dictionary (key: entry_order_id)
        trade_record = {
            "entry_order_id": entry_order_id,
            "exit_order_id": exit_order_id,
            "symbol": symbol,
            "option_type": option_type,
            "entry_time": entry_time.strftime("%H:%M:%S"),
            "entry_price": entry_price,
            "initial_sl": initial_sl,
            "target_price": target_price,
            "exit_time": datetime.now().strftime("%H:%M:%S"),
            "exit_price": exit_price,
            "final_sl": final_sl,
            "quantity": quantity,
            "pnl": pnl,
            "exit_reason": reason,
            "breakeven_activated": breakeven_activated
        }
        self.trades_dict[entry_order_id] = trade_record

        # Log trade to Excel for long-term tracking
        self.log_trade_to_excel(trade_record)

        # Clear position
        self.position = None

        # Increment completed trades
        self.completed_trades += 1

        # Log summary
        pnl_str = f"+₹{pnl:.2f}" if pnl >= 0 else f"₹{pnl:.2f}"
        self.log(f"✓ EXIT @ {exit_price:.2f} | PnL: {pnl_str} | Trades: {self.completed_trades}/{self.config.MAX_COMPLETED_TRADES}")

        # Check if max trades reached - stop new entries
        if self.completed_trades >= self.config.MAX_COMPLETED_TRADES:
            self.can_trade = False
            self.log(f"MAX COMPLETED TRADES ({self.config.MAX_COMPLETED_TRADES}) REACHED - No more entries today")

        return True

    def is_force_exit_time(self) -> bool:
        """Check if it's time to force exit all positions (15:00)"""
        now = datetime.now()
        exit_time = datetime.strptime(
            f"{now.strftime('%Y-%m-%d')} {self.config.FORCE_EXIT_TIME}",
            "%Y-%m-%d %H:%M"
        )
        return now >= exit_time

    def is_new_entry_allowed(self) -> bool:
        """
        Check if new entries are allowed.
        Returns False if:
        - Time is past NO_NEW_ENTRY_TIME (14:15)
        - can_trade flag is False (max completed trades reached)
        """
        if not self.can_trade:
            return False

        now = datetime.now()
        cutoff_time = datetime.strptime(
            f"{now.strftime('%Y-%m-%d')} {self.config.NO_NEW_ENTRY_TIME}",
            "%Y-%m-%d %H:%M"
        )
        return now < cutoff_time

    def print_summary(self):
        """Print comprehensive end-of-day summary"""
        self.log("=" * 70)
        self.log("DAY SUMMARY")
        self.log("=" * 70)
        self.log(f"Index: {self.config.INDEX} | ATM Strike: {self.entry_levels.atm_strike if self.entry_levels else 'N/A'}")
        self.log(f"Completed Trades: {self.completed_trades}/{self.config.MAX_COMPLETED_TRADES}")

        # Calculate win/loss stats
        if self.trades_dict:
            wins = sum(1 for t in self.trades_dict.values() if t['pnl'] > 0)
            losses = sum(1 for t in self.trades_dict.values() if t['pnl'] < 0)
            win_rate = (wins / len(self.trades_dict) * 100) if self.trades_dict else 0
            self.log(f"Win Rate: {wins}W / {losses}L ({win_rate:.1f}%)")

        pnl_color = "+" if self.daily_pnl >= 0 else ""
        self.log(f"Daily PnL: {pnl_color}₹{self.daily_pnl:.2f}")
        self.log("=" * 70)

        if not self.trades_dict:
            self.log("No trades executed today")
        else:
            for i, (entry_order_id, trade) in enumerate(self.trades_dict.items(), 1):
                pnl_sign = "+" if trade['pnl'] >= 0 else ""
                be_flag = " [BE]" if trade['breakeven_activated'] else ""

                self.log(f"\nTrade #{i} ({trade['exit_reason']})")
                self.log(f"  Entry Order : {entry_order_id}")
                self.log(f"  Exit Order  : {trade.get('exit_order_id', 'N/A')}")
                self.log(f"  Symbol      : {trade['option_type']} {trade['symbol']}")
                self.log(f"  Entry Time  : {trade['entry_time']} @ ₹{trade['entry_price']:.2f}")
                self.log(f"  Initial SL  : ₹{trade['initial_sl']:.2f}")
                self.log(f"  Target      : ₹{trade['target_price']:.2f}")
                self.log(f"  Exit Time   : {trade['exit_time']} @ ₹{trade['exit_price']:.2f}")
                self.log(f"  Final SL    : ₹{trade['final_sl']:.2f}{be_flag}")
                self.log(f"  Quantity    : {trade['quantity']}")
                self.log(f"  PnL         : {pnl_sign}₹{trade['pnl']:.2f}")

        self.log("=" * 70)

    def print_status(self):
        """Print current status (called periodically)"""
        if not self.entry_levels:
            return

        pe_ltp = self.tick_manager.get_ltp(self.entry_levels.pe_symbol)
        ce_ltp = self.tick_manager.get_ltp(self.entry_levels.ce_symbol)

        status = f"PE: {pe_ltp:.2f} (Entry: {self.entry_levels.pe_entry_level:.2f}) | "
        status += f"CE: {ce_ltp:.2f} (Entry: {self.entry_levels.ce_entry_level:.2f})"

        if self.position:
            pos_ltp = self.get_ltp_with_fallback(self.position.symbol)
            pnl = (pos_ltp - self.position.entry_price) * self.position.quantity if pos_ltp > 0 else 0
            pnl_points = pos_ltp - self.position.entry_price if pos_ltp > 0 else 0
            be_status = "[BE]" if self.position.breakeven_activated else ""
            status += f" | POS: {self.position.option_type} LTP: {pos_ltp:.2f} (+{pnl_points:.1f}pts) PnL: {pnl:.2f} SL: {self.position.sl_price:.2f} {be_status}"
            status += f" Tgt: {self.position.target_price:.2f}"

        self.log(status)

    def safe_execute(self, func: Callable, description: str, *args, **kwargs):
        """Execute a function with error handling and retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"{description} failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    def run(self):
        """Main strategy loop - optimized for speed (25pt scalping)"""
        self.running = True
        self.log(f"Starting {self.config.STRATEGY_NAME} for {self.config.INDEX} | Target: +25pts | BE: 20pts")

        try:
            # FIRST: Check market hours (local check, no API needed)
            # This allows immediate exit if running after market close
            can_trade, hours_message = self.is_market_hours()
            if not can_trade:
                self.log(f"{hours_message}. Exiting strategy.")
                return

            self.log(f"Market hours: {hours_message}")

            # SECOND: Check if today is a trading day (holiday check via API)
            is_trading, message = is_trading_day(self.client)
            if not is_trading:
                self.log(f"{message}. Exiting strategy.")
                return

            self.log(f"Market status: {message}")

            # Wait for market open (if before 9:15)
            self.wait_for_market_open()

            # Wait for first 15-min candle (9:30)
            self.wait_for_first_candle()

            # Initialize capital allocation (check once at start, cap at 2L if > 2.5L)
            self.initialize_capital()
            if self.allocated_capital <= 0:
                self.log("Insufficient capital - cannot trade. Exiting.")
                return

            # Calculate entry levels (with retry)
            self.entry_levels = self.safe_execute(
                self.calculate_entry_levels,
                "Calculate entry levels"
            )

            # Setup WebSocket (with retry)
            self.safe_execute(self.setup_websocket, "Setup WebSocket")

            # Main trading loop - optimized for speed
            self.log("Trading loop active. Monitoring for entries...")

            while self.running:
                try:
                    # Check force exit time (15:00)
                    if self.is_force_exit_time():
                        if self.position:
                            self.log("Force exit time reached")
                            self.close_position("TIME_EXIT")
                        break

                    # Check WebSocket health
                    if self.tick_manager.state == ConnectionState.FAILED:
                        if self.position:
                            self.close_position("WS_FAILED")
                        break

                    # Check if max completed trades reached - exit strategy
                    if not self.position and not self.can_trade:
                        self.log("Max completed trades reached - exiting strategy")
                        break

                    # Skip if WebSocket not connected
                    if not self.tick_manager.connected:
                        time.sleep(1)
                        continue

                    # If we have a position, manage it
                    if self.position:
                        self.manage_position()

                    # If no position and new entries allowed, check entry conditions
                    elif self.is_new_entry_allowed():
                        # Check PE entry
                        if check_entry_condition_ws(
                            self.tick_manager,
                            self.entry_levels.pe_symbol,
                            self.entry_levels.pe_entry_level
                        ):
                            ltp = self.tick_manager.get_ltp(self.entry_levels.pe_symbol)
                            self.log(f"✓ PE ENTRY @ {ltp:.2f}")
                            self.position = self.place_order("PE", ltp)

                        # Check CE entry
                        elif check_entry_condition_ws(
                            self.tick_manager,
                            self.entry_levels.ce_symbol,
                            self.entry_levels.ce_entry_level
                        ):
                            ltp = self.tick_manager.get_ltp(self.entry_levels.ce_symbol)
                            self.log(f"✓ CE ENTRY @ {ltp:.2f}")
                            self.position = self.place_order("CE", ltp)

                    # Fast loop for scalping (50ms)
                    time.sleep(0.05)

                except Exception as loop_error:
                    self.logger.error(f"Error in main loop: {loop_error}")
                    self.logger.debug(traceback.format_exc())
                    time.sleep(1)  # Prevent tight error loop

        except KeyboardInterrupt:
            self.log("Strategy interrupted by user (Ctrl+C)")
            self.running = False
            if self.position:
                self.close_position("MANUAL_EXIT")

        except Exception as e:
            self.logger.critical(f"Critical error: {str(e)}")
            self.logger.error(traceback.format_exc())
            if self.position:
                try:
                    self.close_position("ERROR_EXIT")
                except Exception:
                    self.logger.error("Failed to exit position on error")
            raise  # Re-raise for crash recovery

        finally:
            self.running = False
            self.cleanup_websocket()
            self.print_summary()
            self.log("Strategy finished.")

    def _log_health_status(self):
        """Log current health status"""
        ws_state = self.tick_manager.state.value
        stale = self.tick_manager.is_data_stale()
        reconnects = self.tick_manager.reconnect_attempts

        self.logger.info(
            f"[Health] WS: {ws_state}, Stale: {stale}, Reconnects: {reconnects}, "
            f"Completed: {self.completed_trades}/{self.config.MAX_COMPLETED_TRADES}, PnL: {self.daily_pnl:.2f}"
        )

    def stop(self):
        """Gracefully stop the strategy"""
        self.log("Stop requested...")
        self.running = False


# =============================================================================
# ENTRY POINT WITH CRASH RECOVERY
# =============================================================================

def run_with_crash_recovery():
    """Run strategy with automatic crash recovery"""
    crash_count = 0

    while True:
        try:
            # Create fresh config and strategy instance
            config = Config()
            strategy = OptionsAlphaStrategy(config)

            if crash_count > 0:
                strategy.log(f"Restarting after crash (attempt {crash_count}/{MAX_CRASH_RESTARTS})")

            # Run strategy
            strategy.run()

            # If we get here normally (no exception), exit the loop
            break

        except KeyboardInterrupt:
            print("\n[MAIN] Keyboard interrupt - exiting")
            break

        except Exception as e:
            crash_count += 1
            print(f"\n[MAIN] Strategy crashed: {e}")
            print(traceback.format_exc())

            if not AUTO_RESTART_ON_CRASH:
                print("[MAIN] Auto-restart disabled - exiting")
                break

            if crash_count >= MAX_CRASH_RESTARTS:
                print(f"[MAIN] Max crash restarts ({MAX_CRASH_RESTARTS}) exceeded - exiting")
                break

            # Check if market is still open before restarting
            now = datetime.now()
            exit_time = datetime.strptime(
                f"{now.strftime('%Y-%m-%d')} 15:00",
                "%Y-%m-%d %H:%M"
            )
            if now >= exit_time:
                print("[MAIN] Market closed - not restarting")
                break

            # Wait before restart
            restart_delay = 10 * crash_count  # Increasing delay
            print(f"[MAIN] Restarting in {restart_delay} seconds...")
            time.sleep(restart_delay)


if __name__ == "__main__":
    # Validate API key
    if API_KEY == "YOUR_API_KEY_HERE" or len(API_KEY) < 10:
        print("=" * 60)
        print("ERROR: API key not configured!")
        print("Configure API key in ~/.config/openalgo/config.json")
        print("Get your key from: http://127.0.0.1:5003/apikey")
        print("=" * 60)
        exit(1)

    print("=" * 60)
    print("Options Alpha 25 - Simplified Version")
    print("Target: Fixed 25 points | Breakeven SL at 20 points")
    print("Features: WebSocket, Auto-Reconnect, Crash Recovery")
    print("=" * 60)

    # Run with crash recovery wrapper
    run_with_crash_recovery()
