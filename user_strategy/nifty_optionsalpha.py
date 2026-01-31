#!/usr/bin/env python
"""
Options Alpha Strategy - First 15-Min Breakout (WebSocket Version)
===================================================================
A momentum-based options buying strategy that:
1. Waits for first 15-min candle to form
2. Calculates ATM strike and entry levels
3. Uses WebSocket for real-time price updates
4. Enters on breakout with trailing SL
5. Manages position locally with target/SL

Author: Generated with Claude
Version: 2.1 (WebSocket + Holiday Check)
"""

import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, List
from collections import deque
import pandas as pd
from openalgo import api


# =============================================================================
# USER CONFIGURATION - EDIT THESE VALUES
# =============================================================================

# OpenAlgo API Key (get from http://127.0.0.1:5000/apikey)
API_KEY = "5901954d90a7aeb2339a374cc2cfb8a3e868c8c61cbdddc5c7b89b3d8c84d562"

# OpenAlgo Server
HOST = "http://127.0.0.1:5000"
WS_URL = "ws://127.0.0.1:8765"

# Index to trade
INDEX = "NIFTY"                    # NIFTY, BANKNIFTY, SENSEX

# Expiry week (1 = current week, 2 = next week, etc.)
EXPIRY_WEEK = 1

# Capital & Position Sizing
CAPITAL_PERCENT = 0.80             # Use 80% of available capital

# Risk Management
SL_PERCENT = 0.05                  # 5% stop loss below entry level
TARGET_MULTIPLIER = 2.80           # 180% profit target (entry * 2.80)

# Trailing SL
TRAIL_TRIGGER_PERCENT = 0.20       # Move SL to cost at 20% profit
TRAIL_RATIO = 1.0                  # 1:1 trailing after trigger

# Trading Limits
MAX_TRADES_PER_DAY = 2

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
    TARGET_MULTIPLIER: float = TARGET_MULTIPLIER
    TRAIL_TRIGGER_PERCENT: float = TRAIL_TRIGGER_PERCENT
    TRAIL_RATIO: float = TRAIL_RATIO
    MAX_TRADES_PER_DAY: int = MAX_TRADES_PER_DAY

    # Timing (IST) - usually don't need to change
    MARKET_OPEN: str = "09:15"
    FIRST_CANDLE_CLOSE: str = "09:30"
    MARKET_CLOSE: str = "15:30"
    EXIT_TIME: str = "14:59"

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
        "lot_size": 75  # Fallback, actual fetched from API
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
    """Active position tracking"""
    option_type: Literal["CE", "PE"]
    symbol: str
    entry_price: float
    quantity: int
    sl_price: float
    target_price: float
    entry_time: datetime
    order_id: str
    cost_basis: float  # Total cost for PnL calculation
    trail_triggered: bool = False


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
# REAL-TIME TICK MANAGER (WebSocket)
# =============================================================================

class TickManager:
    """Manages real-time tick data via WebSocket"""

    def __init__(self, client: api, config: Config):
        self.client = client
        self.config = config
        self.ticks: Dict[str, TickData] = {}
        self.lock = threading.Lock()
        self.connected = False
        self._current_minute: Dict[str, int] = {}

    def on_tick(self, data: dict):
        """Callback for WebSocket tick updates"""
        try:
            ltp_data = data.get("ltp", {})
            for exchange, symbols in ltp_data.items():
                for symbol, info in symbols.items():
                    ltp = float(info.get("ltp", 0))
                    if ltp > 0:
                        self._update_tick(symbol, ltp)
        except Exception as e:
            print(f"[TickManager] Error processing tick: {e}")

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

    def subscribe(self, symbols: List[dict]):
        """Subscribe to symbols via WebSocket"""
        try:
            self.client.connect()
            self.client.subscribe_ltp(symbols, on_data_received=self.on_tick)
            self.connected = True
            print(f"[TickManager] Subscribed to {len(symbols)} symbols")
        except Exception as e:
            print(f"[TickManager] Subscribe error: {e}")
            self.connected = False

    def unsubscribe(self, symbols: List[dict]):
        """Unsubscribe from symbols"""
        try:
            self.client.unsubscribe_ltp(symbols)
        except Exception as e:
            print(f"[TickManager] Unsubscribe error: {e}")

    def disconnect(self):
        """Disconnect WebSocket"""
        try:
            self.client.disconnect()
            self.connected = False
            print("[TickManager] Disconnected")
        except Exception as e:
            print(f"[TickManager] Disconnect error: {e}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_trading_day(client: api) -> tuple[bool, str]:
    """
    Check if today is a trading day.
    Returns: (is_trading: bool, message: str)
    """
    today = datetime.now()

    # Check if weekend
    if today.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False, "Weekend - Market closed"

    # Check market timings API
    try:
        result = client.timings(date=today.strftime("%Y-%m-%d"))
        if result.get("status") == "success":
            data = result.get("data", [])
            if not data:
                return False, "Holiday - Market closed"

            # Check if NSE/NFO is open
            for exchange in data:
                if exchange.get("exchange") in ["NSE", "NFO"]:
                    return True, "Trading day"

            return False, "NSE/NFO not open today"
    except Exception as e:
        # If API fails, assume it's a trading day (fail open)
        print(f"Warning: Could not check market timings: {e}")
        return True, "Assuming trading day (API check failed)"

    return True, "Trading day"


def get_nearest_strike(spot_price: float, strike_interval: int) -> int:
    """Calculate nearest strike price (ATM)"""
    return round(spot_price / strike_interval) * strike_interval


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
    """
    return f"{index}{expiry_date}{strike}{option_type}"


def get_lot_size(client: api, symbol: str, exchange: str) -> int:
    """Fetch lot size for an option symbol"""
    result = client.symbol(symbol=symbol, exchange=exchange)
    if result.get("status") == "success":
        return result.get("data", {}).get("lotsize", 75)
    return INDEX_CONFIG.get(symbol[:5], {}).get("lot_size", 75)


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
    """Calculate quantity based on capital and lot size"""
    if price <= 0:
        return 0
    max_lots = int(capital / (price * lot_size))
    return max(max_lots, 1) * lot_size  # At least 1 lot


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
    # Get LTP
    ltp = tick_manager.get_ltp(symbol)
    if ltp <= entry_level:
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
    """Main strategy class with WebSocket support"""

    def __init__(self, config: Config):
        self.config = config
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

        # Tick manager for WebSocket
        self.tick_manager = TickManager(self.client, config)

        # State
        self.entry_levels: Optional[EntryLevels] = None
        self.position: Optional[Position] = None
        self.trades_today: int = 0
        self.daily_pnl: float = 0.0
        self.trade_log: list = []
        self.subscribed_symbols: List[dict] = []

    def log(self, message: str):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def wait_for_market_open(self):
        """Wait until market opens"""
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

    def wait_for_first_candle(self):
        """Wait for first 15-min candle to complete (9:30)"""
        self.log(f"Waiting for first 15-min candle to close at {self.config.FIRST_CANDLE_CLOSE}...")

        while True:
            now = datetime.now()
            candle_close = datetime.strptime(
                f"{now.strftime('%Y-%m-%d')} {self.config.FIRST_CANDLE_CLOSE}",
                "%Y-%m-%d %H:%M"
            )

            if now >= candle_close:
                break

            time.sleep(5)

        # Wait a bit more for data to be available
        time.sleep(10)
        self.log("First 15-min candle closed!")

    def calculate_entry_levels(self) -> EntryLevels:
        """Calculate ATM strike and entry levels from first 15-min candle"""
        self.log("Calculating entry levels...")

        # Get first 15-min candle of index
        index_candle = get_15min_candle(
            self.client,
            self.config.INDEX,
            self.index_exchange
        )
        spot_close = index_candle["close"]
        self.log(f"{self.config.INDEX} first 15-min candle close: {spot_close}")

        # Calculate ATM strike
        atm_strike = get_nearest_strike(spot_close, self.strike_interval)
        self.log(f"ATM Strike: {atm_strike}")

        # Get expiry date
        expiry_date = get_expiry_date(
            self.client,
            self.config.INDEX,
            self.options_exchange,
            self.config.EXPIRY_WEEK
        )
        self.log(f"Expiry Date: {expiry_date}")

        # Build option symbols
        pe_symbol = build_option_symbol(self.config.INDEX, expiry_date, atm_strike, "PE")
        ce_symbol = build_option_symbol(self.config.INDEX, expiry_date, atm_strike, "CE")
        self.log(f"PE Symbol: {pe_symbol}, CE Symbol: {ce_symbol}")

        # Get lot size
        lot_size = get_lot_size(self.client, ce_symbol, self.options_exchange)
        self.log(f"Lot Size: {lot_size}")

        # Get first 15-min candle for PE and CE
        pe_candle = get_15min_candle(self.client, pe_symbol, self.options_exchange)
        ce_candle = get_15min_candle(self.client, ce_symbol, self.options_exchange)

        self.log(f"PE 15-min candle: Low={pe_candle['low']}, High={pe_candle['high']}")
        self.log(f"CE 15-min candle: Low={ce_candle['low']}, High={ce_candle['high']}")

        # Calculate entry levels
        # PE_entry_level = Avg(PE 15min low, CE 15min high)
        pe_entry_level = (pe_candle["low"] + ce_candle["high"]) / 2

        # CE_entry_level = Avg(CE 15min low, CE 15min high)
        ce_entry_level = (ce_candle["low"] + ce_candle["high"]) / 2

        self.log(f"PE Entry Level: {pe_entry_level:.2f}")
        self.log(f"CE Entry Level: {ce_entry_level:.2f}")

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
        self.log("Setting up WebSocket connection...")

        self.subscribed_symbols = [
            {"exchange": self.options_exchange, "symbol": self.entry_levels.pe_symbol},
            {"exchange": self.options_exchange, "symbol": self.entry_levels.ce_symbol}
        ]

        self.tick_manager.subscribe(self.subscribed_symbols)

        # Wait for initial ticks
        self.log("Waiting for initial tick data...")
        time.sleep(3)

        # Verify we're receiving data
        pe_ltp = self.tick_manager.get_ltp(self.entry_levels.pe_symbol)
        ce_ltp = self.tick_manager.get_ltp(self.entry_levels.ce_symbol)
        self.log(f"Initial LTP - PE: {pe_ltp:.2f}, CE: {ce_ltp:.2f}")

    def cleanup_websocket(self):
        """Cleanup WebSocket connection"""
        if self.subscribed_symbols:
            self.tick_manager.unsubscribe(self.subscribed_symbols)
        self.tick_manager.disconnect()

    def get_available_capital(self) -> float:
        """Get available capital from account"""
        result = self.client.funds()
        if result.get("status") == "success":
            available = float(result.get("data", {}).get("availablecash", 0))
            return available * self.config.CAPITAL_PERCENT
        return 0.0

    def place_order(self, option_type: Literal["CE", "PE"], entry_price: float) -> Optional[Position]:
        """Place buy order and create position"""
        if self.trades_today >= self.config.MAX_TRADES_PER_DAY:
            self.log(f"Max trades ({self.config.MAX_TRADES_PER_DAY}) reached for today")
            return None

        symbol = self.entry_levels.ce_symbol if option_type == "CE" else self.entry_levels.pe_symbol
        entry_level = self.entry_levels.ce_entry_level if option_type == "CE" else self.entry_levels.pe_entry_level

        # Calculate quantity
        capital = self.get_available_capital()
        quantity = calculate_quantity(capital, entry_price, self.entry_levels.lot_size)

        if quantity <= 0:
            self.log("Insufficient capital for order")
            return None

        self.log(f"Placing {option_type} BUY order: {symbol} @ ~{entry_price:.2f}, Qty: {quantity}")

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
            self.log(f"Order failed: {result.get('message')}")
            return None

        order_id = result.get("orderid", "")
        self.log(f"Order placed successfully! Order ID: {order_id}")

        # Get actual fill price from WebSocket
        time.sleep(1)
        actual_price = self.tick_manager.get_ltp(symbol)
        if actual_price <= 0:
            actual_price = entry_price

        # Calculate SL and Target
        sl_price = entry_level * (1 - self.config.SL_PERCENT)
        target_price = actual_price * self.config.TARGET_MULTIPLIER

        self.trades_today += 1

        position = Position(
            option_type=option_type,
            symbol=symbol,
            entry_price=actual_price,
            quantity=quantity,
            sl_price=sl_price,
            target_price=target_price,
            entry_time=datetime.now(),
            order_id=order_id,
            cost_basis=actual_price * quantity,
            trail_triggered=False
        )

        self.log(f"Position created: Entry={actual_price:.2f}, SL={sl_price:.2f}, Target={target_price:.2f}")
        return position

    def manage_position(self) -> bool:
        """
        Manage active position with trailing SL using WebSocket data.
        Returns True if position is closed, False otherwise.
        """
        if not self.position:
            return False

        # Get LTP from WebSocket (instant!)
        ltp = self.tick_manager.get_ltp(self.position.symbol)
        if ltp <= 0:
            return False

        current_value = ltp * self.position.quantity
        pnl = current_value - self.position.cost_basis
        pnl_percent = pnl / self.position.cost_basis

        # Check target hit
        if ltp >= self.position.target_price:
            self.log(f"TARGET HIT! LTP: {ltp:.2f}, Target: {self.position.target_price:.2f}")
            return self.exit_position("TARGET")

        # Check SL hit
        if ltp <= self.position.sl_price:
            self.log(f"SL HIT! LTP: {ltp:.2f}, SL: {self.position.sl_price:.2f}")
            return self.exit_position("STOPLOSS")

        # Trailing SL logic
        if not self.position.trail_triggered:
            # Check if trail trigger reached (20% profit)
            if pnl_percent >= self.config.TRAIL_TRIGGER_PERCENT:
                self.position.sl_price = self.position.entry_price  # Move SL to cost
                self.position.trail_triggered = True
                self.log(f"Trail triggered! SL moved to cost: {self.position.sl_price:.2f}")
        else:
            # 1:1 trailing after trigger
            # If PnL is X%, SL should be at (X - TRAIL_TRIGGER_PERCENT)% profit
            if pnl_percent > self.config.TRAIL_TRIGGER_PERCENT:
                trail_profit = pnl_percent - self.config.TRAIL_TRIGGER_PERCENT
                new_sl = self.position.entry_price * (1 + trail_profit * self.config.TRAIL_RATIO)
                if new_sl > self.position.sl_price:
                    self.position.sl_price = new_sl
                    self.log(f"SL trailed to: {self.position.sl_price:.2f} (PnL: {pnl_percent*100:.1f}%)")

        return False

    def exit_position(self, reason: str) -> bool:
        """Exit current position"""
        if not self.position:
            return False

        self.log(f"Exiting position: {reason}")

        result = self.client.placeorder(
            strategy=self.config.STRATEGY_NAME,
            symbol=self.position.symbol,
            exchange=self.options_exchange,
            action="SELL",
            quantity=self.position.quantity,
            price_type="MARKET",
            product="MIS"
        )

        if result.get("status") != "success":
            self.log(f"Exit order failed: {result.get('message')}")
            return False

        # Get exit price from WebSocket
        time.sleep(1)
        exit_price = self.tick_manager.get_ltp(self.position.symbol)
        if exit_price <= 0:
            exit_price = self.position.sl_price if reason == "STOPLOSS" else self.position.target_price

        pnl = (exit_price - self.position.entry_price) * self.position.quantity
        self.daily_pnl += pnl

        trade_record = {
            "symbol": self.position.symbol,
            "option_type": self.position.option_type,
            "entry_price": self.position.entry_price,
            "exit_price": exit_price,
            "quantity": self.position.quantity,
            "pnl": pnl,
            "exit_reason": reason,
            "entry_time": self.position.entry_time.strftime("%H:%M:%S"),
            "exit_time": datetime.now().strftime("%H:%M:%S")
        }
        self.trade_log.append(trade_record)

        self.log(f"Position closed: Entry={self.position.entry_price:.2f}, Exit={exit_price:.2f}, PnL={pnl:.2f}")
        self.position = None
        return True

    def is_exit_time(self) -> bool:
        """Check if it's time to force exit"""
        now = datetime.now()
        exit_time = datetime.strptime(
            f"{now.strftime('%Y-%m-%d')} {self.config.EXIT_TIME}",
            "%Y-%m-%d %H:%M"
        )
        return now >= exit_time

    def print_summary(self):
        """Print end of day summary"""
        self.log("=" * 60)
        self.log("END OF DAY SUMMARY")
        self.log("=" * 60)
        self.log(f"Index: {self.config.INDEX}")
        self.log(f"ATM Strike: {self.entry_levels.atm_strike if self.entry_levels else 'N/A'}")
        self.log(f"Total Trades: {self.trades_today}")
        self.log(f"Daily PnL: {self.daily_pnl:.2f}")
        self.log("-" * 60)

        for i, trade in enumerate(self.trade_log, 1):
            self.log(f"Trade {i}: {trade['option_type']} {trade['symbol']}")
            self.log(f"  Entry: {trade['entry_price']:.2f} @ {trade['entry_time']}")
            self.log(f"  Exit: {trade['exit_price']:.2f} @ {trade['exit_time']} ({trade['exit_reason']})")
            self.log(f"  PnL: {trade['pnl']:.2f}")

        self.log("=" * 60)

    def print_status(self):
        """Print current status (called periodically)"""
        if not self.entry_levels:
            return

        pe_ltp = self.tick_manager.get_ltp(self.entry_levels.pe_symbol)
        ce_ltp = self.tick_manager.get_ltp(self.entry_levels.ce_symbol)

        status = f"PE: {pe_ltp:.2f} (Entry: {self.entry_levels.pe_entry_level:.2f}) | "
        status += f"CE: {ce_ltp:.2f} (Entry: {self.entry_levels.ce_entry_level:.2f})"

        if self.position:
            pos_ltp = self.tick_manager.get_ltp(self.position.symbol)
            pnl = (pos_ltp - self.position.entry_price) * self.position.quantity
            status += f" | POS: {self.position.option_type} PnL: {pnl:.2f} SL: {self.position.sl_price:.2f}"

        self.log(status)

    def run(self):
        """Main strategy loop"""
        self.log(f"Starting {self.config.STRATEGY_NAME} Strategy for {self.config.INDEX} (WebSocket)")

        try:
            # Check if today is a trading day
            is_trading, message = is_trading_day(self.client)
            if not is_trading:
                self.log(f"{message}. Exiting strategy.")
                return

            self.log(f"Market status: {message}")

            # Wait for market open
            self.wait_for_market_open()

            # Wait for first 15-min candle
            self.wait_for_first_candle()

            # Calculate entry levels
            self.entry_levels = self.calculate_entry_levels()

            # Setup WebSocket
            self.setup_websocket()

            # Main trading loop
            self.log("Starting main trading loop (WebSocket mode)...")
            last_status_time = datetime.now()
            status_interval = 30  # Print status every 30 seconds

            while True:
                # Check exit time
                if self.is_exit_time():
                    self.log("Exit time reached!")
                    if self.position:
                        self.exit_position("TIME_EXIT")
                    break

                # Print status periodically
                if (datetime.now() - last_status_time).seconds >= status_interval:
                    self.print_status()
                    last_status_time = datetime.now()

                # If we have a position, manage it
                if self.position:
                    self.manage_position()

                # If no position and haven't hit max trades, check entry conditions
                elif self.trades_today < self.config.MAX_TRADES_PER_DAY:
                    # Check PE entry condition
                    if check_entry_condition_ws(
                        self.tick_manager,
                        self.entry_levels.pe_symbol,
                        self.entry_levels.pe_entry_level
                    ):
                        ltp = self.tick_manager.get_ltp(self.entry_levels.pe_symbol)
                        self.log(f"PE Entry condition met! LTP: {ltp:.2f}")
                        self.position = self.place_order("PE", ltp)

                    # Check CE entry condition
                    elif check_entry_condition_ws(
                        self.tick_manager,
                        self.entry_levels.ce_symbol,
                        self.entry_levels.ce_entry_level
                    ):
                        ltp = self.tick_manager.get_ltp(self.entry_levels.ce_symbol)
                        self.log(f"CE Entry condition met! LTP: {ltp:.2f}")
                        self.position = self.place_order("CE", ltp)

                # Small sleep to prevent CPU spinning (WebSocket handles the data)
                time.sleep(0.1)

        except KeyboardInterrupt:
            self.log("Strategy interrupted by user")
            if self.position:
                self.exit_position("MANUAL_EXIT")

        except Exception as e:
            self.log(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            if self.position:
                self.exit_position("ERROR_EXIT")

        finally:
            self.cleanup_websocket()
            self.print_summary()
            self.log("Strategy finished.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Validate API key
    if API_KEY == "your_api_key_here" or len(API_KEY) < 10:
        print("=" * 60)
        print("ERROR: API key not configured!")
        print("Edit the API_KEY variable at the top of this file")
        print("Get your key from: http://127.0.0.1:5000/apikey")
        print("=" * 60)
        exit(1)

    # Create config (uses top-level constants)
    config = Config()

    # Run strategy
    strategy = OptionsAlphaStrategy(config)
    strategy.run()
