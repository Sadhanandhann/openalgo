# Options Alpha Strategy

This directory contains two Options Alpha trading strategies with critical bug fixes, Excel performance tracking, and enhanced error handling.

## 🚀 Quick Start

### Run Simplified 25-Point Strategy (from anywhere):
```bash
strategy_25
```

### Run Resistance Level Strategy:
```bash
cd /Users/sadhanandhann/Code/openalgo_v2/openalgo_dhan/user_strategy/optionalpha
./run_strategy.sh nifty_optionsalpha.py
```

## 📊 Strategies

### 1. optionalpha_25.py (Simplified Scalper)
- **Target**: Fixed 25 points from entry
- **Stop Loss**: Initial 5% below entry, moves to breakeven at +20 points
- **Entry**: Momentum breakout from 9:30 calculated levels
- **Max Trades**: 2 per day
- **Exit Time**: 14:59
- **Loop Speed**: 50ms (optimized for scalping)
- **Excel File**: `optionalpha_performance.xlsx`

### 2. nifty_optionsalpha.py (Resistance Level Strategy)
- **Target**: R1-R6 resistance levels (dynamic, calculated from straddle/strangle pricing)
- **Stop Loss**: Trailing 10% below current R level
- **Entry**: TRUE ATM at 9:20 + 5-min candle confirmation
- **Max Trades**: 2 per day
- **Exit Time**: 15:00
- **Loop Speed**: 100ms (standard)
- **Excel File**: `nifty_optionsalpha_performance.xlsx`

## ✅ Recent Improvements (Applied to Both Strategies)

### Critical Bug Fixes
1. **Order ID Shadowing Fixed** - Entry and exit order IDs now tracked separately
2. **Recursive Reconnect Fixed** - Changed to iterative to prevent stack overflow
3. **Capital Validation** - Returns 0 if insufficient capital (prevents failed orders)
4. **Symbol Parsing** - Added error handling for ATM strike extraction
5. **Fail-Safe Trading Day Check** - Won't trade if API fails to confirm trading day
6. **LTP Validation** - Checks for zero/stale data before entry
7. **Thread Safety** - Prevents multiple health check threads

### Enhanced Features
- **Excel Performance Tracking** - Automatic logging of all trades with 22+ fields
- **Tradebook Integration** - Actual broker execution prices for entry and exit
- **Both Order IDs Logged** - Entry and exit order IDs for complete audit trail
- **Better Error Messages** - Clear exception messages for debugging
- **Win/Loss Statistics** - Daily summary shows win rate and performance metrics

See `IMPROVEMENTS.md` for detailed list of all 43 issues identified and fixes applied.

## 📈 Performance Tracking

### Excel Logging (Automatic)

Each strategy logs to its own Excel file:
- **optionalpha_25.py** → `optionalpha_performance.xlsx`
- **nifty_optionsalpha.py** → `nifty_optionsalpha_performance.xlsx`

### Fields Logged (22 total):
- Date, ATM Strike, PE/CE Symbols, PE/CE Entry Levels
- Expiry Date
- Entry Order ID, Exit Order ID
- Trade Type (CE/PE), Traded Symbol
- Entry Time, Entry Price (from tradebook)
- Initial SL, Target Price
- Exit Time, Exit Price (from tradebook)
- Final SL (after trailing)
- Quantity, PnL
- Exit Reason (TARGET, STOPLOSS, TIME_EXIT, etc.)
- Breakeven Activated (optionalpha_25 only)
- R Level Reached (nifty_optionsalpha only)

### Console Summary (End of Day)
- Completed trades count
- Win rate (W/L ratio)
- Daily PnL
- Trade-by-trade breakdown with all details

## 🎯 Strategy Comparison

Run both strategies on alternate days to compare performance over time.

**Recommended Schedule:**
- **Mon/Wed/Fri**: `strategy_25` (scalper)
- **Tue/Thu**: `nifty_optionsalpha` (resistance levels)

After 1 month, compare:
1. Win rate
2. Average PnL per trade
3. Maximum drawdown
4. Consistency
5. Which captures bigger moves?

See `STRATEGY_COMPARISON.md` for detailed comparison guide.

## ⚙️ Configuration

Edit these constants at the top of each strategy file:

```python
INDEX = "NIFTY"                    # NIFTY, BANKNIFTY, SENSEX
EXPIRY_WEEK = 1                    # 1=current, 2=next week
CAPITAL_PERCENT = 0.90             # Use 90% of available capital
MAX_COMPLETED_TRADES = 2           # Max trades per day
```

**optionalpha_25.py specific:**
```python
TARGET_POINTS = 25                 # Fixed 25-point target
BREAKEVEN_POINTS = 20              # Move SL to cost at +20 points
FORCE_EXIT_TIME = "14:59"          # Exit time
```

**nifty_optionsalpha.py specific:**
```python
FORCE_EXIT_TIME = "15:00"          # Exit time
# R1-R6 levels calculated automatically from BEP and straddle pricing
```

## 📋 Requirements

- **OpenAlgo API**: http://127.0.0.1:5003
- **WebSocket Server**: ws://127.0.0.1:8765
- **API Key**: Configured in `~/.config/openalgo/config.json`
- **Broker**: Live connection with sufficient margin
- **Python Packages**: pandas, openpyxl (auto-installed via uv)

## 📁 Files

### Strategy Files
- `optionalpha_25.py` - Simplified 25-point scalper
- `nifty_optionsalpha.py` - Resistance level strategy
- `run_strategy.sh` - Strategy runner (unsets SSL env vars)

### Performance Tracking
- `optionalpha_performance.xlsx` - Trade history for scalper (auto-generated)
- `nifty_optionsalpha_performance.xlsx` - Trade history for resistance strategy (auto-generated)

### Documentation
- `README.md` - This file
- `IMPROVEMENTS.md` - Detailed bug fix documentation (43 issues identified)
- `STRATEGY_COMPARISON.md` - Side-by-side comparison guide

## 🛠️ Manual Usage

### Run with uv (recommended):
```bash
cd /Users/sadhanandhann/Code/openalgo_v2/openalgo_dhan/user_strategy/optionalpha
uv run python3 optionalpha_25.py
```

### Run with wrapper script:
```bash
./run_strategy.sh optionalpha_25.py
# or
./run_strategy.sh nifty_optionsalpha.py
```

## 🔍 Troubleshooting

### Excel file not created
- Check write permissions in the directory
- Pandas/openpyxl will be auto-installed on first run via uv

### Order verification fails
- Check OpenAlgo API is running on port 5003
- Verify API key in `~/.config/openalgo/config.json`

### WebSocket connection issues
- Ensure WebSocket server is running on port 8765
- Check firewall settings

### Strategy won't start
- Run `./run_strategy.sh <file>` to unset SSL env vars
- Check broker connection and margin availability

## 📊 Performance Analysis

After collecting data, analyze using:
```python
import pandas as pd

# Load both strategy results
df_25 = pd.read_excel('optionalpha_performance.xlsx')
df_nifty = pd.read_excel('nifty_optionsalpha_performance.xlsx')

# Compare win rates
print(f"optionalpha_25 win rate: {(df_25['PnL'] > 0).mean():.2%}")
print(f"nifty_optionsalpha win rate: {(df_nifty['PnL'] > 0).mean():.2%}")

# Compare average PnL
print(f"optionalpha_25 avg PnL: ₹{df_25['PnL'].mean():.2f}")
print(f"nifty_optionsalpha avg PnL: ₹{df_nifty['PnL'].mean():.2f}")
```

## 📝 Notes

- Both strategies use **actual broker execution prices** from tradebook
- Trades are tracked by **entry order ID** to avoid confusion with manual orders
- Strategies will **not trade** if they cannot confirm it's a trading day (fail-safe)
- **Max 2 trades per day** by default (configurable)
- All trades automatically logged to Excel - **no manual recording needed**

---

**Version**: 2.0 (with critical bug fixes and Excel tracking)
**Last Updated**: 2026-02-05
**Author**: Generated with Claude Code
