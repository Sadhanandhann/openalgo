# Strategy Comparison: optionalpha_25 vs nifty_optionsalpha

## Overview

Both strategies have been improved with critical bug fixes, Excel performance tracking, and better error handling. Run them on alternate days to compare performance.

## Key Differences

### optionalpha_25.py (Simplified Scalper)
- **Target**: Fixed 25 points from entry
- **Stop Loss**: Initial 5% below entry, moves to breakeven at +20 points
- **Entry Logic**: Simple momentum breakout from 9:30 levels
- **Loop Speed**: 50ms (optimized for scalping)
- **Max Trades**: 2 per day
- **Exit Time**: 14:59
- **Excel File**: `optionalpha_performance.xlsx`

### nifty_optionsalpha.py (Resistance Level Strategy)
- **Target**: Dynamic R1-R6 resistance levels (calculated from straddle/strangle pricing)
- **Stop Loss**: Trailing 10% below current R level
- **Entry Logic**: 5-min candle confirmation + TRUE ATM at 9:20
- **Loop Speed**: 100ms (standard)
- **Max Trades**: 2 per day  
- **Exit Time**: 15:00
- **Excel File**: `nifty_optionsalpha_performance.xlsx`

## Excel Tracking Fields

Both strategies log identical fields for comparison:

| Field | Description |
|-------|-------------|
| Date | Trade date (YYYY-MM-DD) |
| ATM Strike | Calculated ATM strike at 9:30/9:20 |
| PE Symbol | PE option symbol calculated |
| PE Entry Level | PE entry level calculated |
| CE Symbol | CE option symbol calculated |
| CE Entry Level | CE entry level calculated |
| Expiry | Expiry date used |
| Entry Order ID | Entry order ID from broker |
| Exit Order ID | Exit order ID from broker |
| Trade Type | CE or PE (what was actually traded) |
| Traded Symbol | Actual option symbol traded |
| Entry Time | Entry execution time (HH:MM:SS) |
| Entry Price | Broker execution price (from tradebook) |
| Initial SL | Original stop loss at entry |
| Target Price | Target price at entry |
| Exit Time | Exit execution time |
| Exit Price | Broker execution price (from tradebook) |
| Final SL | Final stop loss (after trailing) |
| Quantity | Lot size × lots traded |
| PnL | Profit/Loss in ₹ |
| Exit Reason | TARGET, STOPLOSS, TIME_EXIT, etc. |
| Breakeven Activated | Yes/No (25 only) |
| R Level Reached | R1-R6 (nifty only) |

## Common Improvements Applied

Both strategies now have:

✅ **Critical Bug Fixes**
- Order ID shadowing fixed (entry_order_id vs exit_order_id)
- Recursive reconnect → iterative (prevents stack overflow)
- calculate_quantity edge cases (insufficient capital check)

✅ **Enhanced Error Handling**
- Symbol parsing with try-catch
- Fail-safe trading day check (don't trade if uncertain)
- LTP validation (check for zero/stale data)

✅ **Thread Safety**
- Health check thread leak prevention
- Proper thread lifecycle management

✅ **Excel Performance Tracking**
- Automatic logging after every trade
- Long-term historical record
- Same sheet, new row per trade

## Performance Comparison Workflow

### Day 1: Run optionalpha_25
```bash
strategy_25
```
- Fixed 25-point scalper
- Breakeven protection at +20 points
- Fast execution (50ms loop)

### Day 2: Run nifty_optionsalpha
```bash
cd /Users/sadhanandhann/Code/openalgo_v2/openalgo_dhan/user_strategy/optionalpha
./run_strategy.sh nifty_optionsalpha.py
```
- Dynamic resistance levels (R1-R6)
- Trailing SL based on resistance
- TRUE ATM at 9:20 + 5-min candle

### Compare Results

After a month, compare:
1. **Win Rate**: Check both Excel files for % profitable trades
2. **Average PnL per Trade**: Which strategy generates better per-trade profits?
3. **Drawdown**: Which strategy has smaller losing trades?
4. **Consistency**: Which has more stable daily P&L?
5. **Max Profit**: Which captures bigger moves (R6 vs fixed 25)?

## Recommended Testing Schedule

- **Week 1-2**: Alternate daily (Mon/Wed/Fri = strategy_25, Tue/Thu = nifty)
- **Week 3-4**: Run each for full week to see weekly performance
- **Month 2**: Focus on better performer, tweak parameters

## Files Location

All files in: `/Users/sadhanandhann/Code/openalgo_v2/openalgo_dhan/user_strategy/optionalpha/`

- `optionalpha_25.py` - Simplified scalper
- `nifty_optionsalpha.py` - Resistance level strategy  
- `run_strategy.sh` - Wrapper script (SSL fix)
- `optionalpha_performance.xlsx` - Scalper trade history
- `nifty_optionsalpha_performance.xlsx` - Resistance strategy trade history
- `IMPROVEMENTS.md` - Detailed fix documentation
- `STRATEGY_COMPARISON.md` - This file

## Quick Start Alias

Already configured in `~/.zshrc`:
```bash
strategy_25  # Runs optionalpha_25.py from anywhere
```

For nifty strategy:
```bash
cd /Users/sadhanandhann/Code/openalgo_v2/openalgo_dhan/user_strategy/optionalpha
./run_strategy.sh nifty_optionsalpha.py
```

