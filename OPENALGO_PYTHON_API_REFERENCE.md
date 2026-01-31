# OpenAlgo Python API Reference

Complete reference for the `openalgo` Python library methods.

**Installation:** `pip install openalgo`

**Documentation:** https://docs.openalgo.in

---

## Initialization

```python
from openalgo import api

client = api(
    api_key="your_api_key",
    host="http://127.0.0.1:5000",  # OpenAlgo server URL
    ws_url="ws://127.0.0.1:8765",  # WebSocket URL (optional)
    timeout=120.0,                  # Request timeout in seconds
    verbose=False                   # 0=silent, 1=info, 2=debug
)
```

---

## 1. Order Management

### placeorder()
Place a regular order.

```python
client.placeorder(
    symbol="RELIANCE",       # Required: Trading symbol
    exchange="NSE",          # Required: NSE, BSE, NFO, MCX, etc.
    action="BUY",            # Required: BUY or SELL
    quantity=1,              # Quantity to trade
    price_type="MARKET",     # MARKET, LIMIT, SL, SL-M
    product="MIS",           # MIS, CNC, NRML
    strategy="Python",       # Strategy name
    price="2500.00",         # For LIMIT orders
    trigger_price="2490.00"  # For SL/SL-M orders
)
```

### placesmartorder()
Place order with position sizing (adjusts based on current position).

```python
client.placesmartorder(
    symbol="RELIANCE",
    exchange="NSE",
    action="BUY",
    quantity=1,
    position_size=100,       # Required: Target position size
    price_type="MARKET",
    product="MIS"
)
```

### basketorder()
Place multiple orders simultaneously.

```python
orders = [
    {"symbol": "RELIANCE", "exchange": "NSE", "action": "BUY", "quantity": 1, "pricetype": "MARKET", "product": "MIS"},
    {"symbol": "INFY", "exchange": "NSE", "action": "SELL", "quantity": 1, "pricetype": "MARKET", "product": "MIS"}
]
client.basketorder(orders=orders)
```

### splitorder()
Split large order into smaller chunks.

```python
client.splitorder(
    symbol="YESBANK",
    exchange="NSE",
    action="SELL",
    quantity=105,
    splitsize=20,            # Each chunk size
    price_type="MARKET",
    product="MIS"
)
```

### modifyorder()
Modify an existing order.

```python
client.modifyorder(
    order_id="24120900146469",
    symbol="RELIANCE",
    action="BUY",
    exchange="NSE",
    quantity=2,
    price="2100",
    product="MIS",
    price_type="LIMIT"
)
```

### cancelorder()
Cancel a specific order.

```python
client.cancelorder(order_id="24120900146469")
```

### cancelallorder()
Cancel all open orders.

```python
client.cancelallorder()
```

### orderstatus()
Get status of a specific order.

```python
client.orderstatus(order_id="24120900146469")
```

### openposition()
Get current open position for a symbol.

```python
client.openposition(
    symbol="YESBANK",
    exchange="NSE",
    product="CNC"
)
```

### closeposition()
Close all open positions.

```python
client.closeposition()
```

---

## 2. Account Information

### funds()
Get account funds and margin details.

```python
result = client.funds()
# Returns: availablecash, collateral, m2mrealized, m2munrealized, utiliseddebits
```

### orderbook()
Get all orders with statistics.

```python
result = client.orderbook()
# Returns: orders list + statistics (total_buy_orders, total_sell_orders, etc.)
```

### tradebook()
Get executed trades.

```python
result = client.tradebook()
# Returns: List of trades with average_price, quantity, trade_value, timestamp
```

### positionbook()
Get all current positions.

```python
result = client.positionbook()
# Returns: List of positions with symbol, quantity, average_price
```

### holdings()
Get stock holdings with P&L.

```python
result = client.holdings()
# Returns: holdings list + statistics (totalholdingvalue, totalprofitandloss)
```

### margin()
Calculate margin requirements for positions.

```python
# Single position
client.margin(positions=[{
    "symbol": "SBIN",
    "exchange": "NSE",
    "action": "BUY",
    "product": "MIS",
    "pricetype": "LIMIT",
    "quantity": "10",
    "price": "750.50"
}])

# Multiple positions (basket margin)
client.margin(positions=[
    {"symbol": "NIFTY30DEC2526000CE", "exchange": "NFO", "action": "SELL", "product": "NRML", "pricetype": "LIMIT", "quantity": "75", "price": "150"},
    {"symbol": "NIFTY30DEC2526000PE", "exchange": "NFO", "action": "SELL", "product": "NRML", "pricetype": "LIMIT", "quantity": "75", "price": "125"}
])
# Returns: total_margin_required, span_margin, exposure_margin
```

### analyzerstatus()
Get analyzer (paper trading) mode status.

```python
result = client.analyzerstatus()
# Returns: analyze_mode (bool), mode (live/analyze), total_logs
```

### analyzertoggle()
Toggle between live and analyzer mode.

```python
client.analyzertoggle(mode=True)   # Enable analyzer (paper trading)
client.analyzertoggle(mode=False)  # Switch to live trading
```

---

## 3. Market Data (REST)

### quotes()
Get real-time quote for a symbol.

```python
result = client.quotes(symbol="RELIANCE", exchange="NSE")
# Returns: ltp, bid, ask, volume, open, high, low, close
```

### multiquotes()
Get quotes for multiple symbols.

```python
result = client.multiquotes(symbols=[
    {"symbol": "RELIANCE", "exchange": "NSE"},
    {"symbol": "TCS", "exchange": "NSE"},
    {"symbol": "NIFTY", "exchange": "NSE_INDEX"}
])
```

### depth()
Get market depth (order book).

```python
result = client.depth(symbol="RELIANCE", exchange="NSE")
# Returns: Top 5 bids and asks with price, quantity, orders
```

### history()
Get historical OHLC data (returns pandas DataFrame).

```python
df = client.history(
    symbol="RELIANCE",
    exchange="NSE",
    interval="5m",           # 1m, 5m, 15m, 30m, 60m, D
    start_date="2024-01-01",
    end_date="2024-01-31"
)
# Returns: DataFrame with timestamp, open, high, low, close, volume
```

### intervals()
Get supported time intervals.

```python
result = client.intervals()
# Returns: seconds, minutes, hours, days, weeks, months intervals
```

### symbol()
Get symbol details.

```python
result = client.symbol(symbol="NIFTY24APR25FUT", exchange="NFO")
# Returns: token, lotsize, tick_size, expiry, instrumenttype
```

### search()
Search for symbols.

```python
result = client.search(query="RELIANCE")
result = client.search(query="NIFTY", exchange="NFO")
# Returns: List of matching symbols with details
```

### expiry()
Get expiry dates for derivatives.

```python
result = client.expiry(symbol="NIFTY", exchange="NFO", instrumenttype="futures")
result = client.expiry(symbol="NIFTY", exchange="NFO", instrumenttype="options")
```

### instruments()
Download all trading instruments.

```python
df = client.instruments()                    # All exchanges
df = client.instruments(exchange="NSE")      # Specific exchange
df = client.instruments(exchange="NFO")      # F&O instruments
# Returns: DataFrame with symbol, token, lotsize, expiry, etc.
```

### syntheticfuture()
Calculate synthetic futures price from options.

```python
result = client.syntheticfuture(
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="28NOV25"
)
# Returns: synthetic_future_price, atm_strike, underlying_ltp
```

---

## 4. WebSocket Feed (Real-time)

### connect() / disconnect()
Manage WebSocket connection.

```python
client.connect()
# ... subscribe and use data ...
client.disconnect()
```

### subscribe_ltp() / unsubscribe_ltp()
Subscribe to Last Traded Price updates.

```python
instruments = [
    {"exchange": "NSE", "symbol": "RELIANCE"},
    {"exchange": "NSE", "symbol": "TCS"}
]

def on_ltp(data):
    print(f"LTP Update: {data}")

client.connect()
client.subscribe_ltp(instruments, on_data_received=on_ltp)

# Poll data
ltp_data = client.get_ltp()

client.unsubscribe_ltp(instruments)
client.disconnect()
```

### subscribe_quote() / unsubscribe_quote()
Subscribe to quote updates (OHLC + LTP).

```python
client.subscribe_quote(instruments)
quotes = client.get_quotes()
client.unsubscribe_quote(instruments)
```

### subscribe_depth() / unsubscribe_depth()
Subscribe to market depth updates.

```python
client.subscribe_depth(instruments)
depth = client.get_depth()
client.unsubscribe_depth(instruments)
```

---

## 5. Options API

### optiongreeks()
Calculate Option Greeks and IV.

```python
greeks = client.optiongreeks(
    symbol="NIFTY28NOV2526000CE",
    exchange="NFO",
    interest_rate=6.5,                    # Optional: RBI repo rate
    underlying_symbol="NIFTY28NOV25FUT",  # Optional: Use futures as underlying
    underlying_exchange="NFO",
    expiry_time="15:30"                   # Optional: Custom expiry time
)
# Returns: delta, gamma, theta, vega, rho, implied_volatility
```

### optionsymbol()
Get option symbol details without placing order.

```python
result = client.optionsymbol(
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="28NOV24",
    offset="ATM",            # ATM, ITM1-ITM50, OTM1-OTM50
    option_type="CE"         # CE or PE
)
# Returns: symbol, lotsize, tick_size, underlying_ltp
```

### optionsorder()
Place option order with auto-symbol resolution.

```python
result = client.optionsorder(
    strategy="test_strategy",
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="28NOV24",
    offset="ATM",
    option_type="CE",
    action="BUY",
    quantity=75,
    price_type="MARKET",
    product="MIS"
)
```

### optionsmultiorder()
Place multiple option legs (spreads, straddles, etc.).

```python
# Iron Condor
result = client.optionsmultiorder(
    strategy="Iron Condor",
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="25NOV25",
    legs=[
        {"offset": "OTM10", "option_type": "CE", "action": "BUY", "quantity": 75},
        {"offset": "OTM10", "option_type": "PE", "action": "BUY", "quantity": 75},
        {"offset": "OTM5", "option_type": "CE", "action": "SELL", "quantity": 75},
        {"offset": "OTM5", "option_type": "PE", "action": "SELL", "quantity": 75}
    ]
)
```

### optionchain()
Get full option chain with real-time data.

```python
chain = client.optionchain(
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="30DEC25",
    strike_count=10          # Optional: Limit strikes around ATM
)
# Returns: chain with CE/PE data for each strike (ltp, bid, ask, oi, volume)
```

---

## 6. Strategy (Webhook)

For TradingView/external integrations.

```python
from openalgo import Strategy

strategy = Strategy(
    host_url="http://127.0.0.1:5000",
    webhook_id="your-webhook-id"
)

# Send order via webhook
strategy.strategyorder(
    symbol="RELIANCE",
    action="BUY",
    position_size=1
)
```

---

## 7. Telegram Notifications

```python
# Basic alert
client.telegram(
    username="your_openalgo_username",
    message="NIFTY crossed 24000!",
    priority=5               # 1-10 (higher = more urgent)
)

# Trade alert with formatting
client.telegram(
    username="trader",
    message="📈 BUY Signal\nSymbol: NIFTY 24000 CE\nEntry: ₹145.50\nTarget: ₹165.00",
    priority=9
)
```

**Priority Levels:**
- 1-3: Low (general updates)
- 4-6: Normal (trade signals)
- 7-8: High (price alerts)
- 9-10: Urgent (stop loss, risk alerts)

---

## 8. Utilities

### holidays()
Get market holidays.

```python
result = client.holidays(year=2025)
# Returns: List of holidays with date, description, closed_exchanges
```

### timings()
Get market timings for a date.

```python
result = client.timings(date="2025-01-15")
# Returns: Exchange-wise start_time and end_time
```

---

## Exchanges

| Code | Description |
|------|-------------|
| `NSE` | National Stock Exchange (Equity) |
| `BSE` | Bombay Stock Exchange (Equity) |
| `NFO` | NSE Futures & Options |
| `BFO` | BSE Futures & Options |
| `MCX` | Multi Commodity Exchange |
| `CDS` | Currency Derivatives |
| `NSE_INDEX` | NSE Indices (NIFTY, BANKNIFTY) |
| `BSE_INDEX` | BSE Indices (SENSEX) |

---

## Product Types

| Code | Description |
|------|-------------|
| `MIS` | Intraday (auto square-off) |
| `CNC` | Cash & Carry (delivery) |
| `NRML` | Normal (F&O carry forward) |

---

## Price Types

| Code | Description |
|------|-------------|
| `MARKET` | Market order |
| `LIMIT` | Limit order (requires `price`) |
| `SL` | Stop Loss Limit (requires `price` + `trigger_price`) |
| `SL-M` | Stop Loss Market (requires `trigger_price`) |

---

## Error Handling

All methods return a dict with `status` field:

```python
result = client.placeorder(...)

if result['status'] == 'success':
    print(f"Order ID: {result['orderid']}")
else:
    print(f"Error: {result['message']}")
    print(f"Error Type: {result.get('error_type')}")
```

**Error Types:**
- `timeout_error`: Request timed out
- `connection_error`: Cannot connect to server
- `http_error`: HTTP error (check status code)
- `api_error`: API returned error
- `validation_error`: Invalid parameters

---

## Quick Examples

### EMA Crossover Strategy

```python
from openalgo import api
import pandas as pd

client = api(api_key="your_key", host="http://127.0.0.1:5000")

# Get historical data
df = client.history(symbol="RELIANCE", exchange="NSE", interval="5m",
                    start_date="2024-01-01", end_date="2024-01-31")

# Calculate EMAs
df['ema_fast'] = df['close'].ewm(span=5).mean()
df['ema_slow'] = df['close'].ewm(span=20).mean()

# Check for crossover
if df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1]:
    client.placeorder(symbol="RELIANCE", exchange="NSE", action="BUY", quantity=1)
```

### Options Straddle

```python
# Sell ATM Straddle
client.optionsmultiorder(
    strategy="Short Straddle",
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="28NOV25",
    legs=[
        {"offset": "ATM", "option_type": "CE", "action": "SELL", "quantity": 75},
        {"offset": "ATM", "option_type": "PE", "action": "SELL", "quantity": 75}
    ]
)
```

### Real-time Price Monitor

```python
client.connect()

def on_price(data):
    for exchange, symbols in data.get('ltp', {}).items():
        for symbol, info in symbols.items():
            print(f"{symbol}: ₹{info['ltp']}")

instruments = [{"exchange": "NSE", "symbol": "RELIANCE"}]
client.subscribe_ltp(instruments, on_data_received=on_price)

import time
time.sleep(60)  # Monitor for 60 seconds

client.disconnect()
```
