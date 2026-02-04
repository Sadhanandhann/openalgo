# User Strategy - Running Guide (Zerodha Instance)

## Quick Start

### Run the Options Alpha Strategy

```bash
# Recommended: Use the wrapper script
./run_strategy.sh

# Or directly with uv (ensure SSL env vars are unset first)
uv run nifty_optionsalpha.py
```

## SSL Certificate Fix (macOS)

**Issue**: If you see SSL certificate errors like:
```
[Errno 2] No such file or directory
Failed to fetch expiry dates: An unexpected error occurred
```

**Cause**: Old `SSL_CERT_FILE` environment variable pointing to deleted Python framework.

**Solution**: Use `run_strategy.sh` wrapper script which automatically unsets these variables.

### Why This Happens

On macOS, if you previously had Python installed via the official installer at `/Library/Frameworks/Python.framework/`, the environment variable `SSL_CERT_FILE` may still be set to point to that location. When you switch to Homebrew Python, this path no longer exists.

### Permanent Fix Options

1. **Use the wrapper script** (recommended for this project):
   ```bash
   ./run_strategy.sh
   ```

2. **Reload your shell** (for new terminal sessions):
   ```bash
   source ~/.zshrc  # or open a new terminal
   ```

3. **Unset manually** (for current session):
   ```bash
   unset SSL_CERT_FILE REQUESTS_CA_BUNDLE
   uv run nifty_optionsalpha.py
   ```

## After OpenAlgo Updates

When you update OpenAlgo from upstream:

```bash
git pull upstream main
```

The `run_strategy.sh` script will be preserved because it's committed to your fork's `zerodha` branch. No action needed!

## Zerodha Instance Configuration

This strategy is configured for the Zerodha OpenAlgo instance:
- **API URL**: http://127.0.0.1:5001
- **WebSocket**: ws://127.0.0.1:8766
- **API Key**: Loaded from `~/.config/openalgo/client.py`

Make sure:
1. Zerodha OpenAlgo is running on port 5001
2. You're logged into your Zerodha broker account
3. Your API key is configured in `~/.config/openalgo/client.py`

## Strategy Configuration

Edit strategy parameters in `nifty_optionsalpha.py`:
- `INDEX`: Which index to trade (NIFTY, BANKNIFTY, SENSEX)
- `EXPIRY_WEEK`: Which weekly expiry to use (1 = current week)
- `ENTRY_TIME`: Time to enter trades
- `EXIT_TIME`: Time to exit trades
- `SL_PERCENT`: Stop loss percentage
- `TARGET_MULTIPLIER`: Target profit multiplier

## Logs

Strategy logs are written to:
```
/Users/sadhanandhann/Code/openalgo_v2/openalgo_zerodha/user_strategy/logs/
```

## Symbol Format

OpenAlgo uses a unified symbol format across all brokers. The strategy automatically:
- Fetches expiry dates via API
- Calculates ATM strike from index price
- Builds correct option symbols (e.g., NIFTY10FEB2625750CE)
- Maps symbols to broker-specific format internally

No manual symbol formatting needed!

## Running Both Dhan and Zerodha Strategies

You can run strategies on both broker instances simultaneously:

**Dhan Instance** (port 5003):
```bash
cd /Users/sadhanandhann/Code/openalgo_v2/openalgo_dhan/user_strategy
./run_strategy.sh
```

**Zerodha Instance** (port 5001):
```bash
cd /Users/sadhanandhann/Code/openalgo_v2/openalgo_zerodha/user_strategy
./run_strategy.sh
```

Both will use their respective broker accounts and operate independently.
