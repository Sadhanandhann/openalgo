# User Strategy - Running Guide

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

The `run_strategy.sh` script will be preserved because it's committed to your fork's `dhan` branch. No action needed!

## Strategy Configuration

Edit strategy parameters in `nifty_optionsalpha.py`:
- `EXPIRY_WEEK`: Which weekly expiry to use (1 = current week)
- `ENTRY_TIME`: Time to enter trades
- `EXIT_TIME`: Time to exit trades
- `STOP_LOSS_PERCENT`: Stop loss percentage
- `TARGET_PERCENT`: Target profit percentage

## Logs

Strategy logs are written to:
```
/Users/sadhanandhann/Code/openalgo_v2/openalgo_dhan/user_strategy/logs/
```
