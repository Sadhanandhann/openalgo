#!/bin/bash
# Strategy Runner - Unsets problematic SSL environment variables
# This ensures API calls work correctly on macOS with Homebrew Python

# Unset old SSL certificate paths
unset SSL_CERT_FILE
unset REQUESTS_CA_BUNDLE

# Run the strategy
uv run nifty_optionsalpha.py "$@"
