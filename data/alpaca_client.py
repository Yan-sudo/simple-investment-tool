"""
Alpaca Markets API Integration
================================

Provides real market data and paper-trading capabilities:
- Historical OHLCV bars for SPY / QQQ / any symbol
- Account info, positions, and order management
- Converts Alpaca data into the project's MarketData format

Setup:
    export ALPACA_API_KEY="your-key-here"
    export ALPACA_SECRET_KEY="your-secret-here"
    export ALPACA_BASE_URL="https://paper-api.alpaca.markets"  # paper trading

    # For live trading (use with caution):
    # export ALPACA_BASE_URL="https://api.alpaca.markets"

Uses only Python standard library (urllib + json) — no extra packages needed.
"""

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

from data.market_data import MarketData


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class AlpacaConfig:
    """Alpaca API configuration — loaded from environment variables."""
    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"

    @classmethod
    def from_env(cls) -> 'AlpacaConfig':
        return cls(
            api_key=os.environ.get("ALPACA_API_KEY", ""),
            secret_key=os.environ.get("ALPACA_SECRET_KEY", ""),
            base_url=os.environ.get(
                "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
            ),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key)

    @property
    def is_paper(self) -> bool:
        return "paper" in self.base_url


# ── API Client ────────────────────────────────────────────────────────────────

class AlpacaClient:
    """Alpaca Markets REST API client.

    Handles authentication, request construction, and response parsing.
    Uses only Python stdlib (urllib) — zero external dependencies.
    """

    def __init__(self, config: Optional[AlpacaConfig] = None):
        self.config = config or AlpacaConfig.from_env()

    @property
    def is_connected(self) -> bool:
        """Test if API credentials are valid."""
        if not self.config.is_configured:
            return False
        try:
            self.get_account()
            return True
        except Exception:
            return False

    def _request(self, url: str, method: str = "GET",
                 data: Optional[Dict] = None,
                 timeout: int = 15) -> Dict:
        """Make authenticated API request."""
        headers = {
            "APCA-API-KEY-ID": self.config.api_key,
            "APCA-API-SECRET-KEY": self.config.secret_key,
            "Content-Type": "application/json",
        }

        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            raise AlpacaAPIError(
                f"HTTP {e.code}: {e.reason} — {error_body}"
            ) from e
        except urllib.error.URLError as e:
            raise AlpacaAPIError(f"Connection error: {e.reason}") from e

    def _trading_url(self, path: str) -> str:
        return f"{self.config.base_url}/v2{path}"

    def _data_url(self, path: str) -> str:
        return f"{self.config.data_url}/v2{path}"

    # ── Account ───────────────────────────────────────────────────────────

    def get_account(self) -> Dict:
        """Get account information (buying power, equity, etc.)."""
        return self._request(self._trading_url("/account"))

    def get_positions(self) -> List[Dict]:
        """Get all open positions."""
        return self._request(self._trading_url("/positions"))

    # ── Market Data ───────────────────────────────────────────────────────

    def get_bars(self, symbol: str, timeframe: str = "1Day",
                 start: Optional[str] = None, end: Optional[str] = None,
                 limit: int = 1000) -> List[Dict]:
        """Fetch historical OHLCV bars.

        Args:
            symbol: Ticker symbol (e.g. "SPY", "QQQ")
            timeframe: Bar timeframe ("1Day", "1Hour", "1Min")
            start: Start date ISO string (YYYY-MM-DD)
            end: End date ISO string (YYYY-MM-DD)
            limit: Max number of bars (default 1000)

        Returns:
            List of bar dicts with keys: t, o, h, l, c, v
        """
        if start is None:
            start = (datetime.now() - timedelta(days=limit * 2)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        url = (f"{self._data_url(f'/stocks/{symbol}/bars')}"
               f"?timeframe={timeframe}&start={start}&end={end}&limit={limit}"
               f"&adjustment=split&feed=iex")

        all_bars = []
        next_token = None

        while True:
            page_url = url
            if next_token:
                page_url += f"&page_token={next_token}"

            resp = self._request(page_url)
            bars = resp.get("bars", [])
            all_bars.extend(bars)

            next_token = resp.get("next_page_token")
            if not next_token or len(all_bars) >= limit:
                break

        return all_bars[:limit]

    def get_latest_quote(self, symbol: str) -> Dict:
        """Get latest quote for a symbol."""
        return self._request(
            self._data_url(f"/stocks/{symbol}/quotes/latest") + "?feed=iex"
        )

    def get_snapshot(self, symbol: str) -> Dict:
        """Get latest snapshot (quote + trade + bar) for a symbol."""
        return self._request(
            self._data_url(f"/stocks/{symbol}/snapshot") + "?feed=iex"
        )

    # ── Orders ────────────────────────────────────────────────────────────

    def submit_order(self, symbol: str, qty: int, side: str,
                     order_type: str = "market",
                     time_in_force: str = "day",
                     limit_price: Optional[float] = None) -> Dict:
        """Submit a stock order.

        Args:
            symbol: Ticker symbol
            qty: Number of shares
            side: "buy" or "sell"
            order_type: "market", "limit", "stop", "stop_limit"
            time_in_force: "day", "gtc", "ioc", "fok"
            limit_price: Required for limit orders
        """
        order_data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if limit_price is not None:
            order_data["limit_price"] = str(limit_price)

        return self._request(
            self._trading_url("/orders"), method="POST", data=order_data
        )

    def get_orders(self, status: str = "open", limit: int = 50) -> List[Dict]:
        """Get orders by status."""
        return self._request(
            self._trading_url(f"/orders?status={status}&limit={limit}")
        )

    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an open order."""
        return self._request(
            self._trading_url(f"/orders/{order_id}"), method="DELETE"
        )

    def cancel_all_orders(self) -> Dict:
        """Cancel all open orders."""
        return self._request(
            self._trading_url("/orders"), method="DELETE"
        )

    # ── Data Conversion ───────────────────────────────────────────────────

    def fetch_market_data(self, symbol: str = "SPY",
                          days: int = 750) -> MarketData:
        """Fetch historical data and convert to MarketData format.

        This is the main integration point — lets you replace synthetic data
        with real Alpaca data seamlessly.

        Usage:
            client = AlpacaClient()
            data = client.fetch_market_data("SPY", days=500)
            # Now use 'data' exactly like generate_spy_data()
        """
        start = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y-%m-%d")
        bars = self.get_bars(symbol, timeframe="1Day", start=start, limit=days)

        if not bars:
            raise AlpacaAPIError(f"No data returned for {symbol}")

        dates = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []

        for bar in bars:
            # Alpaca v2 bar format
            t = bar.get("t", "")[:10]  # YYYY-MM-DD
            dates.append(t)
            opens.append(float(bar.get("o", 0)))
            highs.append(float(bar.get("h", 0)))
            lows.append(float(bar.get("l", 0)))
            closes.append(float(bar.get("c", 0)))
            volumes.append(int(bar.get("v", 0)))

        return MarketData(
            dates=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            volume=volumes,
        )


class AlpacaAPIError(Exception):
    """Raised when an Alpaca API call fails."""
    pass


# ── Convenience Functions ─────────────────────────────────────────────────────

def create_client() -> AlpacaClient:
    """Create an AlpacaClient from environment variables."""
    return AlpacaClient(AlpacaConfig.from_env())


def test_connection() -> Tuple[bool, str]:
    """Test Alpaca API connection. Returns (success, message)."""
    config = AlpacaConfig.from_env()
    if not config.is_configured:
        return False, (
            "Alpaca API not configured. Set environment variables:\n"
            "  export ALPACA_API_KEY='your-key'\n"
            "  export ALPACA_SECRET_KEY='your-secret'"
        )
    try:
        client = AlpacaClient(config)
        account = client.get_account()
        mode = "Paper" if config.is_paper else "LIVE"
        equity = float(account.get("equity", 0))
        buying_power = float(account.get("buying_power", 0))
        return True, (
            f"Connected to Alpaca ({mode} Trading)\n"
            f"  Equity:       ${equity:,.2f}\n"
            f"  Buying Power: ${buying_power:,.2f}"
        )
    except AlpacaAPIError as e:
        return False, f"Alpaca API error: {e}"
    except Exception as e:
        return False, f"Connection failed: {e}"
