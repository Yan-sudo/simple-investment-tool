"""
Black-Scholes Option Pricing Engine with Greeks

Implements the Black-Scholes-Merton model for European option pricing,
plus all first-order Greeks (Delta, Gamma, Theta, Vega, Rho).
Uses only Python standard library (math module for erf/exp/log/sqrt).
"""

import math
from dataclasses import dataclass
from typing import Tuple


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
    """Calculate d1 and d2 for Black-Scholes formula.

    Args:
        S: Current underlying price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)

    Returns:
        Tuple of (d1, d2)
    """
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call option price.

    Args:
        S: Current underlying price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Implied volatility

    Returns:
        Call option price
    """
    if T <= 0:
        return max(S - K, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put option price.

    Args:
        S: Current underlying price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Implied volatility

    Returns:
        Put option price
    """
    if T <= 0:
        return max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def option_price(S: float, K: float, T: float, r: float, sigma: float,
                 option_type: str = "call") -> float:
    """Calculate option price for either calls or puts."""
    if option_type.lower() == "call":
        return call_price(S, K, T, r, sigma)
    else:
        return put_price(S, K, T, r, sigma)


# ── Greeks ───────────────────────────────────────────────────────────────────

def delta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call") -> float:
    """Option Delta: rate of change of option price w.r.t. underlying price."""
    if T <= 0:
        if option_type.lower() == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return _norm_cdf(d1)
    else:
        return _norm_cdf(d1) - 1.0


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Option Gamma: rate of change of delta w.r.t. underlying price.
    Same for calls and puts."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def theta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call") -> float:
    """Option Theta: rate of change of option price w.r.t. time (per day).
    Returned as daily theta (divided by 365)."""
    if T <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    common = -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if option_type.lower() == "call":
        annual_theta = common - r * K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        annual_theta = common + r * K * math.exp(-r * T) * _norm_cdf(-d2)
    return annual_theta / 365.0


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Option Vega: rate of change of option price w.r.t. volatility.
    Returns per 1% move in vol (divided by 100). Same for calls and puts."""
    if T <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * _norm_pdf(d1) * math.sqrt(T) / 100.0


def rho(S: float, K: float, T: float, r: float, sigma: float,
        option_type: str = "call") -> float:
    """Option Rho: rate of change of option price w.r.t. interest rate.
    Returns per 1% move in rate (divided by 100)."""
    if T <= 0:
        return 0.0
    _, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return K * T * math.exp(-r * T) * _norm_cdf(d2) / 100.0
    else:
        return -K * T * math.exp(-r * T) * _norm_cdf(-d2) / 100.0


@dataclass
class GreeksResult:
    """All Greeks for a single option."""
    delta: float
    gamma: float
    theta: float  # daily
    vega: float   # per 1% vol move
    rho: float    # per 1% rate move

    def __repr__(self):
        return (f"Greeks(Δ={self.delta:+.4f}, Γ={self.gamma:.4f}, "
                f"Θ={self.theta:+.4f}/day, V={self.vega:.4f}/1%vol, "
                f"ρ={self.rho:+.4f}/1%rate)")


def all_greeks(S: float, K: float, T: float, r: float, sigma: float,
               option_type: str = "call") -> GreeksResult:
    """Calculate all Greeks at once."""
    return GreeksResult(
        delta=delta(S, K, T, r, sigma, option_type),
        gamma=gamma(S, K, T, r, sigma),
        theta=theta(S, K, T, r, sigma, option_type),
        vega=vega(S, K, T, r, sigma),
        rho=rho(S, K, T, r, sigma, option_type),
    )


# ── Implied Volatility (Newton-Raphson) ─────────────────────────────────────

def implied_volatility(market_price: float, S: float, K: float, T: float,
                       r: float, option_type: str = "call",
                       max_iterations: int = 100, tolerance: float = 1e-8) -> float:
    """Calculate implied volatility using Newton-Raphson method.

    Args:
        market_price: Observed market price of the option
        S: Underlying price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate
        option_type: "call" or "put"
        max_iterations: Max Newton-Raphson iterations
        tolerance: Convergence tolerance

    Returns:
        Implied volatility (annualized)
    """
    if T <= 0:
        return 0.0

    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = math.sqrt(2.0 * math.pi / T) * market_price / S
    sigma = max(sigma, 0.01)  # floor at 1%

    for _ in range(max_iterations):
        price = option_price(S, K, T, r, sigma, option_type)
        diff = price - market_price

        if abs(diff) < tolerance:
            return sigma

        # Vega for Newton step (not the per-1% version)
        d1, _ = _d1_d2(S, K, T, r, sigma)
        v = S * _norm_pdf(d1) * math.sqrt(T)

        if v < 1e-12:
            break

        sigma -= diff / v
        sigma = max(sigma, 0.001)  # prevent negative vol

    return sigma
