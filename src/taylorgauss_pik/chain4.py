"""Public four-site finite-slice authorities and Gaussian-shift controls."""

import math

from .chain4_authority import (
    Chain4Case,
    ChannelAuthority,
    channel_authority,
    determinant,
    physical_trace,
)
from .chain4_multislice import DenseAuthority, dense_authority, validate_dense
from .chain4_pik import ChannelPIK as GaussianShift
from .chain4_pik import endpoint_observables, sample_channels, validate_authority


def nt4_negative_coefficient(*, beta: float, kappa: float, mu: float) -> float:
    """Closed-form four-slice coefficient for occupation masks (3, 5, 6, 10)."""
    for name, value in (("beta", beta), ("kappa", kappa), ("mu", mu)):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    x = beta * kappa / 4
    return float(
        math.exp(2 * beta * mu)
        * (math.sinh(2 * x) / 2) ** 4
        * (math.cosh(2 * x) ** 4 - 3)
    )


__all__ = [
    "Chain4Case",
    "ChannelAuthority",
    "DenseAuthority",
    "GaussianShift",
    "channel_authority",
    "dense_authority",
    "determinant",
    "endpoint_observables",
    "nt4_negative_coefficient",
    "physical_trace",
    "sample_channels",
    "validate_authority",
    "validate_dense",
]
