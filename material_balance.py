"""
Material Balance for Gas Reservoirs

Implements P/Z volumetric decline, OGIP estimation, and Havlena-Odeh
gas material balance.

References:
- Craft, Hawkins & Terry, "Applied Petroleum Reservoir Engineering" (3rd ed.)
- Havlena & Odeh, "The Material Balance as an Equation of a Straight Line",
  JPT, August 1963
- Dake, "Fundamentals of Reservoir Engineering", Ch. 3
"""

import numpy as np
from typing import Tuple


def pz_linear_regression(pressures: np.ndarray, z_factors: np.ndarray,
                         cum_production: np.ndarray) -> Tuple[float, float, float]:
    """
    Perform P/Z vs cumulative production linear regression.

    For a volumetric dry-gas reservoir (no water influx):
        P/Z = (Pi/Zi) * (1 - Gp/G)

    Args:
        pressures: Measured shut-in pressures (psia), array length N
        z_factors: Z-factors at each pressure, array length N
        cum_production: Cumulative gas production (BCF) at each pressure, length N

    Returns:
        Tuple of (ogip_bcf, slope, intercept) where:
            ogip_bcf: Original Gas In Place in BCF (x-intercept)
            slope: Regression slope (negative)
            intercept: Regression intercept (P/Z at Gp=0 ≈ Pi/Zi)
    """
    pz = pressures / z_factors
    # Linear regression: P/Z = intercept + slope * Gp
    coeffs = np.polyfit(cum_production, pz, 1)
    slope, intercept = coeffs[0], coeffs[1]

    # OGIP is where P/Z = 0 → Gp = -intercept / slope
    ogip_bcf = -intercept / slope if slope != 0 else np.inf

    return ogip_bcf, slope, intercept


def extrapolate_recovery(slope: float, intercept: float,
                         p_abandon: float, z_abandon: float) -> Tuple[float, float]:
    """
    Calculate recoverable reserves at abandonment conditions.

    Args:
        slope: P/Z regression slope
        intercept: P/Z regression intercept (≈ Pi/Zi)
        p_abandon: Abandonment pressure (psia)
        z_abandon: Z-factor at abandonment pressure

    Returns:
        Tuple of (gp_abandon_bcf, recovery_factor) where:
            gp_abandon_bcf: Cumulative production at abandonment (BCF)
            recovery_factor: Recovery factor (fraction, 0-1)
    """
    pz_abandon = p_abandon / z_abandon
    # From regression: pz_abandon = intercept + slope * Gp
    gp_abandon = (pz_abandon - intercept) / slope
    ogip = -intercept / slope
    rf = gp_abandon / ogip if ogip > 0 else 0.0
    return max(gp_abandon, 0.0), max(min(rf, 1.0), 0.0)


def havlena_odeh_gas(pressures: np.ndarray, z_factors: np.ndarray,
                     cum_production: np.ndarray,
                     cf: float = 3.0e-6, cw: float = 3.0e-6,
                     swi: float = 0.25, bgi: float = None,
                     pi: float = None, zi: float = None
                     ) -> Tuple[float, float, float]:
    """
    Havlena-Odeh material balance for gas reservoirs with
    rock and water compressibility corrections.

    F = G * Eg  (no water influx case)

    Where:
        F  = Bg - Bgi  (gas expansion term, adjusted for cum production)
        Eg = Bg - Bgi * (1 + cf_eff * ΔP)

    Rearranged as straight line: F/Eg vs 1 → slope = G (OGIP).

    For simplicity this implementation uses the P/Z corrected form:
        (P/Z) * [1 - ce*ΔP] where ce = (cf + cw*Swi)/(1 - Swi)

    Args:
        pressures: Measured pressures (psia)
        z_factors: Z-factors at each pressure
        cum_production: Cumulative production (BCF)
        cf: Formation compressibility (1/psi), default 3e-6
        cw: Water compressibility (1/psi), default 3e-6
        swi: Initial water saturation (fraction), default 0.25
        bgi: Initial gas FVF (optional, calculated from first point)
        pi: Initial pressure (optional, uses first data point)
        zi: Initial Z-factor (optional, uses first data point)

    Returns:
        Tuple of (ogip_bcf, slope, intercept) from corrected analysis
    """
    if pi is None:
        pi = pressures[0]
    if zi is None:
        zi = z_factors[0]

    ce = (cf + cw * swi) / (1 - swi)

    pz_corrected = np.zeros_like(pressures, dtype=float)
    for i, (p, z) in enumerate(zip(pressures, z_factors)):
        delta_p = pi - p
        correction = 1 + ce * delta_p
        pz_corrected[i] = (p / z) * correction

    coeffs = np.polyfit(cum_production, pz_corrected, 1)
    slope, intercept = coeffs[0], coeffs[1]
    ogip_bcf = -intercept / slope if slope != 0 else np.inf

    return ogip_bcf, slope, intercept
