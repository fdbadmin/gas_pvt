"""
Hydrate Prediction for Natural Gas Systems

Implements Katz gravity chart correlation for hydrate equilibrium
and Hammerschmidt inhibitor dosing.

References:
- Katz, "Prediction of Conditions Favoring Hydrate Formation in
  Natural Gases", Trans. AIME (1945), Vol. 160
- Hammerschmidt, "Formation of Gas Hydrates in Natural Gas Transmission
  Lines", Ind. Eng. Chem. (1934), Vol. 26
- Carroll, "Natural Gas Hydrates: A Guide for Engineers" (3rd ed.)
- Sloan & Koh, "Clathrate Hydrates of Natural Gases" (3rd ed.)
"""

import math
import numpy as np
from typing import Tuple, List


def katz_hydrate_temperature(pressure: float, gas_grav: float) -> float:
    """
    Estimate hydrate formation temperature at a given pressure
    using the Katz gas gravity chart correlation.

    Correlation fitted to Katz chart data for gas gravities 0.55–1.0.

    Args:
        pressure: Pressure (psia)
        gas_grav: Gas specific gravity (air = 1.0)

    Returns:
        Hydrate formation temperature (°F)
    """
    if pressure < 50:
        return -999.0  # Below practical hydrate range

    ln_p = math.log(pressure)
    # Polynomial fit to Katz chart (digitized data regression)
    # T_hyd = a0 + a1*ln(P) + a2*ln(P)^2 + a3*gamma + a4*gamma*ln(P)
    a0 = -82.09
    a1 = 28.27
    a2 = -1.585
    a3 = 18.23
    a4 = 5.45

    t_hyd = a0 + a1 * ln_p + a2 * ln_p ** 2 + a3 * gas_grav + a4 * gas_grav * ln_p
    return t_hyd


def hydrate_envelope(gas_grav: float, p_min: float = 100, p_max: float = 10000,
                     n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the hydrate equilibrium curve (P vs T).

    Args:
        gas_grav: Gas specific gravity
        p_min: Minimum pressure (psia)
        p_max: Maximum pressure (psia)
        n_points: Number of curve points

    Returns:
        Tuple of (temperatures_F, pressures_psia)
    """
    pressures = np.logspace(math.log10(p_min), math.log10(p_max), n_points)
    temperatures = np.array([katz_hydrate_temperature(p, gas_grav) for p in pressures])
    return temperatures, pressures


def subcooling(operating_temp: float, pressure: float, gas_grav: float) -> float:
    """
    Calculate subcooling below hydrate equilibrium temperature.

    Args:
        operating_temp: Operating temperature (°F)
        pressure: Operating pressure (psia)
        gas_grav: Gas specific gravity

    Returns:
        Subcooling (°F). Positive means operating below hydrate T (hydrate risk).
        Negative means operating above hydrate T (safe).
    """
    t_hyd = katz_hydrate_temperature(pressure, gas_grav)
    return t_hyd - operating_temp


def hammerschmidt_concentration(delta_t: float, inhibitor: str = "methanol") -> float:
    """
    Calculate required inhibitor concentration using Hammerschmidt equation.

        ΔT = K · W / (M · (100 - W))

    Rearranged:
        W = 100 · M · ΔT / (K + M · ΔT)

    Args:
        delta_t: Required temperature depression (°F)
        inhibitor: 'methanol' (M=32, K=2335) or 'meg' (M=62, K=2335)

    Returns:
        Required inhibitor concentration (wt%)
    """
    if delta_t <= 0:
        return 0.0

    inhibitor_props = {
        "methanol": {"M": 32.04, "K": 2335.0},
        "meg":      {"M": 62.07, "K": 2335.0},
        "deg":      {"M": 106.12, "K": 2335.0},
        "teg":      {"M": 150.17, "K": 2335.0},
    }

    props = inhibitor_props.get(inhibitor.lower(), inhibitor_props["methanol"])
    m = props["M"]
    k = props["K"]

    w = 100.0 * m * delta_t / (k + m * delta_t)
    return min(w, 100.0)


def inhibitor_rate(concentration_wt_pct: float, water_rate_bpd: float,
                   inhibitor: str = "methanol") -> float:
    """
    Calculate inhibitor injection rate.

    Args:
        concentration_wt_pct: Required wt% in aqueous phase
        water_rate_bpd: Produced water rate (bbl/d)
        inhibitor: Inhibitor type

    Returns:
        Inhibitor injection rate (gal/d)
    """
    if concentration_wt_pct <= 0 or concentration_wt_pct >= 100:
        return 0.0

    densities = {
        "methanol": 6.63,  # lb/gal
        "meg": 9.34,
        "deg": 8.81,
        "teg": 9.40,
    }

    rho_inh = densities.get(inhibitor.lower(), 6.63)
    rho_water = 8.34  # lb/gal for fresh water

    # Water mass rate (lb/d): 1 bbl = 42 gal
    water_mass = water_rate_bpd * 42.0 * rho_water

    # W = mass_inh / (mass_inh + mass_water)
    # mass_inh = W * mass_water / (1 - W)
    w = concentration_wt_pct / 100.0
    inh_mass = w * water_mass / (1 - w)

    return inh_mass / rho_inh  # gal/d
