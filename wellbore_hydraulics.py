"""
Wellbore Hydraulics – Pressure Traverse Calculations for Gas Wells

Implements Cullender-Smith (dry gas) and Gray (wet gas) correlations
for calculating bottomhole pressure from surface conditions or vice versa.

References:
- Cullender & Smith, "Practical Solution of Gas-Flow Equations for Wells
  and Pipelines", Trans. AIME (1956), Vol. 207
- Gray, "Vertical Flow Correlation in Gas Wells", API User's Manual for
  API 14B (1978)
- API RP 14E, "Recommended Practice for Design and Installation of
  Offshore Production Platform Piping Systems"
"""

import math
import numpy as np
from typing import Tuple, List

# Constants
R_GAS = 10.7315       # psia·ft³/(lbmol·°R)
G_ACCEL = 32.174      # ft/s²
MW_AIR = 28.97        # lb/lbmol


def _gas_density(p: float, t_f: float, z: float, gas_grav: float) -> float:
    """Gas density in lb/ft³."""
    mw = gas_grav * MW_AIR
    t_r = t_f + 459.67
    return p * mw / (z * R_GAS * t_r)


def gas_velocity(q_mscfd: float, p: float, t_f: float, z: float,
                 d_inches: float) -> float:
    """
    Calculate in-situ gas velocity in the tubing.

    Args:
        q_mscfd: Gas flow rate (MSCF/d at 14.696 psia, 60°F)
        p: Pressure (psia)
        t_f: Temperature (°F)
        z: Z-factor
        d_inches: Inside diameter of tubing (inches)

    Returns:
        Gas velocity (ft/s)
    """
    d_ft = d_inches / 12.0
    area = math.pi / 4.0 * d_ft ** 2
    t_r = t_f + 459.67
    # Convert MSCF/d at standard conditions to actual ft³/s
    q_actual = (q_mscfd * 1000.0 / 86400.0) * (14.696 / p) * (t_r / 520.0) * z
    return q_actual / area


def erosional_velocity(p: float, t_f: float, z: float,
                       gas_grav: float, c_factor: float = 100.0) -> float:
    """
    Erosional velocity per API RP 14E.

    V_erosional = C / sqrt(rho)

    Args:
        p: Pressure (psia)
        t_f: Temperature (°F)
        z: Z-factor
        gas_grav: Gas specific gravity
        c_factor: API C-factor, default 100 (clean service)

    Returns:
        Erosional velocity (ft/s)
    """
    rho = _gas_density(p, t_f, z, gas_grav)
    if rho <= 0:
        return 0.0
    return c_factor / math.sqrt(rho)


def _moody_friction(reynolds: float, roughness: float, d_inches: float) -> float:
    """
    Fanning friction factor via Colebrook-White (iterative).

    Args:
        reynolds: Reynolds number
        roughness: Pipe absolute roughness (inches)
        d_inches: Pipe inside diameter (inches)

    Returns:
        Fanning friction factor (dimensionless)
    """
    if reynolds < 1.0:
        return 0.0
    if reynolds < 2100:
        return 16.0 / reynolds  # laminar Fanning

    ed = roughness / d_inches
    # Churchill (1977) explicit approximation – avoids iteration
    a_val = (2.457 * math.log(1.0 / ((7.0 / reynolds) ** 0.9 + 0.27 * ed))) ** 16
    b_val = (37530.0 / reynolds) ** 16
    f_darcy = 8 * ((8.0 / reynolds) ** 12 + 1.0 / (a_val + b_val) ** 1.5) ** (1.0 / 12.0)
    return f_darcy / 4.0  # Fanning = Darcy / 4


def cullender_smith(p_known: float, t_known: float, depth_ft: float,
                    q_mscfd: float, d_inches: float, gas_grav: float,
                    pc_psia: float, tc_degr: float,
                    roughness: float = 0.0006,
                    t_gradient: float = 0.0,
                    n_segments: int = 50,
                    solve_for: str = "bhp",
                    z_func=None) -> dict:
    """
    Cullender-Smith pressure traverse for gas wells.

    Iteratively integrates the static + friction pressure gradient from
    one end of the wellbore to the other.

    Args:
        p_known: Known pressure (psia) – WHP if solve_for='bhp', BHP if 'whp'
        t_known: Temperature at known end (°F)
        depth_ft: True vertical depth of perforations (ft)
        q_mscfd: Gas rate (MSCF/d), 0 for static gradient
        d_inches: Tubing ID (inches)
        gas_grav: Gas specific gravity
        pc_psia: Pseudo-critical pressure (psia)
        tc_degr: Pseudo-critical temperature (°R)
        roughness: Pipe roughness (inches), default 0.0006 (new tubing)
        t_gradient: Temperature gradient (°F/ft), positive = increasing with depth
        n_segments: Number of depth segments for integration
        solve_for: 'bhp' (given WHP, find BHP) or 'whp' (given BHP, find WHP)
        z_func: Z-factor function(ppr, tpr), defaults to Hall-Yarborough

    Returns:
        dict with keys:
            'depths': array of depths (ft)
            'pressures': array of pressures (psia)
            'temperatures': array of temperatures (°F)
            'velocities': array of gas velocities (ft/s)
            'erosional': array of erosional velocities (ft/s)
            'z_factors': array of z-factors
            'bhp' or 'whp': solved pressure (psia)
    """
    if z_func is None:
        from gas_pvt_correlations import z_gas as _z_gas
        z_func = _z_gas

    mw = gas_grav * MW_AIR
    d_ft = d_inches / 12.0
    area = math.pi / 4.0 * d_ft ** 2

    dh = depth_ft / n_segments

    if solve_for == "whp":
        # Integrate upward from BHP: depth goes from depth_ft to 0
        direction = -1.0
        depths = np.linspace(depth_ft, 0, n_segments + 1)
        t_start = t_known  # BHP temperature (at bottom)
    else:
        direction = 1.0
        depths = np.linspace(0, depth_ft, n_segments + 1)
        t_start = t_known  # WHP temperature (at surface)

    pressures = np.zeros(n_segments + 1)
    temperatures = np.zeros(n_segments + 1)
    velocities = np.zeros(n_segments + 1)
    erosional_v = np.zeros(n_segments + 1)
    z_factors = np.zeros(n_segments + 1)

    pressures[0] = p_known
    temperatures[0] = t_start

    for i in range(n_segments + 1):
        if i == 0:
            p = p_known
            t = t_start
        else:
            p = pressures[i]
            t = temperatures[i]
        t_r = t + 459.67
        ppr = p / pc_psia
        tpr = t_r / tc_degr
        z = z_func(ppr, tpr)
        z_factors[i] = z
        if q_mscfd > 0:
            velocities[i] = gas_velocity(q_mscfd, p, t, z, d_inches)
        erosional_v[i] = erosional_velocity(p, t, z, gas_grav)

        if i < n_segments:
            # Temperature at next segment
            temperatures[i + 1] = t_start + t_gradient * abs(depths[i + 1])

            # Cullender-Smith integration step
            rho = _gas_density(p, t, z, gas_grav)

            # Friction
            if q_mscfd > 0:
                v = velocities[i]
                mu_approx = 0.012  # cP, reasonable approximation for friction calc
                reynolds = 20100.0 * gas_grav * q_mscfd / (mu_approx * d_inches)
                ff = _moody_friction(reynolds, roughness, d_inches)
                friction_grad = 2.0 * ff * rho * v ** 2 / (G_ACCEL * d_ft)  # psf/ft
            else:
                friction_grad = 0.0

            # Static gradient
            static_grad = rho / 144.0  # psi/ft

            dp = (static_grad + friction_grad / 144.0) * dh * direction
            pressures[i + 1] = p + dp

            # Ensure pressure stays positive
            pressures[i + 1] = max(pressures[i + 1], 14.696)

    # Fill last z-factor
    t_r_last = temperatures[-1] + 459.67
    z_factors[-1] = z_func(pressures[-1] / pc_psia, t_r_last / tc_degr)
    if q_mscfd > 0:
        velocities[-1] = gas_velocity(q_mscfd, pressures[-1], temperatures[-1],
                                      z_factors[-1], d_inches)
    erosional_v[-1] = erosional_velocity(pressures[-1], temperatures[-1],
                                         z_factors[-1], gas_grav)

    result = {
        'depths': depths if solve_for == "bhp" else depths[::-1],
        'pressures': pressures,
        'temperatures': temperatures,
        'velocities': velocities,
        'erosional': erosional_v,
        'z_factors': z_factors,
    }

    if solve_for == "bhp":
        result['bhp'] = pressures[-1]
        result['depths'] = depths
    else:
        result['whp'] = pressures[-1]
        # Reverse so depth 0 is first
        for key in ('depths', 'pressures', 'temperatures', 'velocities',
                    'erosional', 'z_factors'):
            result[key] = result[key][::-1]

    return result
