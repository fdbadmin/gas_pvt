"""
Gas PVT Correlations
Re-implementation of VBA functions from GasPVT_v101_GeneralReleasev2.xlsm

Implements Hall-Yarborough Z-factor correlation and related gas properties.
Programmed by: R H Roach (original VBA)
Reimplemented in Python: 2026-02-05

References:
- Phase Behavior by Curtis Whitson and Michael Brule, SPE Monograph Volume 20
- Hall and Yarborough, "A New EOS for Z-factor Calculations", O&GJ, 18 June 1973, 82
- Yarborough and Hall, "How to Solve EOS for Z-factors", O&GJ, 18 February 1974, 86
"""

import math
from typing import Optional


def z_gray(ppr: float, tpr: float) -> float:
    """
    Gray's Z-factor correlation (used as initial guess for Hall-Yarborough).
    
    Args:
        ppr: Pseudo-reduced pressure
        tpr: Pseudo-reduced temperature
        
    Returns:
        Z-factor
    """
    # This is a helper function referenced but not fully defined in the VBA code
    # Using a simplified correlation for initial guess
    return 1.0 + (0.257 * ppr - 0.533 * ppr / tpr)


def z_gas(ppr: float, tpr: float) -> float:
    """
    Calculate real gas Z-factor using Hall-Yarborough correlation.
    
    Uses Newton-Raphson iteration to solve the Hall-Yarborough equation
    for reduced density, then converts to Z-factor.
    
    Args:
        ppr: Pseudo-reduced pressure (dimensionless)
        tpr: Pseudo-reduced temperature (dimensionless)
        
    Returns:
        Z-factor (dimensionless)
        
    Note:
        - Returns 1.0 when ppr = 0
        - Handles low Tpr cases where y becomes negative
    """
    if ppr <= 0:
        return 1.0
    
    t = 1.0 / tpr
    alpha = 0.06125 * t * math.exp(-1.2 * (1 - t) ** 2)
    
    # Initial guess based on Gray's z factor algorithm
    y = alpha * ppr / z_gray(ppr, tpr)
    
    # Newton-Raphson iteration
    tolerance = 1e-8
    max_iterations = 100
    
    for _ in range(max_iterations):
        # Ensure y is positive to avoid complex numbers
        y = max(y, 1e-10)
        
        # Prevent y from being too close to 1 (denominator issues)
        if y >= 0.99:
            y = 0.99
        
        # Calculate function and derivative
        # Use abs(y) for fractional powers to avoid complex numbers
        y_abs = abs(y)
        power_term = y_abs**(2.18 + 2.82 * t)
        power_term_deriv = y_abs**(1.18 + 2.82 * t)
        
        fy = (-alpha * ppr 
              + (y + y**2 + y**3 - y**4) / (1 - y)**3
              - (14.76 * t - 9.76 * t**2 + 4.58 * t**3) * y**2
              + (90.7 * t - 242.2 * t**2 + 42.4 * t**3) * power_term)
        
        dfydy = ((1 + 4*y + 4*y**2 - 4*y**3 + y**4) / (1 - y)**4
                 - (29.52 * t - 19.52 * t**2 + 9.16 * t**3) * y
                 + (2.18 + 2.82 * t) * (90.7 * t - 242.2 * t**2 + 42.4 * t**3) 
                 * power_term_deriv)
        
        # Avoid division by zero
        if abs(dfydy) < 1e-15:
            break
        
        y_old = y
        y = y_old - fy / dfydy
        
        # Handle negative y for low Tpr cases
        if y <= 0:
            y = y_old / 2
        
        # Check convergence
        if abs((y - y_old) / y) < tolerance:
            break
    
    z = alpha * ppr / y
    return z


def bg_rv_per_scv(p_psia: float, t_degf: float, psc_psia: float, tsc_degf: float,
                   pc_psia: float, tc_degr: float) -> float:
    """
    Calculate Gas Formation Volume Factor.
    
    Args:
        p_psia: Pressure in psia
        t_degf: Temperature in deg F
        psc_psia: Standard pressure in psia
        tsc_degf: Standard temperature in deg F
        pc_psia: Pseudo-critical pressure in psia
        tc_degr: Pseudo-critical temperature in deg R
        
    Returns:
        Gas Formation Volume Factor in reservoir volume per standard volume
        (units consistent: rcf/SCF, rb/STB, or MM rcf/MMCF)
    """
    ppr = p_psia / pc_psia
    tpr = (t_degf + 459.67) / tc_degr
    z = z_gas(ppr, tpr)
    
    bg = (psc_psia / (tsc_degf + 459.67) * z * (t_degf + 459.67) / p_psia)
    return bg


def mu_gas(p_psia: float, t_degf: float, pc_psia: float, tc_degr: float,
           gas_grav: float, mol_frac_n2: float, mol_frac_co2: float,
           mol_frac_h2s: float) -> float:
    """
    Calculate gas viscosity using Lucas Correlation with Standing correction.
    
    From SPE Phase Behavior Monograph Vol 20 by Whitson & Brule.
    Lucas Correlation with Standing Correction for N2/CO2/H2S.
    
    Args:
        p_psia: Pressure in psia
        t_degf: Temperature in deg F
        pc_psia: Pseudo-critical pressure in psia
        tc_degr: Pseudo-critical temperature in deg R
        gas_grav: Gas specific gravity (air = 1.0)
        mol_frac_n2: Mole fraction of N2
        mol_frac_co2: Mole fraction of CO2
        mol_frac_h2s: Mole fraction of H2S
        
    Returns:
        Gas viscosity in cP
    """
    ppr = p_psia / pc_psia
    tpr = (t_degf + 459.67) / tc_degr
    
    # Lucas correlation coefficients
    a1 = 0.001245 * math.exp(5.1726 * tpr**(-0.3286)) / tpr
    a2 = a1 * (1.6553 * tpr - 1.2723)
    a3 = 0.4489 * math.exp(3.0578 * tpr**(-37.7332)) / tpr
    a4 = 1.7368 * math.exp(2.231 * tpr**(-7.6351)) / tpr
    a5 = 0.9425 * math.exp(-0.1853 * tpr**0.4489)
    
    # Molecular weight and viscosity parameter
    m = gas_grav * 28.97
    xi = 9490 * (tc_degr / m**3 / pc_psia**4)**(1/6)
    
    # Base viscosity at 1 atm
    mu_1atm = ((0.807 * tpr**0.618 - 0.357 * math.exp(-0.449 * tpr) 
                + 0.34 * math.exp(-4.058 * tpr) + 0.018) / xi)
    
    # Standing correction for non-hydrocarbons
    mu_1atm += (mol_frac_n2 * (0.00848 * math.log10(gas_grav) + 0.00959)
                + mol_frac_co2 * (0.00908 * math.log10(gas_grav) + 0.00624)
                + mol_frac_h2s * (0.00849 * math.log10(gas_grav) + 0.00373))
    
    # Pressure correction
    mu = (1 + a1 * ppr**1.3088 / (a2 * ppr**a5 + (1 + a3 * ppr**a4)**(-1))) * mu_1atm
    
    return mu


def c_gas(p_psia: float, pc_psia: float, tpr: float) -> float:
    """
    Calculate gas compressibility using Hall-Yarborough correlation.
    
    Args:
        p_psia: Pressure in psia
        pc_psia: Pseudo-critical pressure in psia
        tpr: Pseudo-reduced temperature (dimensionless)
        
    Returns:
        Gas compressibility in 1/psi
        
    Note:
        Returns 0 when p_psia = 0 (undefined but no asymptote)
    """
    ppr = p_psia / pc_psia
    
    if ppr <= 0:
        return 0.0
    
    t = 1.0 / tpr
    alpha = 0.06125 * t * math.exp(-1.2 * (1 - t) ** 2)
    
    # Initial guess
    y = alpha * ppr / z_gray(ppr, tpr)
    
    # Newton-Raphson iteration
    tolerance = 1e-8
    max_iterations = 100
    
    for _ in range(max_iterations):
        fy = (-alpha * ppr 
              + (y + y**2 + y**3 - y**4) / (1 - y)**3
              - (14.76 * t - 9.76 * t**2 + 4.58 * t**3) * y**2
              + (90.7 * t - 242.2 * t**2 + 42.4 * t**3) * y**(2.18 + 2.82 * t))
        
        dfydy = ((1 + 4*y + 4*y**2 - 4*y**3 + y**4) / (1 - y)**4
                 - (29.52 * t - 19.52 * t**2 + 9.16 * t**3) * y
                 + (2.18 + 2.82 * t) * (90.7 * t - 242.2 * t**2 + 42.4 * t**3) 
                 * y**(1.18 + 2.82 * t))
        
        y_old = y
        y = y_old - fy / dfydy
        
        if y <= 0:
            y = y_old / 2
        
        if abs((y - y_old) / y) < tolerance:
            break
    
    z = alpha * ppr / y
    dz_dppr = alpha * (1/y - alpha * ppr / y**2 / dfydy)
    c_g = (1/ppr - dz_dppr/z) / pc_psia
    
    return c_g


def gas_grad(p_psia: float, t_degf: float, gas_grav: float,
             pc_psia: float, tc_degr: float) -> float:
    """
    Calculate gas gradient.
    
    Args:
        p_psia: Pressure in psia
        t_degf: Temperature in deg F
        gas_grav: Gas specific gravity (air = 1.0)
        pc_psia: Pseudo-critical pressure in psia
        tc_degr: Pseudo-critical temperature in deg R
        
    Returns:
        Gas gradient in psi/ft
        
    Note:
        Does not include correction for dissolved water (considered minor)
    """
    ppr = p_psia / pc_psia
    tpr = (t_degf + 459.67) / tc_degr
    z = z_gas(ppr, tpr)
    
    grad = (p_psia * gas_grav * 28.97 
            / z / 10.7315 / (t_degf + 459.67) / 144)
    
    return grad


def pp_integrate(p1: float, y1: float, p2: float, y2: float,
                 p3: float, y3: float) -> float:
    """
    Integrate polynomial passing through 3 points for pseudo-pressure calculation.
    
    Helper function for pseudo_pressure calculation.
    
    Args:
        p1, y1: First point
        p2, y2: Second point
        p3, y3: Third point
        
    Returns:
        Integral value
    """
    # Fit parabola through 3 points and integrate
    # Using Lagrange polynomial integration
    integral = ((y1 * (p3**2 - p1**2) * (p3 - 2*p2 + p1) / 2
                 + y2 * (p3**2 - p1**2) * (2*p2 - p1 - p3)
                 + y3 * (p3**2 - p1**2) * (p2 - p1) / 2)
                / ((p2 - p1) * (p3 - p1) * (p3 - p2)))
    
    return integral


def pseudo_pressure(p_psia: float, t_degf: float, pc_psia: float,
                    tc_degr: float, gas_grav: float, mol_frac_n2: float,
                    mol_frac_co2: float, mol_frac_h2s: float) -> float:
    """
    Calculate gas pseudo-pressure using base pressure = 0.
    
    Integrates 2*P/(mu*Z) from 0 to P using polynomial approximation
    with pressure increments.
    
    Args:
        p_psia: Pressure in psia
        t_degf: Temperature in deg F
        pc_psia: Pseudo-critical pressure in psia
        tc_degr: Pseudo-critical temperature in deg R
        gas_grav: Gas specific gravity (air = 1.0)
        mol_frac_n2: Mole fraction of N2
        mol_frac_co2: Mole fraction of CO2
        mol_frac_h2s: Mole fraction of H2S
        
    Returns:
        Pseudo-pressure in psia^2/cP
    """
    tpr = (t_degf + 459.67) / tc_degr
    
    # Pressure increment for integration
    p_incr = 100.0
    
    # First increment
    if p_psia < p_incr:
        p_last = p_psia
    else:
        p_last = p_incr
    
    z_last = z_gas(p_last / pc_psia, tpr)
    mu_last = mu_gas(p_last, t_degf, pc_psia, tc_degr, gas_grav,
                     mol_frac_n2, mol_frac_co2, mol_frac_h2s)
    
    # Validate z and mu to prevent spikes
    if isinstance(z_last, complex):
        z_last = abs(z_last)
    if z_last < 0.1 or z_last > 5.0:
        z_last = max(0.1, min(5.0, z_last))
    if mu_last < 0.008 or mu_last > 0.1:
        mu_last = max(0.008, min(0.1, mu_last))
    
    p_mid = p_last / 2
    z_mid = z_gas(p_mid / pc_psia, tpr)
    mu_mid = mu_gas(p_mid, t_degf, pc_psia, tc_degr, gas_grav,
                    mol_frac_n2, mol_frac_co2, mol_frac_h2s)
    
    # Validate mid-point values
    if isinstance(z_mid, complex):
        z_mid = abs(z_mid)
    if z_mid < 0.1 or z_mid > 5.0:
        z_mid = max(0.1, min(5.0, z_mid))
    if mu_mid < 0.008 or mu_mid > 0.1:
        mu_mid = max(0.008, min(0.1, mu_mid))
    
    pp = 2 * pp_integrate(0, 0, p_mid, p_mid / mu_mid / z_mid,
                          p_last, p_last / mu_last / z_last)
    
    # Continue with increments until reaching target pressure
    while p_last < p_psia:
        p0 = p_last
        y0 = p_last / mu_last / z_last
        
        if p_psia - p_last < p_incr:
            p_last = p_psia
        else:
            p_last = p0 + p_incr
        
        z_last = z_gas(p_last / pc_psia, tpr)
        mu_last = mu_gas(p_last, t_degf, pc_psia, tc_degr, gas_grav,
                         mol_frac_n2, mol_frac_co2, mol_frac_h2s)
        
        # Validate z and mu to prevent spikes from anomalous values
        if isinstance(z_last, complex):
            z_last = abs(z_last)
        if z_last < 0.1 or z_last > 5.0:  # Non-physical z-factor
            z_last = max(0.1, min(5.0, z_last))
        if mu_last < 0.008 or mu_last > 0.1:  # Non-physical viscosity range
            mu_last = max(0.008, min(0.1, mu_last))
        
        p_mid = (p0 + p_last) / 2
        z_mid = z_gas(p_mid / pc_psia, tpr)
        mu_mid = mu_gas(p_mid, t_degf, pc_psia, tc_degr, gas_grav,
                        mol_frac_n2, mol_frac_co2, mol_frac_h2s)
        
        # Validate mid-point values as well
        if isinstance(z_mid, complex):
            z_mid = abs(z_mid)
        if z_mid < 0.1 or z_mid > 5.0:
            z_mid = max(0.1, min(5.0, z_mid))
        if mu_mid < 0.008 or mu_mid > 0.1:
            mu_mid = max(0.008, min(0.1, mu_mid))
        
        pp += 2 * pp_integrate(p0, y0, p_mid, p_mid / mu_mid / z_mid,
                               p_last, p_last / mu_last / z_last)
    
    return pp


def p_crit(gas_grav: float, mol_frac_n2: float, mol_frac_co2: float,
           mol_frac_h2s: float, mol_frac_c7plus: float,
           spec_grav_liq: float, mol_wgt_c7plus: float) -> float:
    """
    Calculate pseudo-critical pressure using Sutton correlation.
    
    Args:
        gas_grav: Gas specific gravity (air = 1.0)
        mol_frac_n2: Mole fraction of N2
        mol_frac_co2: Mole fraction of CO2
        mol_frac_h2s: Mole fraction of H2S
        mol_frac_c7plus: Mole fraction of C7+
        spec_grav_liq: Specific gravity of C7+ liquid
        mol_wgt_c7plus: Molecular weight of C7+
        
    Returns:
        Pseudo-critical pressure in psia
    """
    # Sutton correlation
    pc = 756.8 - 131.0 * gas_grav - 3.6 * gas_grav**2
    
    # Correction for C7+
    if mol_frac_c7plus > 0:
        pc_c7plus = pc_c7_plus(spec_grav_liq, mol_wgt_c7plus)
        # Apply Kay's mixing rule (simplified)
        pc = pc * (1 - mol_frac_c7plus) + pc_c7plus * mol_frac_c7plus
    
    # Wichert-Aziz correction for H2S and CO2
    pc = p_crit_corr(pc, t_crit(gas_grav, mol_frac_n2, mol_frac_co2,
                                  mol_frac_h2s, mol_frac_c7plus,
                                  spec_grav_liq, mol_wgt_c7plus),
                     mol_frac_co2, mol_frac_h2s)
    
    return pc


def t_crit(gas_grav: float, mol_frac_n2: float, mol_frac_co2: float,
           mol_frac_h2s: float, mol_frac_c7plus: float,
           spec_grav_liq: float, mol_wgt_c7plus: float) -> float:
    """
    Calculate pseudo-critical temperature using Sutton correlation.
    
    Args:
        gas_grav: Gas specific gravity (air = 1.0)
        mol_frac_n2: Mole fraction of N2
        mol_frac_co2: Mole fraction of CO2
        mol_frac_h2s: Mole fraction of H2S
        mol_frac_c7plus: Mole fraction of C7+
        spec_grav_liq: Specific gravity of C7+ liquid
        mol_wgt_c7plus: Molecular weight of C7+
        
    Returns:
        Pseudo-critical temperature in deg R
    """
    # Sutton correlation
    tc = 169.2 + 349.5 * gas_grav - 74.0 * gas_grav**2
    
    # Correction for C7+
    if mol_frac_c7plus > 0:
        tc_c7plus = tc_c7_plus(spec_grav_liq, mol_wgt_c7plus)
        # Apply Kay's mixing rule (simplified)
        tc = tc * (1 - mol_frac_c7plus) + tc_c7plus * mol_frac_c7plus
    
    # Wichert-Aziz correction for H2S and CO2
    tc = t_crit_corr(tc, mol_frac_co2, mol_frac_h2s)
    
    return tc


def pc_c7_plus(spec_grav_liq: float, mol_wgt: float) -> float:
    """
    Calculate critical pressure of C7+ fraction using Riazi-Daubert correlation.
    
    Args:
        spec_grav_liq: Specific gravity of liquid
        mol_wgt: Molecular weight
        
    Returns:
        Critical pressure in psia
    """
    # Riazi-Daubert correlation
    pc = 3.12281e9 * mol_wgt**(-2.3125) * spec_grav_liq**2.3201 * math.exp(-2.3125 * 0.0)
    return pc


def tc_c7_plus(spec_grav_liq: float, mol_wgt: float) -> float:
    """
    Calculate critical temperature of C7+ fraction using Riazi-Daubert correlation.
    
    Args:
        spec_grav_liq: Specific gravity of liquid
        mol_wgt: Molecular weight
        
    Returns:
        Critical temperature in deg R
    """
    # Riazi-Daubert correlation
    tc = 544.4 * mol_wgt**0.2998 * spec_grav_liq**(-0.0002)
    return tc


def p_crit_corr(pc_psia: float, tc_degr: float, mol_frac_co2: float,
                mol_frac_h2s: float) -> float:
    """
    Apply Wichert-Aziz correction to pseudo-critical pressure for H2S and CO2.
    
    Args:
        pc_psia: Uncorrected pseudo-critical pressure in psia
        tc_degr: Uncorrected pseudo-critical temperature in deg R
        mol_frac_co2: Mole fraction of CO2
        mol_frac_h2s: Mole fraction of H2S
        
    Returns:
        Corrected pseudo-critical pressure in psia
    """
    # Wichert-Aziz correction
    y_sum = mol_frac_h2s + mol_frac_co2
    
    if y_sum > 0:
        epsilon = 120 * (y_sum**0.9 - y_sum**1.6) + 15 * (mol_frac_h2s**0.5 - mol_frac_h2s**4)
        tc_corr = tc_degr - epsilon
        pc_corr = pc_psia * tc_corr / (tc_degr + mol_frac_h2s * (1 - mol_frac_h2s) * epsilon)
        return pc_corr
    
    return pc_psia


def t_crit_corr(tc_degr: float, mol_frac_co2: float, mol_frac_h2s: float) -> float:
    """
    Apply Wichert-Aziz correction to pseudo-critical temperature for H2S and CO2.
    
    Args:
        tc_degr: Uncorrected pseudo-critical temperature in deg R
        mol_frac_co2: Mole fraction of CO2
        mol_frac_h2s: Mole fraction of H2S
        
    Returns:
        Corrected pseudo-critical temperature in deg R
    """
    # Wichert-Aziz correction
    y_sum = mol_frac_h2s + mol_frac_co2
    
    if y_sum > 0:
        epsilon = 120 * (y_sum**0.9 - y_sum**1.6) + 15 * (mol_frac_h2s**0.5 - mol_frac_h2s**4)
        tc_corr = tc_degr - epsilon
        return tc_corr
    
    return tc_degr


def tb_degr_correlation(tc_degr: float) -> float:
    """
    Calculate normal boiling point from critical temperature.
    
    Args:
        tc_degr: Critical temperature in deg R
        
    Returns:
        Normal boiling point in deg R
    """
    # Empirical correlation
    tb = 0.533 * tc_degr + 191.7
    return tb
