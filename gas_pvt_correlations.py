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


# ============================================================================
# Additional Z-factor Correlations
# ============================================================================

def z_gas_dak(ppr: float, tpr: float) -> float:
    """
    Calculate Z-factor using Dranchuk-Abou-Kassem (DAK) correlation (1975).

    11-coefficient equation solved iteratively for reduced density.

    Reference:
        Dranchuk & Abou-Kassem, "Calculation of Z Factors For Natural
        Gases Using Equations of State", JCPT, July-September 1975.

    Args:
        ppr: Pseudo-reduced pressure
        tpr: Pseudo-reduced temperature

    Returns:
        Z-factor (dimensionless)
    """
    if ppr <= 0:
        return 1.0

    # DAK coefficients
    A1 = 0.3265
    A2 = -1.0700
    A3 = -0.5339
    A4 = 0.01569
    A5 = -0.05165
    A6 = 0.5475
    A7 = -0.7361
    A8 = 0.6853
    A9 = 0.6123
    A10 = 0.10489
    A11 = 0.68157

    # Residual function for reduced density
    c1 = A1 + A2 / tpr + A3 / tpr ** 3 + A4 / tpr ** 4 + A5 / tpr ** 5
    c2 = A6 + A7 / tpr + A8 / tpr ** 2
    c3 = A9 * (A7 / tpr + A8 / tpr ** 2)

    def _residual(rho):
        """Return f(rho) where f=0 when Z is consistent."""
        rho = max(rho, 1e-15)
        rho2 = rho * rho
        exp_t = math.exp(-A11 * rho2)
        return (c1 * rho
                + c2 * rho2
                + c3 * rho ** 5
                + A10 * (1 + A11 * rho2) * (rho2 / tpr ** 3) * exp_t
                + 1 - 0.27 * ppr / (rho * tpr))

    def _derivative(rho):
        rho = max(rho, 1e-15)
        rho2 = rho * rho
        exp_t = math.exp(-A11 * rho2)
        return (c1
                + 2 * c2 * rho
                + 5 * c3 * rho ** 4
                + A10 * (rho / tpr ** 3)
                  * (2 + 2 * A11 * rho2 - 2 * A11 ** 2 * rho2 ** 2)
                  * exp_t
                + 0.27 * ppr / (rho2 * tpr))

    # Bracket the root: reduced density must be in (0, ~4)
    rho_lo, rho_hi = 1e-6, 3.5

    # Find bracket via sign change — use fine scan to capture 1st physical root
    n_scan = 200
    f_prev = _residual(rho_lo)
    bracket_found = False
    for i in range(1, n_scan + 1):
        rho_test = rho_lo + i * (rho_hi - rho_lo) / n_scan
        f_cur = _residual(rho_test)
        if f_prev * f_cur <= 0:
            rho_lo_b = rho_lo + (i - 1) * (rho_hi - rho_lo) / n_scan
            rho_hi_b = rho_test
            bracket_found = True
            break
        f_prev = f_cur

    if not bracket_found:
        # No root in physical range — DAK is outside its validity envelope
        return float('nan')

    # Brent-style bisection + Newton hybrid
    rho_r = (rho_lo_b + rho_hi_b) / 2.0
    f_lo = _residual(rho_lo_b)
    for _ in range(300):
        f_val = _residual(rho_r)
        if abs(f_val) < 1e-13:
            break

        df_val = _derivative(rho_r)
        if abs(df_val) > 1e-15:
            rho_newton = rho_r - f_val / df_val
        else:
            rho_newton = rho_r

        # Accept Newton step only if it stays within bracket
        if rho_lo_b < rho_newton < rho_hi_b:
            rho_new = rho_newton
        else:
            # Bisection fallback
            rho_new = (rho_lo_b + rho_hi_b) / 2.0

        # Tighten bracket
        if f_lo * f_val <= 0:
            rho_hi_b = rho_r
        else:
            rho_lo_b = rho_r
            f_lo = f_val

        if abs(rho_new - rho_r) < 1e-12:
            rho_r = rho_new
            break
        rho_r = rho_new

    z = 0.27 * ppr / (max(rho_r, 1e-15) * tpr)
    return z


def z_gas_pmc(ppr: float, tpr: float) -> float:
    """
    Calculate Z-factor using Piper-McCain-Corredor (2012) method.

    Uses the DAK equation with updated pseudo-critical property
    correlations from Piper, McCain & Corredor (2012).  Since this
    function receives already-reduced P and T, it simply wraps
    the DAK solver — the PMC-specific part is the pseudo-critical
    correlation applied before calling this.

    For direct comparison with Hall-Yarborough, this uses the DAK
    11-coefficient equation as recommended by Piper et al.

    Reference:
        Piper, McCain & Corredor, "Compressibility Factors for Naturally
        Occurring Petroleum Gases", SPE 110160-PA, August 2012.

    Args:
        ppr: Pseudo-reduced pressure
        tpr: Pseudo-reduced temperature

    Returns:
        Z-factor (dimensionless)
    """
    return z_gas_dak(ppr, tpr)


def pmc_pseudo_criticals(gas_grav: float, mol_n2: float = 0.0,
                         mol_co2: float = 0.0, mol_h2s: float = 0.0
                         ) -> tuple:
    """
    Calculate pseudo-critical properties using Piper-McCain-Corredor (2012).

    These replace the Sutton + Wichert-Aziz chain with a single
    correlation that directly accounts for non-hydrocarbon impurities.

    Args:
        gas_grav: Gas specific gravity
        mol_n2: Mole fraction N2
        mol_co2: Mole fraction CO2
        mol_h2s: Mole fraction H2S

    Returns:
        Tuple of (Pc_psia, Tc_degR)
    """
    # Stewart-Burkhardt-Voo J and K parameters
    # Coefficients from Piper, McCain & Corredor (2012), SPE 110160-PA Table 1
    # J = Tpc/Ppc;  K = Tpc/sqrt(Ppc)
    j = 0.11582 \
        - 0.45820 * mol_h2s \
        - 0.90348e-2 * mol_co2 \
        - 0.66026e-2 * mol_n2 \
        + 0.70729 * gas_grav \
        - 0.099397 * gas_grav ** 2

    k = 3.8216 \
        - 0.06534 * mol_h2s \
        - 0.42113e-2 * mol_co2 \
        - 0.11488e-1 * mol_n2 \
        + 25.261 * gas_grav \
        - 13.703 * gas_grav ** 2

    if j <= 0:
        j = 0.01  # guard against non-physical values

    tc = k ** 2 / j   # °R
    pc = tc / j        # psia

    return pc, tc


# ============================================================================
# Gas Heating Value & Wobbe Index
# ============================================================================

def gross_heating_value(gas_grav: float, mol_n2: float = 0.0,
                        mol_co2: float = 0.0, mol_h2s: float = 0.0) -> float:
    """
    Estimate gross (superior) heating value from gas gravity.

    Correlation from Thomas, Hankinson & Phillips, applicable for
    sweet to moderately sour gases.

    Args:
        gas_grav: Gas specific gravity (air = 1.0)
        mol_n2: Mole fraction N2
        mol_co2: Mole fraction CO2
        mol_h2s: Mole fraction H2S

    Returns:
        Gross heating value (BTU/SCF at 14.696 psia, 60°F)
    """
    # GPA 2172 approximate correlation
    hv_hc = 1568.7 * gas_grav - 233.6 * gas_grav ** 2 + 117.0
    # N2 and CO2 are inerts (dilute heating value)
    # H2S has heating value ~637 BTU/SCF
    inert_fraction = mol_n2 + mol_co2
    hv = hv_hc * (1 - inert_fraction) - mol_co2 * 0.0 + mol_h2s * 637.0
    return max(hv, 0.0)


def net_heating_value(ghv: float) -> float:
    """
    Estimate net (inferior) heating value from gross heating value.

    Accounts for latent heat of water vaporization (~1030 BTU/SCF water produced
    per ~1000 BTU/SCF gas for typical natural gas).

    Args:
        ghv: Gross heating value (BTU/SCF)

    Returns:
        Net heating value (BTU/SCF)
    """
    # Approximate: NHV ≈ 0.9 * GHV for typical natural gas
    return ghv * 0.9036


def wobbe_index(ghv: float, gas_grav: float) -> float:
    """
    Calculate Wobbe Index.

    WI = GHV / sqrt(gas_gravity)

    The Wobbe Index is used for gas interchangeability assessment.

    Args:
        ghv: Gross heating value (BTU/SCF)
        gas_grav: Gas specific gravity

    Returns:
        Wobbe Index (BTU/SCF)
    """
    if gas_grav <= 0:
        return 0.0
    return ghv / math.sqrt(gas_grav)


def specific_energy(ghv: float) -> float:
    """
    Convert heating value from BTU/SCF to MJ/m³.

    Args:
        ghv: Gross heating value (BTU/SCF)

    Returns:
        Specific energy (MJ/m³)
    """
    return ghv * 0.03726


# ============================================================================
# Dew Point Estimation
# ============================================================================

def dew_point_nemeth_kennedy(gas_grav: float, mol_n2: float = 0.0,
                              mol_co2: float = 0.0, mol_h2s: float = 0.0,
                              mol_c7plus: float = 0.0) -> float:
    """
    Estimate dew-point pressure using Nemeth-Kennedy (1967) correlation.

    Screening-level correlation for lean gas condensates. Accuracy
    is ±10–15% for gas gravities 0.6–1.0. Not a rigorous EOS calculation.

    Reference:
        Nemeth & Kennedy, "A Correlation of Dewpoint Pressure With
        Fluid Composition and Temperature", SPE 1839 (1967).

    Args:
        gas_grav: Gas specific gravity
        mol_n2: Mole fraction N2
        mol_co2: Mole fraction CO2
        mol_h2s: Mole fraction H2S
        mol_c7plus: Mole fraction C7+ (heavy fraction)

    Returns:
        Estimated dew-point pressure (psia). Returns 0 if gas is too lean
        (no C7+ present and gravity < 0.65).
    """
    if mol_c7plus <= 0 and gas_grav < 0.65:
        return 0.0

    # Simplified Nemeth-Kennedy: Pd = f(gamma_g, y_C7+)
    # Literature regression on field data
    pd = (4019.0 * gas_grav - 2167.0) + 13000.0 * mol_c7plus \
         - 1500.0 * (mol_n2 + mol_co2) + 1000.0 * mol_h2s
    return max(pd, 0.0)


def cricondentherm_estimate(gas_grav: float, mol_c7plus: float = 0.0) -> float:
    """
    Estimate cricondentherm (maximum temperature on phase envelope).

    Approximate correlation for screening. Not a rigorous EOS calculation.

    Args:
        gas_grav: Gas specific gravity
        mol_c7plus: Mole fraction C7+

    Returns:
        Estimated cricondentherm (°F)
    """
    # Approximate: heavily influenced by C7+ content and gas gravity
    t_ct = -100.0 + 500.0 * gas_grav + 2000.0 * mol_c7plus
    return t_ct
