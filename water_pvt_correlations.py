"""
Water PVT Correlations
Re-implementation of VBA functions for water properties

References:
- SPE Phase Behavior Monograph Vol 20 by Whitson & Brule
- "The Properties of Petroleum Fluids - Second Edition" by William McCain, Jr
- "PVT and Phase Behaviour Petroleum Reservoir Fluids" by Ali Danesh
"""

import math
from typing import Optional


def wtr_in_gas_bbl_per_mmcf(p_psia: float, t_degf: float, tds_ppm: float,
                             gas_grav: float) -> float:
    """
    Calculate water vapor content in gas.
    
    Args:
        p_psia: Pressure in psia
        t_degf: Temperature in deg F
        tds_ppm: Total dissolved solids in ppm
        gas_grav: Gas specific gravity (air = 1.0)
        
    Returns:
        Water content in bbl/MMCF
    """
    psc = 14.696
    tsc = 60.0
    zsc = 1.0
    
    # Vapor pressure of water
    pvp_h2o = math.exp(69.10351 - 13064.76 / (t_degf + 459.6)
                       - 7.3037 * math.log(t_degf + 459.6)
                       + 0.0000012856 * (t_degf + 459.6) ** 2)
    
    a = pvp_h2o * 18000000.0 * psc / 10.73 / (459.6 + tsc) / zsc
    b = 10 ** (-3083.87 / (459.6 + t_degf) + 6.69449)
    w = a / p_psia + b
    
    # Gas gravity correction
    ag = 1 + (gas_grav - 0.55) / (15500.0 * gas_grav * t_degf ** -1.446
                                    - 18300.0 * t_degf ** -1.288)
    
    # Salinity correction
    acs = 1 - 0.00000000392 * tds_ppm ** 1.44
    
    w = w * ag * acs
    
    # Convert to bbl/MMCF
    wtr_content = w / 350.506987
    
    return wtr_content


def gas_in_wtr_scf_per_stb(p_psia: float, t_degf: float, tds_ppm: float,
                            tb_degr: float = 200.988) -> float:
    """
    Calculate gas solubility in water.
    
    Args:
        p_psia: Pressure in psia
        t_degf: Temperature in deg F
        tds_ppm: Total dissolved solids in ppm
        tb_degr: Normal boiling point in deg R (default for methane)
        
    Returns:
        Gas solubility in SCF/STB
    """
    # Correlation coefficients
    a = [
        [0.299, -0.001273, 0.0, 0.0],
        [0.002283, -0.0000187, 0.00000007494, -0.00000000007881],
        [-0.000000285, 0.00000000272, -0.00000000001123, 1.361e-14],
        [0.00000000001181, -1.082e-13, 4.275e-16, -4.846e-19]
    ]
    
    tb_degk = tb_degr * 5.0 / 9.0
    
    # Molality of salt concentration
    if tds_ppm == 0:
        csw = 0.0
    else:
        csw = 17.1107565969194 / (1000000.0 / tds_ppm + 1)
    
    # Solubility of methane in pure water
    xc1 = 0.0
    for i in range(4):
        for j in range(4):
            xc1 += a[i][j] * t_degf ** j * p_psia ** i
    
    xc1 *= 0.001
    rsw = 7370 * xc1 / (1 - xc1)
    
    # Setchenow constant for methane
    ks = (0.1813 - 0.0007692 * t_degf + 0.0000026614 * t_degf ** 2
          - 0.000000002612 * t_degf ** 3)
    
    # Correction for gas other than methane
    ks += 0.000445 * (tb_degk - 111.66)
    
    # Solubility in salt water
    rsw *= 10 ** (-ks * csw)
    
    return rsw


def wtr_sp_vol_psc(t_degf: float, tds_ppm: float) -> float:
    """
    Calculate water specific volume at standard pressure.
    
    Args:
        t_degf: Temperature in deg F
        tds_ppm: Total dissolved solids in ppm
        
    Returns:
        Specific volume in cm^3/g
    """
    t_degk = (t_degf + 459.67) * 5.0 / 9.0
    ws = tds_ppm / 1000000.0
    
    a0 = (5.91635 - 0.01035794 * t_degk + 0.000009270048 * t_degk ** 2
          - 1127.522 / t_degk + 100674.1 / t_degk ** 2)
    a1 = -2.5166 + 0.0111766 * t_degk - 0.0000170552 * t_degk ** 2
    a2 = 2.84851 - 0.0154305 * t_degk + 0.0000223982 * t_degk ** 2
    
    sp_vol = a0 + a1 * ws + a2 * ws ** 2
    
    return sp_vol


def wtr_fvf(p_psia: float, t_degf: float, tds_ppm: float,
            tb_degr: float = 200.988) -> float:
    """
    Calculate water formation volume factor.
    
    Args:
        p_psia: Pressure in psia
        t_degf: Temperature in deg F
        tds_ppm: Total dissolved solids in ppm
        tb_degr: Normal boiling point in deg R
        
    Returns:
        Water FVF in RB/STB
    """
    psc_psia = 14.696
    tsc_degf = 60.0
    ws = tds_ppm / 1000000.0
    
    # Bw at standard pressure and reservoir temperature
    bw = wtr_sp_vol_psc(t_degf, tds_ppm) / wtr_sp_vol_psc(tsc_degf, tds_ppm)
    
    # Pressure correction
    a0 = (0.314 + 0.58 * ws + 0.00019 * t_degf) * 1000000.0
    a1 = 8 + 50 * ws - 0.125 * ws * t_degf
    
    bw *= (1 + a1 / a0 * p_psia) ** (-1 / a1)
    
    # Dissolved gas correction
    rsw = gas_in_wtr_scf_per_stb(p_psia, t_degf, tds_ppm, tb_degr)
    bw *= (1 + 0.0001 * rsw ** 1.5)
    
    return bw


def wtr_grad(p_psia: float, t_degf: float, tds_ppm: float,
             gas_grav: float = 0.55378, tb_degr: float = 200.988) -> float:
    """
    Calculate water gradient.
    
    Args:
        p_psia: Pressure in psia
        t_degf: Temperature in deg F
        tds_ppm: Total dissolved solids in ppm
        gas_grav: Gas specific gravity (air = 1.0)
        tb_degr: Normal boiling point in deg R
        
    Returns:
        Water gradient in psi/ft
    """
    psc_psia = 14.696
    tsc_degf = 60.0
    
    # Mass of gas in solution (lbm/STB)
    rsw = gas_in_wtr_scf_per_stb(p_psia, t_degf, tds_ppm, tb_degr)
    lbm_gas = rsw * psc_psia * gas_grav * 28.97 / 10.7315 / (tsc_degf + 459.67)
    
    # Calculate gradient
    fvf = wtr_fvf(p_psia, t_degf, tds_ppm, tb_degr)
    grad = ((0.433527504001004 / wtr_sp_vol_psc(tsc_degf, tds_ppm)
             + lbm_gas / 808.5) / fvf)
    
    return grad


def mu_wtr(p_psia: float, t_degf: float, tds_ppm: float) -> float:
    """
    Calculate water viscosity.
    
    Args:
        p_psia: Pressure in psia
        t_degf: Temperature in deg F
        tds_ppm: Total dissolved solids in ppm
        
    Returns:
        Water viscosity in cP
    """
    # Correlation constants
    a = [
        [0.0, 0.03324, 0.003624, -0.0001879],
        [0.0, -0.0396, 0.0102, -0.000702],
        [0.0, 1.2378, -0.001303, 0.00000306, 0.0000000255]
    ]
    
    # Unit conversions
    t_degc = (t_degf - 32) * 5.0 / 9.0
    p_mpa = 6.89475729316836e-3 * p_psia
    
    # Molality of salt concentration
    if tds_ppm == 0:
        csw = 0.0
    else:
        csw = 17.1107565969194 / (1000000.0 / tds_ppm + 1)
    
    # Viscosity of pure water at 20 deg C
    mu0w20 = 1.002
    
    # Viscosity of pure water at reservoir temperature
    log_mu0w_mu0w20 = sum(a[2][i] * (20 - t_degc) ** i / (96 + t_degc)
                          for i in range(1, 5))
    mu0w = mu0w20 * 10 ** log_mu0w_mu0w20
    
    # Viscosity of salt water at reservoir temperature
    a1 = sum(a[0][i] * csw ** i for i in range(1, 4))
    a2 = sum(a[1][i] * csw ** i for i in range(1, 4))
    mu_w_ast = mu0w * 10 ** (a1 + a2 * log_mu0w_mu0w20)
    
    # Pressure correction
    a0 = (0.8 + 0.01 * (t_degc - 90) * math.exp(-0.25 * csw)) * 0.001
    mu_w = (1 + a0 * p_mpa) * mu_w_ast
    
    return mu_w


def c_wtr(p_psia: float, t_degf: float, tds_ppm: float, psc_psia: float,
          tsc_degf: float, gas_grav: float = 0.55378,
          tb_degr: float = 200.988) -> float:
    """
    Calculate water compressibility.
    
    Args:
        p_psia: Pressure in psia
        t_degf: Temperature in deg F
        tds_ppm: Total dissolved solids in ppm
        psc_psia: Standard pressure in psia
        tsc_degf: Standard temperature in deg F
        gas_grav: Gas specific gravity (air = 1.0)
        tb_degr: Normal boiling point in deg R
        
    Returns:
        Water compressibility in 1/psi
    """
    # Use numerical derivative of FVF with respect to pressure
    dp = 1.0  # Small pressure increment
    
    bw1 = wtr_fvf(p_psia - dp/2, t_degf, tds_ppm, tb_degr)
    bw2 = wtr_fvf(p_psia + dp/2, t_degf, tds_ppm, tb_degr)
    
    dbw_dp = (bw2 - bw1) / dp
    bw = wtr_fvf(p_psia, t_degf, tds_ppm, tb_degr)
    
    cw = dbw_dp / bw
    
    return cw
