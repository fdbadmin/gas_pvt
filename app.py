"""
Gas PVT Analysis - Streamlit Web Application

Interactive web interface for performing Gas and Water PVT calculations
using industry-standard correlations.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats

from gas_pvt_correlations import (
    z_gas, bg_rv_per_scv, mu_gas, c_gas, gas_grad, pseudo_pressure,
    p_crit, t_crit, pc_c7_plus, tc_c7_plus, p_crit_corr, t_crit_corr,
    tb_degr_correlation
)
from water_pvt_correlations import (
    wtr_in_gas_bbl_per_mmcf, gas_in_wtr_scf_per_stb, wtr_fvf,
    wtr_grad, mu_wtr, c_wtr
)
from uncertainty_enhanced import (
    generate_samples, calculate_tornado_data, create_box_plots,
    create_tornado_chart, create_distribution_plot_with_cdf
)


# ============================================================================
#  UNIT CONVERSION FUNCTIONS
# ============================================================================

class UnitConverter:
    """Handle unit conversions for PVT calculations."""
    
    # Temperature conversions
    @staticmethod
    def celsius_to_fahrenheit(temp_c):
        """Convert Celsius to Fahrenheit."""
        return temp_c * 9/5 + 32
    
    @staticmethod
    def fahrenheit_to_celsius(temp_f):
        """Convert Fahrenheit to Celsius."""
        return (temp_f - 32) * 5/9
    
    @staticmethod
    def rankine_to_fahrenheit(temp_r):
        """Convert Rankine to Fahrenheit."""
        return temp_r - 459.67
    
    @staticmethod
    def rankine_to_celsius(temp_r):
        """Convert Rankine to Celsius."""
        return (temp_r - 491.67) * 5/9
    
    # Pressure conversions (all to/from psia)
    @staticmethod
    def mpa_to_psia(pressure_mpa):
        """Convert MPa to psia."""
        return pressure_mpa * 145.038
    
    @staticmethod
    def psia_to_mpa(pressure_psia):
        """Convert psia to MPa."""
        return pressure_psia / 145.038
    
    @staticmethod
    def kpa_to_psia(pressure_kpa):
        """Convert kPa to psia."""
        return pressure_kpa * 0.145038
    
    @staticmethod
    def psia_to_kpa(pressure_psia):
        """Convert psia to kPa."""
        return pressure_psia / 0.145038
    
    @staticmethod
    def bar_to_psia(pressure_bar):
        """Convert bar to psia."""
        return pressure_bar * 14.5038
    
    @staticmethod
    def psia_to_bar(pressure_psia):
        """Convert bar to psia."""
        return pressure_psia / 14.5038
    
    @staticmethod
    def psig_to_psia(pressure_psig):
        """Convert psig to psia (assuming 14.696 psia atmospheric)."""
        return pressure_psig + 14.696
    
    @staticmethod
    def psia_to_psig(pressure_psia):
        """Convert psia to psig."""
        return pressure_psia - 14.696
    
    # Compressibility conversions
    @staticmethod
    def per_psi_to_per_mpa(comp_per_psi):
        """Convert 1/psi to 1/MPa."""
        return comp_per_psi * 145.038
    
    @staticmethod
    def per_psi_to_per_kpa(comp_per_psi):
        """Convert 1/psi to 1/kPa."""
        return comp_per_psi * 0.145038


def get_user_units():
    """Get user's preferred units from session state or sidebar."""
    if 'pressure_unit' not in st.session_state:
        st.session_state.pressure_unit = 'psia'
    if 'temperature_unit' not in st.session_state:
        st.session_state.temperature_unit = '°F'
    
    return st.session_state.pressure_unit, st.session_state.temperature_unit


def convert_pressure_to_psia(pressure, unit):
    """Convert pressure from user unit to psia for calculations."""
    if unit == 'psia':
        return pressure
    elif unit == 'psig':
        return UnitConverter.psig_to_psia(pressure)
    elif unit == 'MPa':
        return UnitConverter.mpa_to_psia(pressure)
    elif unit == 'kPa':
        return UnitConverter.kpa_to_psia(pressure)
    elif unit == 'bar':
        return UnitConverter.bar_to_psia(pressure)
    return pressure


def convert_pressure_from_psia(pressure, unit):
    """Convert pressure from psia to user unit for display."""
    if unit == 'psia':
        return pressure
    elif unit == 'psig':
        return UnitConverter.psia_to_psig(pressure)
    elif unit == 'MPa':
        return UnitConverter.psia_to_mpa(pressure)
    elif unit == 'kPa':
        return UnitConverter.psia_to_kpa(pressure)
    elif unit == 'bar':
        return UnitConverter.psia_to_bar(pressure)
    return pressure


def convert_temperature_to_fahrenheit(temp, unit):
    """Convert temperature from user unit to Fahrenheit for calculations."""
    if unit == '°F':
        return temp
    elif unit == '°C':
        return UnitConverter.celsius_to_fahrenheit(temp)
    return temp


def convert_temperature_from_fahrenheit(temp, unit):
    """Convert temperature from Fahrenheit to user unit for display."""
    if unit == '°F':
        return temp
    elif unit == '°C':
        return UnitConverter.fahrenheit_to_celsius(temp)
    return temp


def convert_temperature_from_rankine(temp_r, unit):
    """Convert temperature from Rankine to user unit for display."""
    if unit == '°F':
        return UnitConverter.rankine_to_fahrenheit(temp_r)
    elif unit == '°C':
        return UnitConverter.rankine_to_celsius(temp_r)
    return temp_r


def convert_compressibility_from_per_psi(comp, unit):
    """Convert compressibility from 1/psi to user pressure unit."""
    if unit in ['psia', 'psig']:
        return comp
    elif unit == 'MPa':
        return UnitConverter.per_psi_to_per_mpa(comp)
    elif unit == 'kPa':
        return UnitConverter.per_psi_to_per_kpa(comp)
    elif unit == 'bar':
        return comp * 14.5038
    return comp


# ============================================================================
#  AUTOMATED INTERPRETATION AND FLAGGING
# ============================================================================

class PVTInterpreter:
    """Automated interpretation and flagging of PVT results."""
    
    @staticmethod
    def interpret_z_factor(z, ppr, tpr):
        """Interpret Z-factor and flag potential issues."""
        flags = []
        severity = []  # 'info', 'warning', 'error'
        
        # Basic validity check
        if z < 0.3 or z > 2.0:
            flags.append(f"⚠️ **Unusual Z-factor ({z:.3f})**: Outside typical range (0.3-2.0). Verify input data.")
            severity.append('error')
        
        # Deviation from ideal gas
        if z < 0.85:
            deviation = (1 - z) * 100
            flags.append(f"📊 **Significant non-ideal behavior**: Z = {z:.4f} ({deviation:.1f}% deviation from ideal gas)")
            severity.append('info')
        elif z > 1.15:
            flags.append(f"📈 **Supercompressibility detected**: Z = {z:.4f} (high pressure effects)")
            severity.append('info')
        else:
            flags.append(f"✅ **Near-ideal gas behavior**: Z = {z:.4f}")
            severity.append('info')
        
        # Near critical point warning
        if 0.9 < tpr < 1.2 and 0.5 < ppr < 2.0:
            flags.append("⚠️ **Near critical region**: Results may be sensitive to composition. Consider detailed EOS if accuracy is critical.")
            severity.append('warning')
        
        # High pressure warning
        if ppr > 15:
            flags.append("⚠️ **Extrapolation warning**: Ppr > 15. Hall-Yarborough correlation less reliable at extreme pressures.")
            severity.append('warning')
        
        return flags, severity
    
    @staticmethod
    def interpret_viscosity(mu, pressure_psia, temperature_f):
        """Interpret gas viscosity."""
        flags = []
        severity = []
        
        # Typical range check
        if mu < 0.01:
            flags.append(f"⚠️ **Unusually low viscosity** ({mu:.4f} cP): Verify temperature and composition.")
            severity.append('warning')
        elif mu > 0.05:
            flags.append(f"📊 **High viscosity** ({mu:.4f} cP): May indicate high molecular weight or contaminants.")
            severity.append('info')
        
        # Temperature effects
        if temperature_f < 50:
            flags.append("❄️ **Low temperature**: Consider hydrate formation risk and water content.")
            severity.append('warning')
        
        return flags, severity
    
    @staticmethod
    def interpret_compressibility(cg, ppr):
        """Interpret gas compressibility."""
        flags = []
        severity = []
        
        # High compressibility (near critical)
        if cg > 5e-4:
            flags.append(f"⚠️ **High compressibility** ({cg:.2e} 1/psi): Near critical conditions. Pressure changes cause large volume changes.")
            severity.append('warning')
        
        # Very low compressibility
        if cg < 1e-5:
            flags.append(f"📊 **Low compressibility** ({cg:.2e} 1/psi): Gas behaves more like incompressible fluid at this pressure.")
            severity.append('info')
        
        return flags, severity
    
    @staticmethod
    def interpret_acid_gases(mol_co2, mol_h2s, mol_n2):
        """Interpret acid gas content and provide recommendations."""
        flags = []
        severity = []
        
        total_acid = mol_co2 + mol_h2s
        
        # H2S warnings (safety critical)
        if mol_h2s > 0:
            if mol_h2s < 0.001:
                flags.append(f"⚠️ **Trace H₂S** ({mol_h2s*100:.2f}%): Use sour service materials.")
                severity.append('warning')
            elif mol_h2s < 0.15:
                flags.append(f"☠️ **Sour gas** ({mol_h2s*100:.1f}% H₂S): CRITICAL - Requires H₂S treatment, safety protocols, and specialized materials.")
                severity.append('error')
            else:
                flags.append(f"☠️ **HIGH H₂S** ({mol_h2s*100:.1f}%): EXTREMELY HAZARDOUS - Mandatory treatment required.")
                severity.append('error')
        
        # CO2 warnings (corrosion)
        if mol_co2 > 0.02:
            if mol_co2 < 0.10:
                flags.append(f"⚠️ **Moderate CO₂** ({mol_co2*100:.1f}%): Monitor corrosion, consider treatment if >3%.")
                severity.append('warning')
            else:
                flags.append(f"⚠️ **High CO₂** ({mol_co2*100:.1f}%): Treat for corrosion control and BTU enhancement.")
                severity.append('error')
        
        # Combined acid gas
        if total_acid > 0.10:
            flags.append(f"📊 **Total acid gases** ({total_acid*100:.1f}%): Significantly affects PVT properties and requires treatment.")
            severity.append('warning')
        
        # N2 (diluent)
        if mol_n2 > 0.05:
            flags.append(f"📊 **Nitrogen content** ({mol_n2*100:.1f}%): Reduces heating value, may require rejection if >4%.")
            severity.append('info')
        
        return flags, severity
    
    @staticmethod
    def interpret_water_content(water_bbl_mmcf, pressure_psia, temperature_f):
        """Interpret water content in gas."""
        flags = []
        severity = []
        
        # Hydrate formation risk
        if temperature_f < 60 and pressure_psia > 300:
            hydrate_temp = estimate_hydrate_temperature(pressure_psia)
            if temperature_f < hydrate_temp:
                flags.append(f"❄️ **HYDRATE RISK**: T={temperature_f:.0f}°F < Hydrate T≈{hydrate_temp:.0f}°F at {pressure_psia:.0f} psia. Dehydration required!")
                severity.append('error')
            else:
                margin = temperature_f - hydrate_temp
                if margin < 20:
                    flags.append(f"⚠️ **Marginal hydrate margin** ({margin:.0f}°F): Consider dehydration or inhibition.")
                    severity.append('warning')
        
        # Water content levels
        if water_bbl_mmcf > 100:
            flags.append(f"💧 **Very high water content** ({water_bbl_mmcf:.1f} bbl/MMCF): Dehydration strongly recommended.")
            severity.append('warning')
        elif water_bbl_mmcf > 50:
            flags.append(f"💧 **High water content** ({water_bbl_mmcf:.1f} bbl/MMCF): Monitor for liquids, consider dehydration.")
            severity.append('info')
        elif water_bbl_mmcf > 7:
            flags.append(f"💧 **Moderate water content** ({water_bbl_mmcf:.1f} bbl/MMCF): Typical for unprocessed gas.")
            severity.append('info')
        else:
            flags.append(f"✅ **Low water content** ({water_bbl_mmcf:.1f} bbl/MMCF): Meets pipeline specs (<7 lb/MMCF).")
            severity.append('info')
        
        return flags, severity
    
    @staticmethod
    def interpret_gas_gravity(gas_grav, mol_co2, mol_n2, mol_h2s):
        """Interpret gas gravity and composition."""
        flags = []
        severity = []
        
        # Correct for contaminants
        contaminant_contribution = (mol_co2 * (44.0/28.97)) + (mol_n2 * (28.0/28.97)) + (mol_h2s * (34.1/28.97))
        hydrocarbon_grav = gas_grav - contaminant_contribution
        
        if gas_grav < 0.60:
            flags.append(f"📊 **Light gas** (SG={gas_grav:.3f}): High methane content, good heating value.")
            severity.append('info')
        elif gas_grav > 0.75:
            flags.append(f"📊 **Heavy gas** (SG={gas_grav:.3f}): Higher condensate potential, check for liquid dropout.")
            severity.append('warning')
        
        if gas_grav > 0.80:
            flags.append(f"⚠️ **Very heavy gas**: Consider retrograde condensation risk below dewpoint.")
            severity.append('warning')
        
        return flags, severity


def estimate_hydrate_temperature(pressure_psia):
    """Estimate hydrate formation temperature (simple correlation for sweet gas)."""
    # Simplified Hammerschmidt equation
    if pressure_psia < 100:
        return 32.0
    elif pressure_psia < 1000:
        return 32 + 8.9 * np.log10(pressure_psia / 100)
    else:
        return 32 + 8.9 * np.log10(10) + 4.5 * np.log10(pressure_psia / 1000)


def display_interpretation_flags(flags, severity):
    """Display interpretation flags with appropriate styling."""
    if not flags:
        return
    
    # Group by severity
    errors = [f for f, s in zip(flags, severity) if s == 'error']
    warnings = [f for f, s in zip(flags, severity) if s == 'warning']
    info = [f for f, s in zip(flags, severity) if s == 'info']
    
    # Display errors first (critical)
    if errors:
        st.error("**CRITICAL ISSUES:**\n\n" + "\n\n".join(errors))
    
    # Then warnings
    if warnings:
        st.warning("**CAUTION:**\n\n" + "\n\n".join(warnings))
    
    # Finally info/recommendations
    if info:
        with st.expander("💡 **Engineering Insights** (click to expand)", expanded=True):
            for item in info:
                st.markdown(item)


# Page configuration
st.set_page_config(
    page_title="Shell Gas PVT Analysis",
    page_icon="🐚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
SHELL_RED = "#DD1D21"
SHELL_YELLOW = "#FBCE07"
SHELL_DARK = "#333333"

st.markdown(f"""
    <style>
    @import url('https://fonts.cdnfonts.com/css/futura-md-bt');
    html, body, [class*="css"] {{
        font-family: 'Futura Medium', 'Futura Md BT', 'Futura', sans-serif;
    }}
    .main-header {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {SHELL_RED};
        margin-bottom: 0.5rem;
        font-family: 'Futura Medium', 'Futura', sans-serif;
    }}
    .sub-header {{
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
        font-family: 'Futura Medium', 'Futura', sans-serif;
    }}
    .result-box {{
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid {SHELL_RED};
        margin: 1rem 0;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
    }}
    .shell-banner {{
        background: linear-gradient(135deg, {SHELL_RED} 0%, #C41017 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
    }}
    .shell-banner img {{
        height: 64px;
        width: auto;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
    }}
    .shell-banner h1 {{
        color: white;
        margin: 0;
        font-size: 2rem;
        font-family: 'Futura Medium', 'Futura', sans-serif;
    }}
    .shell-banner p {{
        color: {SHELL_YELLOW};
        margin: 0;
        font-size: 1.1rem;
        font-family: 'Futura Medium', 'Futura', sans-serif;
    }}
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: #FFFEF5;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: {SHELL_YELLOW};
    }}
    </style>
    """, unsafe_allow_html=True)


# Global Plotly theme — Shell palette
PLOT_FONT = dict(family="Futura Medium, Futura, sans-serif", size=12, color=SHELL_DARK)
SHELL_PECTEN_URL = "https://upload.wikimedia.org/wikipedia/en/thumb/e/e8/Shell_logo.svg/200px-Shell_logo.svg.png"
SHELL_LINE_COLORS = [SHELL_RED, "#FBCE07", "#009639", "#003DA5", "#6D2077",
                     "#E87722", "#00A3E0", "#8B8B8D", "#B5121B", "#6B4C00"]
AXIS_STYLE = dict(
    showgrid=True, gridcolor="#e0e0e0", gridwidth=1,
    ticks="outside", ticklen=5, tickwidth=1.5, tickcolor=SHELL_DARK,
    showline=True, linecolor=SHELL_DARK, linewidth=1.5, mirror="allticks",
    title_font=dict(size=13),
)
LAYOUT_STYLE = dict(
    font=PLOT_FONT,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(t=60, b=10),
)


def user_guide_and_background():
    """Display user guide and technical background information."""
    
    st.title("📚 User Guide & Technical Background")
    
    # Quick Start Guide
    st.header("🚀 Quick Start Guide")
    
    st.markdown("""
    ### Getting Started
    
    1. **Select Analysis Type** from the sidebar:
       - **Single Point Calculator**: Calculate PVT properties at one pressure/temperature
       - **Pressure Profile**: Generate plots across a pressure range
       - **Complete PVT Table**: Generate comprehensive 14-property tables with plots (supports multiple temperature curves)
       - **Water-Gas Properties**: Analyze water content and gas-water interactions
       - **Critical Properties**: Estimate pseudo-critical properties from composition
       - **Uncertainty Analysis**: Advanced Monte Carlo simulation with distribution options, sensitivity analysis, and statistical visualizations
    
    2. **Set Unit Preferences** in the sidebar (psia, psig, MPa, kPa, bar; °F, °C)
    
    3. **Enter Input Parameters**:
       - Temperature
       - Pressure (or pressure range)
       - Gas specific gravity
       - Gas composition (N₂, CO₂, H₂S mol fractions)
       - Standard conditions (Psc, Tsc)
    
    4. **Review Results**:
       - Calculated properties displayed in tables
       - Interactive plots (hover for values)
       - Automated interpretation flags and warnings
       - Export data as needed
    """)
    
    # What is PVT Analysis
    st.header("🔬 What is PVT Analysis?")
    
    st.markdown("""
    **PVT (Pressure-Volume-Temperature) analysis** characterizes how reservoir fluids behave under varying conditions. 
    These properties are critical for:
    
    - **Reservoir Engineering**: Material balance, GIIP estimation, production forecasting
    - **Well Testing**: Pressure transient analysis, flow regime identification
    - **Production Operations**: Pipeline design, compression requirements, processing
    - **Facilities Design**: Separator sizing, compressor selection, flow assurance
    
    This tool uses **industry-standard correlations** validated against thousands of laboratory measurements.
    """)
    
    # Key Properties Explained
    st.header("📊 Key PVT Properties")
    
    with st.expander("**Z-factor (Gas Compressibility Factor)**", expanded=False):
        st.markdown("""
        **Definition**: Ratio of actual gas volume to ideal gas volume at the same P and T.
        
        $$Z = \\frac{PV}{nRT}$$
        
        - **Z = 1**: Ideal gas behavior
        - **Z < 1**: Gas is more compressible than ideal (attractive forces dominate)
        - **Z > 1**: Gas is less compressible than ideal (repulsive forces dominate, high pressure)
        
        **Typical Ranges**:
        - Low pressure: Z ≈ 0.85 - 0.95
        - Moderate pressure: Z ≈ 0.70 - 1.10
        - High pressure (>10,000 psia): Z > 1.0 (supercompressibility)
        
        **Correlation Used**: Hall-Yarborough (1973) - Newton-Raphson iterative solution
        """)
    
    with st.expander("**Bg (Gas Formation Volume Factor)**", expanded=False):
        st.markdown("""
        **Definition**: Volume in reservoir barrels (rb) occupied by one standard cubic foot (scf) of gas.
        
        $$B_g = \\frac{P_{sc}}{P} \\times \\frac{T}{T_{sc}} \\times Z \\ \\text{[rb/scf]}$$
        
        - Converts surface volumes to reservoir volumes
        - Decreases with increasing pressure (gas compresses)
        - Critical for GIIP calculations and material balance
        
        **Typical Values**:
        - 1,000 psia: Bg ≈ 0.010 - 0.020 rb/scf
        - 5,000 psia: Bg ≈ 0.002 - 0.004 rb/scf
        """)
    
    with st.expander("**μg (Gas Viscosity)**", expanded=False):
        st.markdown("""
        **Definition**: Resistance to flow, measured in centipoise (cP).
        
        - Increases with pressure and temperature
        - Affects pressure drop in pipelines and wells
        - Critical for transient pressure analysis (permeability estimation)
        
        **Typical Range**: 0.012 - 0.035 cP for natural gas
        
        **Correlations Used**:
        - Lucas (1980) for base viscosity
        - Corrections for N₂, CO₂, H₂S using mixing rules
        """)
    
    with st.expander("**cg (Gas Compressibility)**", expanded=False):
        st.markdown("""
        **Definition**: Fractional change in gas volume per unit pressure change.
        
        $$c_g = -\\frac{1}{V}\\left(\\frac{\\partial V}{\\partial P}\\right)_T = \\frac{1}{P} - \\frac{1}{Z}\\left(\\frac{\\partial Z}{\\partial P}\\right)_T$$
        
        - Units: 1/psi (typically × 10⁻⁵)
        - Highest near critical pressure
        - Used in material balance and well test analysis
        
        **Typical Values**: 10 - 500 × 10⁻⁶ psi⁻¹
        """)
    
    with st.expander("**m(p) (Pseudo-Pressure)**", expanded=False):
        st.markdown("""
        **Definition**: Pressure-dependent function that linearizes gas flow equations.
        
        $$m(p) = 2\\int_0^p \\frac{P}{\\mu Z} dP \\ \\text{[psia²/cP]}$$
        
        - Used extensively in well testing and transient analysis
        - Transforms non-linear diffusivity equation to linear form
        - Enables straight-line analysis on pressure buildup/drawdown plots
        
        **Applications**:
        - Permeability calculation
        - Skin factor determination
        - Rate-time decline analysis
        """)
    
    with st.expander("**Gas Gradient**", expanded=False):
        st.markdown("""
        **Definition**: Vertical pressure change per unit depth (psi/ft).
        
        - Accounts for gravitational effects in wellbores
        - Varies with pressure, temperature, and gas density
        - Critical for bottomhole pressure calculations
        
        **Typical Values**: 0.05 - 0.15 psi/ft (much lower than liquid gradients ~0.4-0.5 psi/ft)
        """)
    
    with st.expander("**Water Content in Gas**", expanded=False):
        st.markdown("""
        **Definition**: Mass of water vapor carried by gas (bbl H₂O / MMscf gas).
        
        - Depends on temperature, pressure, salinity, and gas gravity
        - Decreases with increasing pressure (lower water-holding capacity)
        - Increases with temperature (higher vapor pressure)
        
        **Significance**:
        - Hydrate formation risk assessment
        - Pipeline water accumulation
        - Gas processing (dehydration) requirements
        - Sales gas specifications (typically < 7 lb/MMscf)
        
        **Correlation Used**: McCarthy-Boyd-Reid (GPSA)
        """)
    
    with st.expander("**Gas Solubility in Water (Rs,w)**", expanded=False):
        st.markdown("""
        **Definition**: Volume of gas dissolved in water at reservoir conditions (scf gas / STB water).
        
        - Increases with pressure and decreases with salinity
        - Important for aquifer modeling and water coning
        
        **Typical Values**: 5 - 30 scf/STB at typical reservoir conditions
        """)
    
    with st.expander("**Bw (Water Formation Volume Factor)**", expanded=False):
        st.markdown("""
        **Definition**: Volume of water at reservoir conditions per unit volume at standard conditions (rb/STB).
        
        - Typically Bw ≈ 1.0 - 1.05 (water is nearly incompressible)
        - Accounts for thermal expansion and dissolved gas effects
        """)
    
    with st.expander("**μw (Water Viscosity)**", expanded=False):
        st.markdown("""
        **Definition**: Resistance to water flow (cP).
        
        - Decreases with temperature
        - Increases with salinity and pressure
        
        **Typical Range**: 0.3 - 1.0 cP (much higher than gas)
        """)
    
    with st.expander("**cw (Water Compressibility)**", expanded=False):
        st.markdown("""
        **Definition**: Fractional change in water volume per psi pressure change (1/psi).
        
        - Typically cw ≈ 2-4 × 10⁻⁶ psi⁻¹
        - Nearly constant (water is relatively incompressible)
        - Important for aquifer influx calculations
        """)
    
    # Technical Background
    st.header("🔧 Technical Background")
    
    with st.expander("**Correlations & Methods**", expanded=False):
        st.markdown("""
        ### Z-factor Calculation
        - **Hall-Yarborough (1973)**: Explicit correlation based on Starling-Carnahan equation of state
        - Newton-Raphson iterative solution for reduced density
        - Valid for: 0.2 < Ppr < 20, 1.0 < Tpr < 3.0
        - Average error: ±0.5% vs laboratory data
        
        ### Viscosity Calculation
        - **Lucas (1980)**: Corresponding states method using Tr and Pr
        - Corrections for polar/non-hydrocarbon components
        - Valid for wide range of conditions
        
        ### Pseudo-Critical Properties
        - **Sutton (1985)**: Improved correlation from gas gravity
        - **Wichert-Aziz (1972)**: Correction for acid gases (H₂S, CO₂)
        - Adjusts Pc and Tc based on sour gas content
        
        ### Water Content
        - **McCarthy-Boyd-Reid**: GPSA Engineering Data Book correlation
        - Empirical fit to experimental vapor-liquid equilibrium data
        - Corrections for salinity and non-hydrocarbon gases
        
        ### Integration Methods
        - **Pseudo-pressure**: Trapezoidal rule with adaptive step size
        - Ensures smooth, monotonic curves without discontinuities
        """)
    
    with st.expander("**Automated Interpretation System**", expanded=False):
        st.markdown("""
        ### Engineering Flags
        
        The tool provides intelligent interpretation of results:
        
        **🔴 Error-level Flags**:
        - Non-physical values (Z < 0.3, Z > 2.0)
        - Calculation failures
        - Out-of-range inputs
        
        **🟡 Warning-level Flags**:
        - Near-critical conditions (0.9 < Tpr < 1.2)
        - H₂S safety concerns (> 0.1 mol% = sour gas)
        - CO₂ corrosion risk (> 2 mol%)
        - High extrapolation (Ppr > 15)
        - Hydrate formation risk
        
        **ℹ️ Info-level Flags**:
        - Gas behavior characterization (ideal vs non-ideal)
        - Gravity classification (light < 0.6, heavy > 0.75)
        - Compressibility levels
        - Water content assessment
        
        ### Safety Standards Referenced
        - **NACE MR0175/ISO 15156**: H₂S service limits
        - **GPSA Engineering Data Book**: Hydrocarbon processing standards
        - **API RP 14E**: Offshore production platform design
        """)
    
    with st.expander("**Input Guidelines**", expanded=False):
        st.markdown("""
        ### Gas Specific Gravity
        - Ratio of gas molecular weight to air molecular weight (28.97)
        - **Typical Range**: 0.55 - 0.80 (sweet natural gas)
        - Values > 0.75 indicate heavier gas (higher C₂₊ content)
        - Values < 0.60 indicate dry gas (mostly methane)
        
        ### Component Mol Fractions
        - Must be **decimal fractions** (e.g., 0.05 = 5%)
        - Sum of all components should not exceed 1.0
        - **N₂**: Reduces heating value, increases compression cost
        - **CO₂**: Corrosive, reduces heating value, forms hydrates
        - **H₂S**: Toxic, corrosive, requires special handling
        
        ### Standard Conditions
        - **Common Standards**:
          - USA: 14.696 psia, 60°F
          - Canada: 14.73 psia, 60°F
          - UK: 14.696 psia, 60°F
          - Europe: Often 1.01325 bar (14.696 psia), 15°C (59°F)
        - Ensure consistency with field/company standards
        
        ### Pressure Ranges
        - **Low**: < 1,000 psia (near-atmospheric, pipelines)
        - **Moderate**: 1,000 - 5,000 psia (typical reservoirs)
        - **High**: 5,000 - 15,000 psia (deep reservoirs, HPHT)
        - **Ultra-high**: > 15,000 psia (correlation extrapolation, use with caution)
        
        ### Temperature Ranges
        - **Cold**: < 100°F (shallow formations, surface equipment)
        - **Moderate**: 100 - 250°F (typical reservoirs)
        - **Hot**: 250 - 400°F (deep/HPHT reservoirs)
        - For T < 32°F, consider hydrate/ice formation
        - **Complete PVT Table**: Use temperature range option to plot multiple temperature curves simultaneously for comparing thermal effects across pressure range
        """)
    
    # References
    st.header("📖 References")
    
    st.markdown("""
    ### Key Publications
    
    1. **Hall, K.R. and Yarborough, L. (1973)**: "A New Equation of State for Z-factor Calculations", 
       *Oil & Gas Journal*, June 18, pp. 82-92.
    
    2. **Lucas, K. (1980)**: "Die Druckabhängigkeit der Viskosität von Flüssigkeiten", 
       *Chemie Ingenieur Technik*, 52(10), pp. 788-789.
    
    3. **Sutton, R.P. (1985)**: "Compressibility Factors for High-Molecular-Weight Reservoir Gases", 
       *SPE Paper 14265*, presented at SPE Annual Technical Conference.
    
    4. **Wichert, E. and Aziz, K. (1972)**: "Calculate Z's for Sour Gases", 
       *Hydrocarbon Processing*, May, pp. 119-122.
    
    5. **McCain, W.D., Jr. (1990)**: *The Properties of Petroleum Fluids*, 2nd Edition, 
       PennWell Publishing Company, Tulsa, OK.
    
    6. **GPSA Engineering Data Book** (2022): 15th Edition, Gas Processors Suppliers Association, Tulsa, OK.
    
    7. **Whitson, C.H. and Brulé, M.R. (2000)**: *Phase Behavior*, SPE Monograph Series, Vol. 20.
    
    ### Standards & Guidelines
    
    - **NACE MR0175/ISO 15156**: Materials for Use in H₂S-Containing Environments
    - **API RP 14E**: Design and Installation of Offshore Production Platform Piping Systems
    - **ISO 13443**: Natural Gas - Standard Reference Conditions
    - **AGA Report No. 8**: Compressibility Factors of Natural Gas and Other Related Hydrocarbon Gases
    """)
    
    # Tips and Best Practices
    st.header("💡 Tips & Best Practices")
    
    st.markdown("""
    ### Accuracy Considerations
    
    1. **Gas Characterization**: More accurate composition data yields better results
       - Use chromatography (GC) analysis when available
       - Gas gravity alone is less accurate than full composition
    
    2. **Correlation Applicability**:
       - Hall-Yarborough: Best for Ppr < 15, Tpr > 1.05
       - Near critical (Tpr ≈ 1.0): Consider equation of state (EOS) models
       - Sour gas (H₂S, CO₂ > 5%): Wichert-Aziz corrections are applied
    
    3. **Validation**:
       - Compare with laboratory PVT data when available
       - Cross-check with other correlations (Dranchuk-Abu-Kassem, Brill-Beggs)
       - Verify Z-factor trends (should be smooth, monotonic)
    
    4. **Uncertainty**:
       - Gas gravity uncertainty: ±0.02 typical (about 1-3%)
       - Temperature uncertainty: ±5-10°F typical
       - Correlation accuracy: ±1-3% for Z-factor, ±5-10% for viscosity
       - Use Uncertainty Analysis module for:
         * **Mean ± Std Dev** input when you have measured standard deviations
         * **Mean ± Range** input when you have min/max bounds (automatically converts to std dev)
         * **Normal distribution** for measurement errors (most common)
         * **Uniform distribution** when all values in range are equally likely
         * **Triangular distribution** when you know most likely value and bounds
       - CoV (Coefficient of Variation) interpretation:
         * 🟢 Low (<5%): Well-constrained, confident predictions
         * 🟡 Moderate (5-15%): Typical uncertainty, acceptable for most decisions
         * 🔴 High (>15%): High uncertainty, consider refining measurements or probabilistic design
       - Use Tornado Chart to identify which inputs drive uncertainty most (focus measurement efforts there)
    
    ### Common Pitfalls
    
    - ❌ Using gauge pressure when absolute pressure is required
    - ❌ Mixing incompatible unit systems
    - ❌ Ignoring composition effects (assuming air-like behavior)
    - ❌ Extrapolating correlations beyond valid ranges
    - ❌ Not accounting for hydrate formation at low temperatures
    
    ### When to Use Each Module
    
    - **Single Point**: Quick checks, well head conditions, spot calculations
    - **Pressure Profile**: Visualizing trends, wellbore gradients, separator stages
    - **Complete PVT Table**: Comprehensive reports, material balance, simulation input. Use temperature range option to compare behavior at multiple temperatures on same plots.
    - **Water-Gas**: Hydrate assessment, dehydration design, water production
    - **Critical Properties**: When composition is unknown but gravity available
    - **Uncertainty Analysis**: Risk assessment, probabilistic forecasting, sensitivity to input uncertainties. Features mean±std or mean±range inputs, multiple distribution types (Normal/Uniform/Triangular), box plots, tornado charts for sensitivity ranking, and Coefficient of Variation (CoV) analysis with traffic-light color coding.
    """)


def main():
    """Main application entry point."""
    
    # Header
    st.markdown(f"""
    <div class="shell-banner">
        <img src="{SHELL_PECTEN_URL}" alt="Shell">
        <div>
            <h1>Gas PVT Analysis Tool</h1>
            <p>Industry-standard correlations for reservoir engineering</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar logo
    st.sidebar.markdown(
        f'<div style="text-align:center;margin-bottom:1rem;">'
        f'<img src="{SHELL_PECTEN_URL}" width="100">'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar unit selection
    st.sidebar.title("⚙️ Unit Settings")
    
    pressure_unit = st.sidebar.selectbox(
        "Pressure Units:",
        ['psia', 'psig', 'MPa', 'kPa', 'bar'],
        index=0,
        key='pressure_unit'
    )
    
    temperature_unit = st.sidebar.selectbox(
        "Temperature Units:",
        ['°F', '°C'],
        index=0,
        key='temperature_unit'
    )
    
    # Add units info expander
    with st.sidebar.expander("ℹ️ Unit Conversions"):
        st.markdown("""
        **Supported Conversions:**
        
        **Pressure:**
        - psia (absolute)
        - psig (gauge)
        - MPa (megapascal)
        - kPa (kilopascal)
        - bar
        
        **Temperature:**
        - °F (Fahrenheit)
        - °C (Celsius)
        
        *All calculations use psia and °F internally, 
        with automatic conversion for inputs/outputs.*
        """)
    
    st.sidebar.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("📊 Analysis Type")
    analysis_type = st.sidebar.radio(
        "Select Analysis:",
        ["📚 User Guide & Background", "Single Point Calculator", "Pressure Profile", "Complete PVT Table", "Water-Gas Properties", "Critical Properties", "Uncertainty Analysis"]
    )
    
    if analysis_type == "📚 User Guide & Background":
        user_guide_and_background()
    elif analysis_type == "Single Point Calculator":
        single_point_calculator()
    elif analysis_type == "Pressure Profile":
        pressure_profile()
    elif analysis_type == "Complete PVT Table":
        complete_pvt_table()
    elif analysis_type == "Water-Gas Properties":
        water_gas_properties()
    elif analysis_type == "Critical Properties":
        critical_properties()
    elif analysis_type == "Uncertainty Analysis":
        uncertainty_analysis()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This tool implements industry-standard correlations:\n\n"
        "- **Z-factor**: Hall-Yarborough\n"
        "- **Viscosity**: Lucas with Standing correction\n"
        "- **Critical Properties**: Sutton, Riazi-Daubert\n"
        "- **Water Properties**: McCain, Whitson-Brule\n\n"
        "Based on SPE Monograph Vol. 20 by Whitson & Brulé"
    )
    st.sidebar.caption("Built with Shell engineering standards")


def single_point_calculator():
    """Single point PVT property calculator."""
    st.header("Single Point Calculator")
    st.markdown("Calculate gas properties at a specific pressure and temperature.")
    
    # Get user's preferred units
    pressure_unit, temperature_unit = get_user_units()
    
    # Show current units
    st.info(f"📏 **Current Units:** Pressure = {pressure_unit}, Temperature = {temperature_unit} (Change in sidebar)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        
        # Reservoir conditions
        st.markdown("**Reservoir Conditions**")
        
        # Set default values based on units
        default_p = 3000.0 if pressure_unit == 'psia' else convert_pressure_from_psia(3000.0, pressure_unit)
        default_t = 180.0 if temperature_unit == '°F' else convert_temperature_from_fahrenheit(180.0, temperature_unit)
        
        pressure_input = st.number_input(f"Pressure ({pressure_unit})", min_value=0.0, value=default_p, step=100.0 if pressure_unit in ['psia', 'psig'] else 1.0)
        temperature_input = st.number_input(f"Temperature ({temperature_unit})", value=default_t, step=10.0 if temperature_unit == '°F' else 5.0)
        
        # Convert to internal units (psia, °F)
        pressure = convert_pressure_to_psia(pressure_input, pressure_unit)
        temperature = convert_temperature_to_fahrenheit(temperature_input, temperature_unit)
        
        # Standard conditions
        st.markdown("**Standard Conditions**")
        col_psc, col_tsc = st.columns(2)
        with col_psc:
            psc_default = 14.696 if pressure_unit == 'psia' else convert_pressure_from_psia(14.696, pressure_unit)
            psc_input = st.number_input(f"Psc ({pressure_unit})", value=psc_default, step=0.001 if pressure_unit in ['psia', 'psig'] else 0.01, format="%.3f")
            psc = convert_pressure_to_psia(psc_input, pressure_unit)
        with col_tsc:
            tsc_default = 60.0 if temperature_unit == '°F' else convert_temperature_from_fahrenheit(60.0, temperature_unit)
            tsc_input = st.number_input(f"Tsc ({temperature_unit})", value=tsc_default, step=1.0)
            tsc = convert_temperature_to_fahrenheit(tsc_input, temperature_unit)
        
        # Gas properties
        st.markdown("**Gas Composition**")
        gas_gravity = st.number_input("Gas Specific Gravity", min_value=0.5, max_value=1.5, value=0.65, step=0.01)
        
        col_n2, col_co2, col_h2s = st.columns(3)
        with col_n2:
            mol_n2 = st.number_input("N₂ (mol frac)", min_value=0.0, max_value=1.0, value=0.02, step=0.01, format="%.3f")
        with col_co2:
            mol_co2 = st.number_input("CO₂ (mol frac)", min_value=0.0, max_value=1.0, value=0.01, step=0.01, format="%.3f")
        with col_h2s:
            mol_h2s = st.number_input("H₂S (mol frac)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.3f")
        
        # Pseudo-critical properties
        st.markdown("**Pseudo-Critical Properties**")
        use_correlation = st.checkbox("Calculate from gas gravity", value=True)
        
        if use_correlation:
            pc = p_crit(gas_gravity, mol_n2, mol_co2, mol_h2s, 0.0, 0.85, 120.0)
            tc = t_crit(gas_gravity, mol_n2, mol_co2, mol_h2s, 0.0, 0.85, 120.0)
            
            # Display in user units
            pc_display = convert_pressure_from_psia(pc, pressure_unit)
            tc_display = convert_temperature_from_rankine(tc, temperature_unit)
            st.info(f"Calculated: Pc = {pc_display:.1f} {pressure_unit}, Tc = {tc_display:.1f} {temperature_unit}")
        else:
            col_pc, col_tc = st.columns(2)
            with col_pc:
                pc_default = 670.0 if pressure_unit == 'psia' else convert_pressure_from_psia(670.0, pressure_unit)
                pc_input = st.number_input(f"Pc ({pressure_unit})", value=pc_default, step=10.0 if pressure_unit in ['psia', 'psig'] else 1.0)
                pc = convert_pressure_to_psia(pc_input, pressure_unit)
            with col_tc:
                tc_default = 380.0 if temperature_unit == '°F' else (380.0 - 459.67) * 5/9  # Rankine to user unit
                tc_input = st.number_input(f"Tc ({temperature_unit})", value=tc_default, step=10.0 if temperature_unit == '°F' else 5.0)
                # Convert from user temperature to Rankine
                if temperature_unit == '°F':
                    tc = tc_input + 459.67
                else:  # °C
                    tc = (tc_input * 9/5) + 491.67
        
        calculate_button = st.button("🔬 Calculate Properties", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Results")
        
        if calculate_button:
            try:
                # Calculate reduced properties
                ppr = pressure / pc
                tpr = (temperature + 459.67) / tc
                
                # Calculate PVT properties
                z = z_gas(ppr, tpr)
                bg = bg_rv_per_scv(pressure, temperature, psc, tsc, pc, tc)
                mu = mu_gas(pressure, temperature, pc, tc, gas_gravity, mol_n2, mol_co2, mol_h2s)
                cg = c_gas(pressure, pc, tpr)
                density = gas_gravity * 28.97 * pressure / (10.73 * (temperature + 459.67) * z)
                pp = pseudo_pressure(pressure, temperature, pc, tc, gas_gravity, mol_n2, mol_co2, mol_h2s)
                
                # Convert results to display units
                cg_display = convert_compressibility_from_per_psi(cg, pressure_unit)
                pp_display = pp if pressure_unit in ['psia', 'psig'] else f"{pp / 145.038**2:.2f}"  # Convert psia²/cP to MPa²/cP
                
                # Determine compressibility units
                if pressure_unit in ['psia', 'psig']:
                    cg_unit = "1/psi"
                elif pressure_unit == 'MPa':
                    cg_unit = "1/MPa"
                elif pressure_unit == 'kPa':
                    cg_unit = "1/kPa"
                else:
                    cg_unit = "1/bar"
                
                # Display results
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    "Property": [
                        "Pseudo-reduced Pressure", 
                        "Pseudo-reduced Temperature",
                        "Z-factor",
                        "Formation Volume Factor",
                        "Viscosity",
                        "Compressibility",
                        "Density",
                        "Pseudo-Pressure"
                    ],
                    "Value": [
                        f"{ppr:.4f}",
                        f"{tpr:.4f}",
                        f"{z:.6f}",
                        f"{bg:.6f}",
                        f"{mu:.6f}",
                        f"{cg_display:.6e}",
                        f"{density:.4f}",
                        f"{pp:.2f}"
                    ],
                    "Units": [
                        "dimensionless",
                        "dimensionless",
                        "dimensionless",
                        "RV/SCV",
                        "cP",
                        cg_unit,
                        "lb/ft³",
                        f"{pressure_unit}²/cP" if pressure_unit in ['psia', 'psig'] else "psia²/cP"
                    ]
                })
                
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Automatic interpretation and flagging
                st.markdown("---")
                st.markdown("### 🔍 Automated Interpretation")
                
                all_flags = []
                all_severity = []
                
                # Z-factor interpretation
                z_flags, z_sev = PVTInterpreter.interpret_z_factor(z, ppr, tpr)
                all_flags.extend(z_flags)
                all_severity.extend(z_sev)
                
                # Viscosity interpretation
                mu_flags, mu_sev = PVTInterpreter.interpret_viscosity(mu, pressure, temperature)
                all_flags.extend(mu_flags)
                all_severity.extend(mu_sev)
                
                # Compressibility interpretation
                cg_flags, cg_sev = PVTInterpreter.interpret_compressibility(cg, ppr)
                all_flags.extend(cg_flags)
                all_severity.extend(cg_sev)
                
                # Gas gravity interpretation
                grav_flags, grav_sev = PVTInterpreter.interpret_gas_gravity(gas_gravity, mol_co2, mol_n2, mol_h2s)
                all_flags.extend(grav_flags)
                all_severity.extend(grav_sev)
                
                # Acid gas interpretation
                acid_flags, acid_sev = PVTInterpreter.interpret_acid_gases(mol_co2, mol_h2s, mol_n2)
                all_flags.extend(acid_flags)
                all_severity.extend(acid_sev)
                
                # Display all interpretations
                display_interpretation_flags(all_flags, all_severity)
                
            except Exception as e:
                st.error(f"❌ Error during calculation: {str(e)}")


def pressure_profile():
    """Generate pressure vs depth profile."""
    st.header("Gas Gradient - Pressure vs Depth Profile")
    st.markdown("Calculate pressure variation with depth in a gas column.")
    
    # Get user's preferred units
    pressure_unit, temperature_unit = get_user_units()
    st.info(f"📏 **Current Units:** Pressure = {pressure_unit}, Temperature = {temperature_unit} (Change in sidebar)")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Datum conditions
        default_p = 3000.0 if pressure_unit == 'psia' else convert_pressure_from_psia(3000.0, pressure_unit)
        default_t = 180.0 if temperature_unit == '°F' else convert_temperature_from_fahrenheit(180.0, temperature_unit)
        
        p_datum_input = st.number_input(f"Datum Pressure ({pressure_unit})", min_value=0.0, value=default_p, step=100.0 if pressure_unit in ['psia', 'psig'] else 1.0)
        t_datum_input = st.number_input(f"Datum Temperature ({temperature_unit})", value=default_t, step=10.0 if temperature_unit == '°F' else 5.0)
        
        p_datum = convert_pressure_to_psia(p_datum_input, pressure_unit)
        t_datum = convert_temperature_to_fahrenheit(t_datum_input, temperature_unit)
        
        # Depth range
        depth_min = st.number_input("Minimum Depth (ft)", value=-500.0, step=50.0)
        depth_max = st.number_input("Maximum Depth (ft)", value=500.0, step=50.0)
        
        # Gas properties
        gas_gravity = st.number_input("Gas Specific Gravity", min_value=0.5, max_value=1.5, value=0.65, step=0.01)
        
        # Gradient
        temp_grad_unit = f"{temperature_unit}/ft"
        temp_gradient_input = st.number_input(f"Temperature Gradient ({temp_grad_unit})", value=0.01 if temperature_unit == '°F' else 0.0056, step=0.001, format="%.3f")
        # Convert to °F/ft for calculations
        temp_gradient = temp_gradient_input if temperature_unit == '°F' else temp_gradient_input * 1.8
        
        # Pseudo-critical
        pc = p_crit(gas_gravity, 0.0, 0.0, 0.0, 0.0, 0.85, 120.0)
        tc = t_crit(gas_gravity, 0.0, 0.0, 0.0, 0.0, 0.85, 120.0)
        
        pc_display = convert_pressure_from_psia(pc, pressure_unit)
        tc_display = convert_temperature_from_rankine(tc, temperature_unit)
        st.info(f"Pc = {pc_display:.1f} {pressure_unit}\nTc = {tc_display:.1f} {temperature_unit}")
        
        generate_button = st.button("📈 Generate Profile", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Pressure-Depth Profile")
        
        if generate_button:
            try:
                # Generate depth array
                depths = np.linspace(depth_min, depth_max, 50)
                pressures = []
                z_factors = []
                
                for depth in depths:
                    temp = t_datum + temp_gradient * depth
                    grad = gas_grad(p_datum, temp, gas_gravity, pc, tc)
                    pressure = p_datum + grad * depth
                    
                    ppr = pressure / pc
                    tpr = (temp + 459.67) / tc
                    z = z_gas(ppr, tpr)
                    
                    pressures.append(pressure)
                    z_factors.append(z)
                
                # Create figure with subplots
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Pressure vs Depth", "Z-factor vs Depth"),
                    horizontal_spacing=0.15
                )
                
                # Pressure plot
                fig.add_trace(
                    go.Scatter(x=pressures, y=depths, mode='lines', name='Pressure',
                              line=dict(color=SHELL_RED, width=2.5)),
                    row=1, col=1
                )
                
                # Z-factor plot
                fig.add_trace(
                    go.Scatter(x=z_factors, y=depths, mode='lines', name='Z-factor',
                              line=dict(color=SHELL_YELLOW, width=2.5)),
                    row=1, col=2
                )
                
                # Update axes
                fig.update_xaxes(title_text="Pressure (psia)", row=1, col=1)
                fig.update_xaxes(title_text="Z-factor", row=1, col=2)
                fig.update_yaxes(title_text="Depth (ft)", autorange="reversed", row=1, col=1)
                fig.update_yaxes(title_text="Depth (ft)", autorange="reversed", row=1, col=2)
                
                layout_with_margin = {**LAYOUT_STYLE, "margin": dict(t=60, b=10, r=40)}
                fig.update_layout(height=450, showlegend=False, **layout_with_margin)
                for ann in fig.layout.annotations:
                    ann.y = ann.y + 0.04
                    ann.font = dict(size=14, family="Futura Medium, Futura, sans-serif", color="#333")
                fig.update_xaxes(**AXIS_STYLE)
                fig.update_yaxes(**AXIS_STYLE)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create data table
                results_df = pd.DataFrame({
                    "Depth (ft)": depths[::5],  # Show every 5th point
                    "Pressure (psia)": [p for p in pressures[::5]],
                    "Z-factor": [z for z in z_factors[::5]]
                })
                
                st.subheader("Data Table")
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=csv,
                    file_name="pressure_profile.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Error generating profile: {str(e)}")


def complete_pvt_table():
    """Generate complete PVT table over pressure range."""
    st.header("Complete PVT Table")
    st.markdown("Generate comprehensive PVT properties over a range of pressures.")
    
    # Get user's preferred units
    pressure_unit, temperature_unit = get_user_units()
    st.info(f"📏 **Current Units:** Pressure = {pressure_unit}, Temperature = {temperature_unit} (Change in sidebar)")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Pressure range
        st.markdown("**Pressure Range**")
        pmin_default = 500.0 if pressure_unit == 'psia' else convert_pressure_from_psia(500.0, pressure_unit)
        pmax_default = 5000.0 if pressure_unit == 'psia' else convert_pressure_from_psia(5000.0, pressure_unit)
        
        p_min_input = st.number_input(f"Minimum Pressure ({pressure_unit})", min_value=0.0, value=pmin_default, step=10.0 if pressure_unit in ['psia', 'psig'] else 1.0)
        p_max_input = st.number_input(f"Maximum Pressure ({pressure_unit})", min_value=0.0, value=pmax_default, step=100.0 if pressure_unit in ['psia', 'psig'] else 10.0)
        n_points = st.slider("Number of Points", min_value=10, max_value=100, value=20)
        
        p_min = convert_pressure_to_psia(p_min_input, pressure_unit)
        p_max = convert_pressure_to_psia(p_max_input, pressure_unit)
        
        # Temperature range option
        st.markdown("**Temperature**")
        use_temp_range = st.checkbox("Use Temperature Range", value=False, help="Plot multiple temperature curves")
        
        if use_temp_range:
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                tmin_default = 150.0 if temperature_unit == '°F' else convert_temperature_from_fahrenheit(150.0, temperature_unit)
                t_min_input = st.number_input(f"Min Temp ({temperature_unit})", value=tmin_default, step=10.0 if temperature_unit == '°F' else 5.0)
            with col_t2:
                tmax_default = 250.0 if temperature_unit == '°F' else convert_temperature_from_fahrenheit(250.0, temperature_unit)
                t_max_input = st.number_input(f"Max Temp ({temperature_unit})", value=tmax_default, step=10.0 if temperature_unit == '°F' else 5.0)
            
            n_temps = st.slider("Number of Temperature Curves", min_value=2, max_value=10, value=3)
            
            t_min = convert_temperature_to_fahrenheit(t_min_input, temperature_unit)
            t_max = convert_temperature_to_fahrenheit(t_max_input, temperature_unit)
            temperatures = np.linspace(t_min, t_max, n_temps)
        else:
            temp_default = 180.0 if temperature_unit == '°F' else convert_temperature_from_fahrenheit(180.0, temperature_unit)
            temperature_input = st.number_input(f"Temperature ({temperature_unit})", value=temp_default, step=10.0 if temperature_unit == '°F' else 5.0)
            temperature = convert_temperature_to_fahrenheit(temperature_input, temperature_unit)
            temperatures = [temperature]
        
        # Standard conditions
        psc_default = 14.696 if pressure_unit == 'psia' else convert_pressure_from_psia(14.696, pressure_unit)
        tsc_default = 60.0 if temperature_unit == '°F' else convert_temperature_from_fahrenheit(60.0, temperature_unit)
        
        psc_input = st.number_input(f"Psc ({pressure_unit})", value=psc_default, step=0.001 if pressure_unit in ['psia', 'psig'] else 0.01, format="%.3f")
        tsc_input = st.number_input(f"Tsc ({temperature_unit})", value=tsc_default, step=1.0)
        
        psc = convert_pressure_to_psia(psc_input, pressure_unit)
        tsc = convert_temperature_to_fahrenheit(tsc_input, temperature_unit)
        
        # Gas properties
        gas_gravity = st.number_input("Gas Specific Gravity", min_value=0.5, max_value=1.5, value=0.65, step=0.01)
        mol_n2 = st.number_input("N₂ (mol frac)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.3f")
        mol_co2 = st.number_input("CO₂ (mol frac)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.3f")
        mol_h2s = st.number_input("H₂S (mol frac)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.3f")
        
        # Calculate pseudo-critical properties
        pc = p_crit(gas_gravity, mol_n2, mol_co2, mol_h2s, 0.0, 0.85, 120.0)
        tc = t_crit(gas_gravity, mol_n2, mol_co2, mol_h2s, 0.0, 0.85, 120.0)
        
        st.info(f"Pc = {pc:.1f} psia\nTc = {tc:.1f} °R")
        
        generate_button = st.button("📊 Generate PVT Table", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("PVT Properties vs Pressure")
        
        if generate_button:
            try:
                # Validate inputs
                if p_min >= p_max:
                    st.error("❌ Minimum pressure must be less than maximum pressure!")
                    return
                
                if pc <= 0 or tc <= 0:
                    st.error(f"❌ Invalid pseudo-critical properties: Pc={pc:.1f} psia, Tc={tc:.1f} °R")
                    return
                
                # Generate pressure array
                pressures = np.linspace(p_min, p_max, n_points)
                
                # Calculate properties (matching Excel format) for each temperature
                all_results = []
                
                # Color palette for different temperatures
                colors = SHELL_LINE_COLORS
                
                # Assume TDS for water calculations
                tds = 0.0  # ppm, fresh water
                
                for temp_idx, temperature in enumerate(temperatures):
                    results = {
                        "Pressure": [],
                        "Temperature": [],
                        "z": [],
                        "Bg": [],
                        "Mug": [],
                        "cGas": [],
                        "GasGrad": [],
                        "m(p)": [],
                        "WtrInGas": [],
                        "GasInWtr": [],
                        "Bw": [],
                        "Muw": [],
                        "cWtr": [],
                        "WtrGrad": []
                    }
                    
                    # Initialize pseudo-pressure for incremental calculation
                    pp_cumulative = 0.0
                    p_prev = 0.0
                    z_prev = 1.0
                    mu_prev = 0.01
                    
                    for i, p in enumerate(pressures):
                        try:
                            ppr = p / pc
                            tpr = (temperature + 459.67) / tc
                            
                            # Gas properties
                            z = z_gas(ppr, tpr)
                            
                            # Ensure z is real (not complex)
                            if isinstance(z, complex):
                                z = abs(z)
                            
                            # Validate z-factor to prevent anomalies
                            if z < 0.3 or z > 2.5:
                                st.warning(f"⚠️ Unusual Z-factor at P={p:.0f} psia, T={temperature:.0f}°F: z={z:.4f}")
                                z = max(0.3, min(2.5, z))  # Clamp to reasonable range
                            
                            bg = bg_rv_per_scv(p, temperature, psc, tsc, pc, tc)
                            mu = mu_gas(p, temperature, pc, tc, gas_gravity, mol_n2, mol_co2, mol_h2s)
                            
                            # Validate viscosity
                            if mu < 0.008 or mu > 0.1:
                                mu = max(0.008, min(0.1, mu))
                            
                            cg = c_gas(p, pc, tpr)
                            gas_gradient = gas_grad(p, temperature, gas_gravity, pc, tc)
                            
                            # Calculate pseudo-pressure incrementally using trapezoidal rule
                            # m(p) = integral of 2*P/(mu*Z) dP
                            if i > 0:
                                dp = p - p_prev
                                # Trapezoidal integration
                                integrand_prev = 2 * p_prev / (mu_prev * z_prev)
                                integrand_curr = 2 * p / (mu * z)
                                pp_cumulative += 0.5 * dp * (integrand_prev + integrand_curr)
                            
                            pp = pp_cumulative
                            p_prev = p
                            z_prev = z
                            mu_prev = mu
                            
                            # Water content in gas
                            wtr_in_gas = wtr_in_gas_bbl_per_mmcf(p, temperature, tds, gas_gravity)
                            
                            # Gas solubility in water
                            gas_in_wtr = gas_in_wtr_scf_per_stb(p, temperature, tds)
                            
                            # Water properties
                            bw = wtr_fvf(p, temperature, tds, gas_in_wtr)
                            mu_w = mu_wtr(p, temperature, tds)
                            cw = c_wtr(p, temperature, tds, psc, tsc, gas_gravity, temperature + 459.67)
                            
                            # Water gradient (approximate)
                            water_density = 62.4 * bw  # lb/ft³
                            wtr_grad = water_density / 144.0  # psi/ft
                            
                        except Exception as e:
                            st.warning(f"⚠️ Calculation failed at P={p:.1f} psia, T={temperature:.0f}°F: {str(e)}. Using NaN.")
                            z = bg = mu = cg = gas_gradient = pp = np.nan
                            wtr_in_gas = gas_in_wtr = bw = mu_w = cw = wtr_grad = np.nan
                        
                        results["Pressure"].append(p)
                        results["Temperature"].append(temperature)
                        results["z"].append(z)
                        results["Bg"].append(bg)
                        results["Mug"].append(mu)
                        results["cGas"].append(cg)
                        results["GasGrad"].append(gas_gradient)
                        results["m(p)"].append(pp)
                        results["WtrInGas"].append(wtr_in_gas)
                        results["GasInWtr"].append(gas_in_wtr)
                        results["Bw"].append(bw)
                        results["Muw"].append(mu_w)
                        results["cWtr"].append(cw)
                        results["WtrGrad"].append(wtr_grad)
                    
                    # Store results for this temperature
                    all_results.append((temperature, results))
                
                # Create plots with NaN filtering for smooth curves
                fig = make_subplots(
                    rows=2, cols=3,
                    subplot_titles=("Z-factor vs P", "Bg vs P", "Gas Viscosity vs P", 
                                  "Water in Gas", "Gas Gradient", "m(p) vs P"),
                    vertical_spacing=0.22,
                    horizontal_spacing=0.14
                )
                
                # Helper function to filter out NaN and inf values
                def filter_valid(x_data, y_data):
                    """Remove NaN and inf values from datasets."""
                    x = np.array(x_data)
                    y = np.array(y_data)
                    mask = np.isfinite(x) & np.isfinite(y)
                    return x[mask], y[mask]
                
                # Plot each temperature as a separate line
                for temp_idx, (temperature, results) in enumerate(all_results):
                    color = colors[temp_idx % len(colors)]
                    temp_label = f"T={temperature:.0f}°F"
                    show_legend = (temp_idx == 0) if len(all_results) == 1 else True
                    
                    # Z-factor
                    x_valid, y_valid = filter_valid(results["Pressure"], results["z"])
                    fig.add_trace(
                        go.Scatter(x=x_valid, y=y_valid, 
                                  mode='lines', name=temp_label,
                                  line=dict(color=color, width=2.5),
                                  showlegend=show_legend,
                                  legendgroup=temp_label),
                        row=1, col=1
                    )
                    
                    # Bg
                    x_valid, y_valid = filter_valid(results["Pressure"], results["Bg"])
                    fig.add_trace(
                        go.Scatter(x=x_valid, y=y_valid, 
                                  mode='lines', name=temp_label,
                                  line=dict(color=color, width=2.5),
                                  showlegend=False,
                                  legendgroup=temp_label),
                        row=1, col=2
                    )
                    
                    # Viscosity
                    x_valid, y_valid = filter_valid(results["Pressure"], results["Mug"])
                    fig.add_trace(
                        go.Scatter(x=x_valid, y=y_valid, 
                                  mode='lines', name=temp_label,
                                  line=dict(color=color, width=2.5),
                                  showlegend=False,
                                  legendgroup=temp_label),
                        row=1, col=3
                    )
                    
                    # Water in gas
                    x_valid, y_valid = filter_valid(results["Pressure"], results["WtrInGas"])
                    fig.add_trace(
                        go.Scatter(x=x_valid, y=y_valid, 
                                  mode='lines', name=temp_label,
                                  line=dict(color=color, width=2.5),
                                  showlegend=False,
                                  legendgroup=temp_label),
                        row=2, col=1
                    )
                    
                    # Gas gradient
                    x_valid, y_valid = filter_valid(results["Pressure"], results["GasGrad"])
                    fig.add_trace(
                        go.Scatter(x=x_valid, y=y_valid, 
                                  mode='lines', name=temp_label,
                                  line=dict(color=color, width=2.5),
                                  showlegend=False,
                                  legendgroup=temp_label),
                        row=2, col=2
                    )
                    
                    # Pseudo-pressure
                    x_valid, y_valid = filter_valid(results["Pressure"], results["m(p)"])
                    fig.add_trace(
                        go.Scatter(x=x_valid, y=y_valid, 
                                  mode='lines', name=temp_label,
                                  line=dict(color=color, width=2.5),
                                  showlegend=False,
                                  legendgroup=temp_label),
                        row=2, col=3
                    )
                
                # Update axes
                for i in range(1, 3):
                    for j in range(1, 4):
                        fig.update_xaxes(title_text="Pressure (psia)", row=i, col=j)
                
                fig.update_yaxes(title_text="z", row=1, col=1)
                fig.update_yaxes(title_text="Bg (RV/SCV)", row=1, col=2)
                fig.update_yaxes(title_text="μg (cP)", row=1, col=3)
                fig.update_yaxes(title_text="bbl/MMCF", row=2, col=1)
                fig.update_yaxes(title_text="psi/ft", row=2, col=2)
                fig.update_yaxes(title_text="psia²/cP", row=2, col=3)
                
                pvt_layout = {**LAYOUT_STYLE, "margin": dict(t=60, b=10, r=40)}
                fig.update_layout(height=750, showlegend=len(all_results) > 1,
                                  **pvt_layout)
                # Shift subplot titles up so they don't overlap borders
                for ann in fig.layout.annotations:
                    ann.y = ann.y + 0.03
                    ann.font = dict(size=14, family="Futura Medium, Futura, sans-serif", color="#333")
                fig.update_xaxes(**AXIS_STYLE)
                fig.update_yaxes(**AXIS_STYLE)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display table - format like Excel - combine all temperatures
                st.subheader("Complete PVT Table (Excel Format)")
                
                # Combine all results into one dataframe
                all_dfs = []
                for temperature, results in all_results:
                    df_temp = pd.DataFrame(results)
                    all_dfs.append(df_temp)
                
                df = pd.concat(all_dfs, ignore_index=True)
                
                # Format columns to match Excel precision
                df["Pressure"] = df["Pressure"].round(0)
                df["Temperature"] = df["Temperature"].round(0)
                df["z"] = df["z"].round(4)
                df["Bg"] = df["Bg"].apply(lambda x: f"{x:.6f}")
                df["Mug"] = df["Mug"].apply(lambda x: f"{x:.5f}")
                df["cGas"] = df["cGas"].apply(lambda x: f"{x:.3E}")
                df["GasGrad"] = df["GasGrad"].round(5)
                df["m(p)"] = df["m(p)"].apply(lambda x: f"{x:.3E}")
                df["WtrInGas"] = df["WtrInGas"].round(4)
                df["GasInWtr"] = df["GasInWtr"].round(3)
                df["Bw"] = df["Bw"].round(4)
                df["Muw"] = df["Muw"].round(4)
                df["cWtr"] = df["cWtr"].apply(lambda x: f"{x:.3E}")
                df["WtrGrad"] = df["WtrGrad"].round(4)
                
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download PVT Table as CSV",
                    data=csv,
                    file_name="pvt_table.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Error generating PVT table: {str(e)}")


def water_gas_properties():
    """Water-gas interaction properties."""
    st.header("Water-Gas Interaction Properties")
    st.markdown("Calculate water content in gas and gas solubility in water.")
    
    # Get user's preferred units
    pressure_unit, temperature_unit = get_user_units()
    st.info(f"📏 **Current Units:** Pressure = {pressure_unit}, Temperature = {temperature_unit} (Change in sidebar)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        
        default_p = 3000.0 if pressure_unit == 'psia' else convert_pressure_from_psia(3000.0, pressure_unit)
        default_t = 180.0 if temperature_unit == '°F' else convert_temperature_from_fahrenheit(180.0, temperature_unit)
        
        pressure_input = st.number_input(f"Pressure ({pressure_unit})", min_value=0.0, value=default_p, step=100.0 if pressure_unit in ['psia', 'psig'] else 1.0)
        temperature_input = st.number_input(f"Temperature ({temperature_unit})", value=default_t, step=10.0 if temperature_unit == '°F' else 5.0)
        
        pressure = convert_pressure_to_psia(pressure_input, pressure_unit)
        temperature = convert_temperature_to_fahrenheit(temperature_input, temperature_unit)
        
        tds = st.number_input("Total Dissolved Solids (ppm)", min_value=0.0, value=35000.0, step=1000.0)
        gas_gravity = st.number_input("Gas Specific Gravity", min_value=0.5, max_value=1.5, value=0.65, step=0.01)
        
        calculate_button = st.button("💧 Calculate Properties", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Results")
        
        if calculate_button:
            try:
                # Water content in gas
                wtr_content = wtr_in_gas_bbl_per_mmcf(pressure, temperature, tds, gas_gravity)
                
                # Gas solubility in water
                gas_sol = gas_in_wtr_scf_per_stb(pressure, temperature, tds)
                
                # Water FVF
                bw = wtr_fvf(pressure, temperature, tds, gas_sol)
                
                # Water viscosity
                mu_w = mu_wtr(pressure, temperature, tds)
                
                # Water compressibility
                psc = 14.696  # Standard pressure
                tsc = 60.0    # Standard temperature
                cw = c_wtr(pressure, temperature, tds, psc, tsc)
                
                # Display results
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                
                results_df = pd.DataFrame({
                    "Property": [
                        "Water Content in Gas",
                        "Gas Solubility in Water",
                        "Water Formation Volume Factor",
                        "Water Viscosity",
                        "Water Compressibility"
                    ],
                    "Value": [
                        f"{wtr_content:.4f}",
                        f"{gas_sol:.4f}",
                        f"{bw:.6f}",
                        f"{mu_w:.4f}",
                        f"{cw:.6e}"
                    ],
                    "Units": [
                        "bbl/MMCF",
                        "SCF/STB",
                        "RB/STB",
                        "cP",
                        "1/psi"
                    ]
                })
                
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Automated interpretation
                st.markdown("---")
                st.markdown("### 🔍 Automated Interpretation")
                
                all_flags = []
                all_severity = []
                
                # Water content interpretation
                wtr_flags, wtr_sev = PVTInterpreter.interpret_water_content(wtr_content, pressure, temperature)
                all_flags.extend(wtr_flags)
                all_severity.extend(wtr_sev)
                
                # Display interpretations
                display_interpretation_flags(all_flags, all_severity)
                
            except Exception as e:
                st.error(f"❌ Error during calculation: {str(e)}")


def critical_properties():
    """Calculate critical properties."""
    st.header("Critical Property Correlations")
    st.markdown("Calculate pseudo-critical and critical properties for gas mixtures and heavy fractions.")
    
    # Tab selection
    tab1, tab2 = st.tabs(["Gas Mixture", "C7+ Fraction"])
    
    with tab1:
        st.subheader("Gas Mixture Pseudo-Critical Properties")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gas_gravity = st.number_input("Gas Specific Gravity", min_value=0.5, max_value=1.5, value=0.65, step=0.01, key="crit_gg")
            mol_n2 = st.number_input("N₂ (mol frac)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.3f", key="crit_n2")
            mol_co2 = st.number_input("CO₂ (mol frac)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.3f", key="crit_co2")
            mol_h2s = st.number_input("H₂S (mol frac)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.3f", key="crit_h2s")
            
            calc_button = st.button("🔬 Calculate", key="gas_crit", use_container_width=True)
        
        with col2:
            if calc_button:
                try:
                    # Calculate uncorrected properties
                    pc_uncorr = p_crit(gas_gravity, mol_n2, mol_co2, mol_h2s, 0.0, 0.85, 120.0)
                    tc_uncorr = t_crit(gas_gravity, mol_n2, mol_co2, mol_h2s, 0.0, 0.85, 120.0)
                    
                    # Calculate corrected properties (Wichert-Aziz correction)
                    pc_corr = p_crit_corr(pc_uncorr, tc_uncorr, mol_co2, mol_h2s)
                    tc_corr = t_crit_corr(tc_uncorr, mol_co2, mol_h2s)
                    
                    results_df = pd.DataFrame({
                        "Property": [
                            "Pseudo-critical Pressure (Sutton)",
                            "Pseudo-critical Temperature (Sutton)",
                            "Pseudo-critical Pressure (Corrected)",
                            "Pseudo-critical Temperature (Corrected)"
                        ],
                        "Value": [
                            f"{pc_uncorr:.2f}",
                            f"{tc_uncorr:.2f}",
                            f"{pc_corr:.2f}",
                            f"{tc_corr:.2f}"
                        ],
                        "Units": [
                            "psia",
                            "°R",
                            "psia",
                            "°R"
                        ]
                    })
                    
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    if mol_co2 + mol_h2s > 0.01:
                        st.info("ℹ️ Wichert-Aziz correction applied for acid gases")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with tab2:
        st.subheader("C7+ Fraction Critical Properties")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sg_c7 = st.number_input("C7+ Specific Gravity", min_value=0.7, max_value=1.2, value=0.85, step=0.01)
            mw_c7 = st.number_input("C7+ Molecular Weight (lb/lbmol)", min_value=50.0, max_value=500.0, value=120.0, step=10.0)
            
            calc_button_c7 = st.button("🔬 Calculate", key="c7_crit", use_container_width=True)
        
        with col2:
            if calc_button_c7:
                try:
                    # Calculate C7+ critical properties
                    pc_c7 = pc_c7_plus(sg_c7, mw_c7)
                    tc_c7 = tc_c7_plus(sg_c7, mw_c7)
                    
                    # Estimate boiling point
                    tb = tb_degr_correlation(tc_c7)
                    
                    results_df = pd.DataFrame({
                        "Property": [
                            "Critical Pressure",
                            "Critical Temperature",
                            "Normal Boiling Point (estimated)"
                        ],
                        "Value": [
                            f"{pc_c7:.2f}",
                            f"{tc_c7:.2f}",
                            f"{tb:.2f}"
                        ],
                        "Units": [
                            "psia",
                            "°R",
                            "°R"
                        ]
                    })
                    
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    st.success("✅ Riazi-Daubert correlation applied")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


# Complete Enhanced Uncertainty Analysis Function
# Replace lines 1707-1966 in app.py with this

def uncertainty_analysis():
    """Perform enhanced uncertainty analysis with Monte Carlo simulation."""
    st.header("🎲 Uncertainty Analysis")
    st.markdown("Analyze PVT properties with parameter uncertainties using Monte Carlo simulation with advanced visualizations.")
    
    # Get user's preferred units
    pressure_unit, temperature_unit = get_user_units()
    st.info(f"📏 **Current Units:** Pressure = {pressure_unit}, Temperature = {temperature_unit} (Change in sidebar)")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Input mode selector  
        input_mode = st.radio("**Input Method:**", ["Mean ± Std Dev", "Mean ± Range"], horizontal=True)
        use_std_dev = (input_mode == "Mean ± Std Dev")
        
        # Distribution type
        dist_type_display = st.selectbox("**Distribution Type:**", ["Normal (Gaussian)", "Uniform", "Triangular"])
        dist_type = dist_type_display.split()[0].lower()
        
        # Reservoir conditions
        st.markdown("**Reservoir Conditions**")
        pmean_default = 3000.0 if pressure_unit == 'psia' else convert_pressure_from_psia(3000.0, pressure_unit)
        puncert_default = 50.0 if pressure_unit == 'psia' else convert_pressure_from_psia(50.0, pressure_unit)
        
        pressure_mean_input = st.number_input(f"Pressure - Mean ({pressure_unit})", min_value=0.0, value=pmean_default, step=100.0 if pressure_unit in ['psia', 'psig'] else 1.0)
        if use_std_dev:
            pressure_uncert_input = st.number_input(f"Pressure - Std Dev ({pressure_unit})", min_value=0.0, value=puncert_default, step=10.0 if pressure_unit in ['psia', 'psig'] else 1.0)
        else:
            pressure_uncert_input = st.number_input(f"Pressure - ± Range ({pressure_unit})", min_value=0.0, value=puncert_default*3, step=50.0 if pressure_unit in ['psia', 'psig'] else 1.0)
        
        pressure_mean = convert_pressure_to_psia(pressure_mean_input, pressure_unit)
        pressure_std = convert_pressure_to_psia(pressure_uncert_input, pressure_unit) if use_std_dev else convert_pressure_to_psia(pressure_uncert_input/3, pressure_unit)
        
        tmean_default = 180.0 if temperature_unit == '°F' else convert_temperature_from_fahrenheit(180.0, temperature_unit)
        tuncert_default = 5.0 if temperature_unit == '°F' else 5.0 * 5/9
        
        temperature_mean_input = st.number_input(f"Temperature - Mean ({temperature_unit})", value=tmean_default, step=10.0 if temperature_unit == '°F' else 5.0)
        if use_std_dev:
            temperature_uncert_input = st.number_input(f"Temperature - Std Dev ({temperature_unit})", min_value=0.0, value=tuncert_default, step=1.0)
        else:
            temperature_uncert_input = st.number_input(f"Temperature - ± Range ({temperature_unit})", min_value=0.0, value=tuncert_default*3, step=1.0)
        
        temperature_mean = convert_temperature_to_fahrenheit(temperature_mean_input, temperature_unit)
        temperature_std = temperature_uncert_input if (use_std_dev and temperature_unit == '°F') else (temperature_uncert_input * 1.8 if use_std_dev else temperature_uncert_input * 1.8 / 3)
        
        # Standard conditions
        st.markdown("**Standard Conditions**")
        psc_default = 14.696 if pressure_unit == 'psia' else convert_pressure_from_psia(14.696, pressure_unit)
        tsc_default = 60.0 if temperature_unit == '°F' else convert_temperature_from_fahrenheit(60.0, temperature_unit)
        
        psc_input = st.number_input(f"Psc ({pressure_unit})", value=psc_default, step=0.001 if pressure_unit in ['psia', 'psig'] else 0.01, format="%.3f")
        tsc_input = st.number_input(f"Tsc ({temperature_unit})", value=tsc_default, step=1.0)
        
        psc = convert_pressure_to_psia(psc_input, pressure_unit)
        tsc = convert_temperature_to_fahrenheit(tsc_input, temperature_unit)
        
        # Gas properties with uncertainties
        st.markdown("**Gas Properties**")
        gas_grav_mean = st.number_input("Gas Gravity - Mean", min_value=0.5, max_value=1.5, value=0.65, step=0.01)
        if use_std_dev:
            gas_grav_std = st.number_input("Gas Gravity - Std Dev", min_value=0.0, value=0.01, step=0.001, format="%.4f")
        else:
            gas_grav_range = st.number_input("Gas Gravity - ± Range", min_value=0.0, value=0.03, step=0.01, format="%.3f")
            gas_grav_std = gas_grav_range / 3
        
        # Composition with uncertainty
        with st.expander("Composition Uncertainty (Optional)", expanded=False):
            enable_comp_uncert = st.checkbox("Enable composition uncertainty", value=False)
            if enable_comp_uncert:
                col_n2_val, col_n2_std = st.columns(2)
                with col_n2_val:
                    mol_n2_mean = st.number_input("N₂ - Mean", min_value=0.0, max_value=1.0, value=0.02, step=0.01, format="%.3f", key='n2_mean')
                with col_n2_std:
                    if use_std_dev:
                        mol_n2_std = st.number_input("N₂ - Std Dev", min_value=0.0, value=0.005, step=0.001, format="%.4f", key='n2_std')
                    else:
                        mol_n2_range = st.number_input("N₂ - ±Range", min_value=0.0, value=0.01, step=0.001, format="%.4f", key='n2_range')
                        mol_n2_std = mol_n2_range / 3
                
                col_co2_val, col_co2_std = st.columns(2)
                with col_co2_val:
                    mol_co2_mean = st.number_input("CO₂ - Mean", min_value=0.0, max_value=1.0, value=0.01, step=0.01, format="%.3f", key='co2_mean')
                with col_co2_std:
                    if use_std_dev:
                        mol_co2_std = st.number_input("CO₂ - Std Dev", min_value=0.0, value=0.005, step=0.001, format="%.4f", key='co2_std')
                    else:
                        mol_co2_range = st.number_input("CO₂ - ±Range", min_value=0.0, value=0.01, step=0.001, format="%.4f", key='co2_range')
                        mol_co2_std = mol_co2_range / 3
                
                col_h2s_val, col_h2s_std = st.columns(2)
                with col_h2s_val:
                    mol_h2s_mean = st.number_input("H₂S - Mean", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.3f", key='h2s_mean')
                with col_h2s_std:
                    if use_std_dev:
                        mol_h2s_std = st.number_input("H₂S - Std Dev", min_value=0.0, value=0.0, step=0.001, format="%.4f", key='h2s_std')
                    else:
                        mol_h2s_range = st.number_input("H₂S - ±Range", min_value=0.0, value=0.0, step=0.001, format="%.4f", key='h2s_range')
                        mol_h2s_std = mol_h2s_range / 3
            else:
                mol_n2_mean, mol_n2_std = 0.02, 0.0
                mol_co2_mean, mol_co2_std = 0.01, 0.0
                mol_h2s_mean, mol_h2s_std = 0.0, 0.0
        
        # Monte Carlo settings
        st.markdown("**Simulation Settings**")
        n_iterations = st.slider("Number of Iterations", min_value=100, max_value=10000, value=1000, step=100)
        
        # Visualization options
        with st.expander("Visualization Options", expanded=False):
            show_box_plots = st.checkbox("Show Box Plots", value=True)
            show_tornado = st.checkbox("Show Tornado Chart (Sensitivity)", value=True)
            show_dist_plots = st.checkbox("Show Distribution Plots with CDF", value=False)
        
        run_button = st.button("🎲 Run Monte Carlo Simulation", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Uncertainty Results")
        
        if run_button:
            try:
                with st.spinner(f'Running Monte Carlo simulation with {n_iterations} iterations...'):
                    # Set random seed for reproducibility
                    np.random.seed(42)
                    
                    # Generate samples for all inputs
                    pressures = generate_samples(pressure_mean, pressure_std, n_iterations, dist_type)
                    temperatures = generate_samples(temperature_mean, temperature_std, n_iterations, dist_type)
                    gas_gravs = generate_samples(gas_grav_mean, gas_grav_std, n_iterations, dist_type)
                    
                    # Generate composition samples if enabled
                    if enable_comp_uncert:
                        mol_n2s = generate_samples(mol_n2_mean, mol_n2_std, n_iterations, dist_type)
                        mol_co2s = generate_samples(mol_co2_mean, mol_co2_std, n_iterations, dist_type)
                        mol_h2ss = generate_samples(mol_h2s_mean, mol_h2s_std, n_iterations, dist_type)
                        # Clip to [0, 1] and normalize if sum > 1
                        mol_n2s = np.clip(mol_n2s, 0, 1)
                        mol_co2s = np.clip(mol_co2s, 0, 1)
                        mol_h2ss = np.clip(mol_h2ss, 0, 1)
                    else:
                        mol_n2s = np.full(n_iterations, mol_n2_mean)
                        mol_co2s = np.full(n_iterations, mol_co2_mean)
                        mol_h2ss = np.full(n_iterations, mol_h2s_mean)
                    
                    # Clip to valid ranges
                    pressures = np.clip(pressures, 100, 20000)
                    gas_gravs = np.clip(gas_gravs, 0.5, 1.5)
                    
                    # Calculate properties for each iteration
                    z_factors = []
                    bg_values = []
                    mu_values = []
                    cg_values = []
                    gas_grad_values = []
                    pp_values = []
                    
                    progress_bar = st.progress(0)
                    for i in range(n_iterations):
                        p = pressures[i]
                        t = temperatures[i]
                        gg = gas_gravs[i]
                        n2 = mol_n2s[i]
                        co2 = mol_co2s[i]
                        h2s = mol_h2ss[i]
                        
                        # Calculate pseudo-critical properties
                        pc = p_crit(gg, n2, co2, h2s, 0.0, 0.85, 120.0)
                        tc = t_crit(gg, n2, co2, h2s, 0.0, 0.85, 120.0)
                        
                        ppr = p / pc
                        tpr = (t + 459.67) / tc
                        
                        try:
                            z = z_gas(ppr, tpr)
                            if isinstance(z, complex):
                                z = abs(z)
                            bg = bg_rv_per_scv(p, t, psc, tsc, pc, tc)
                            mu = mu_gas(p, t, pc, tc, gg, n2, co2, h2s)
                            cg = c_gas(p, pc, tpr)
                            gg_val = gas_grad(p, t, gg, pc, tc)
                            pp = pseudo_pressure(p, t, pc, tc, gg, n2, co2, h2s)
                            
                            z_factors.append(z)
                            bg_values.append(bg)
                            mu_values.append(mu)
                            cg_values.append(cg)
                            gas_grad_values.append(gg_val)
                            pp_values.append(pp)
                        except:
                            z_factors.append(np.nan)
                            bg_values.append(np.nan)
                            mu_values.append(np.nan)
                            cg_values.append(np.nan)
                            gas_grad_values.append(np.nan)
                            pp_values.append(np.nan)
                        
                        if (i + 1) % 100 == 0:
                            progress_bar.progress((i + 1) / n_iterations)
                    
                    progress_bar.empty()
                    
                    # Convert to arrays and remove NaNs for statistics
                    z_factors = np.array(z_factors)
                    bg_values = np.array(bg_values)
                    mu_values = np.array(mu_values)
                    cg_values = np.array(cg_values)
                    gas_grad_values = np.array(gas_grad_values)
                    pp_values = np.array(pp_values)
                    
                    valid_mask = ~(np.isnan(z_factors) | np.isnan(bg_values) | np.isnan(mu_values))
                    n_valid = np.sum(valid_mask)
                    
                    if n_valid < n_iterations * 0.9:
                        st.warning(f"⚠️ {n_iterations - n_valid} iterations failed. Results based on {n_valid} successful runs.")
                    
                    # Calculate statistics
                    def calc_stats(data):
                        data_clean = data[~np.isnan(data)]
                        if len(data_clean) == 0:
                            return {
                                'mean': np.nan, 'std': np.nan, 'cov': np.nan,
                                'min': np.nan, 'max': np.nan,
                                'p10': np.nan, 'p50': np.nan, 'p90': np.nan
                            }
                        return {
                            'mean': np.mean(data_clean),
                            'std': np.std(data_clean),
                            'cov': (np.std(data_clean) / np.mean(data_clean) * 100) if np.mean(data_clean) != 0 else 0,
                            'min': np.min(data_clean),
                            'max': np.max(data_clean),
                            'p10': np.percentile(data_clean, 10),
                            'p50': np.percentile(data_clean, 50),
                            'p90': np.percentile(data_clean, 90)
                        }
                    
                    z_stats = calc_stats(z_factors)
                    bg_stats = calc_stats(bg_values)
                    mu_stats = calc_stats(mu_values)
                    cg_stats = calc_stats(cg_values)
                    gg_stats = calc_stats(gas_grad_values)
                    pp_stats = calc_stats(pp_values)
                    
                    # Display summary statistics with CoV color coding
                    st.markdown("### 📊 Statistical Summary")
                    
                    def format_cov(cov):
                        if cov < 5:
                            return f"🟢 {cov:.1f}%"
                        elif cov < 15:
                            return f"🟡 {cov:.1f}%"
                        else:
                            return f"🔴 {cov:.1f}%"
                    
                    summary_df = pd.DataFrame({
                        'Property': ['Z-factor', 'Bg (RV/SCV)', 'Viscosity (cP)', 'Compressibility (1/psi)', 'Gas Gradient (psi/ft)', 'm(p) (psia²/cP)'],
                        'P10': [
                            f"{z_stats['p10']:.6f}",
                            f"{bg_stats['p10']:.6f}",
                            f"{mu_stats['p10']:.6f}",
                            f"{cg_stats['p10']:.3e}",
                            f"{gg_stats['p10']:.6f}",
                            f"{pp_stats['p10']:.3e}"
                        ],
                        'P50 (Median)': [
                            f"{z_stats['p50']:.6f}",
                            f"{bg_stats['p50']:.6f}",
                            f"{mu_stats['p50']:.6f}",
                            f"{cg_stats['p50']:.3e}",
                            f"{gg_stats['p50']:.6f}",
                            f"{pp_stats['p50']:.3e}"
                        ],
                        'P90': [
                            f"{z_stats['p90']:.6f}",
                            f"{bg_stats['p90']:.6f}",
                            f"{mu_stats['p90']:.6f}",
                            f"{cg_stats['p90']:.3e}",
                            f"{gg_stats['p90']:.6f}",
                            f"{pp_stats['p90']:.3e}"
                        ],
                        'Mean ± Std': [
                            f"{z_stats['mean']:.6f} ± {z_stats['std']:.6f}",
                            f"{bg_stats['mean']:.6f} ± {bg_stats['std']:.6f}",
                            f"{mu_stats['mean']:.6f} ± {mu_stats['std']:.6f}",
                            f"{cg_stats['mean']:.3e} ± {cg_stats['std']:.3e}",
                            f"{gg_stats['mean']:.6f} ± {gg_stats['std']:.6f}",
                            f"{pp_stats['mean']:.3e} ± {pp_stats['std']:.3e}"
                        ],
                        'CoV': [
                            format_cov(z_stats['cov']),
                            format_cov(bg_stats['cov']),
                            format_cov(mu_stats['cov']),
                            format_cov(cg_stats['cov']),
                            format_cov(gg_stats['cov']),
                            format_cov(pp_stats['cov'])
                        ]
                    })
                    
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    st.caption("CoV = Coefficient of Variation (Std/Mean × 100%). 🟢 Low (<5%), 🟡 Moderate (5-15%), 🔴 High (>15%)")
                    
                    # Box plots
                    if show_box_plots:
                        st.markdown("### 📦 Distribution Box Plots")
                        data_dict = {
                            'Z-factor': z_factors[valid_mask],
                            'Bg (RV/SCV)': bg_values[valid_mask],
                            'Viscosity (cP)': mu_values[valid_mask],
                            'Gas Gradient (psi/ft)': gas_grad_values[valid_mask]
                        }
                        fig_box = create_box_plots(data_dict, list(data_dict.keys()))
                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Tornado chart (sensitivity analysis)
                    if show_tornado:
                        st.markdown("### 🌪️ Sensitivity Analysis (Tornado Chart)")
                        
                        # Prepare input/output dictionaries
                        inputs_dict = {
                            'Pressure': pressures[valid_mask],
                            'Temperature': temperatures[valid_mask],
                            'Gas Gravity': gas_gravs[valid_mask]
                        }
                        if enable_comp_uncert:
                            inputs_dict['N₂'] = mol_n2s[valid_mask]
                            inputs_dict['CO₂'] = mol_co2s[valid_mask]
                            inputs_dict['H₂S'] = mol_h2ss[valid_mask]
                        
                        outputs_dict = {
                            'Z-factor': z_factors[valid_mask],
                            'Bg': bg_values[valid_mask],
                            'Viscosity': mu_values[valid_mask],
                            'Compressibility': cg_values[valid_mask]
                        }
                        
                        sensitivity_df = calculate_tornado_data(inputs_dict, outputs_dict, {})
                        
                        # Show tornado for most important output (Z-factor)
                        fig_tornado = create_tornado_chart(sensitivity_df, 'Z-factor', top_n=min(8, len(inputs_dict)))
                        st.plotly_chart(fig_tornado, use_container_width=True)
                        
                        with st.expander("View Sensitivity Data", expanded=False):
                            st.dataframe(sensitivity_df.sort_values('Impact', ascending=False), use_container_width=True)
                    
                    # Distribution plots with CDF
                    if show_dist_plots:
                        st.markdown("### 📈 Distribution Plots with CDF")
                        col_dist1, col_dist2 = st.columns(2)
                        with col_dist1:
                            fig_z_dist = create_distribution_plot_with_cdf(z_factors[valid_mask], 'Z-factor')
                            st.plotly_chart(fig_z_dist, use_container_width=True)
                        with col_dist2:
                            fig_bg_dist = create_distribution_plot_with_cdf(bg_values[valid_mask], 'Bg (RV/SCV)')
                            st.plotly_chart(fig_bg_dist, use_container_width=True)
                    
                    # Interpretation
                    st.markdown("### 💡 Interpretation")
                    
                    z_range_pct = (z_stats['p90'] - z_stats['p10']) / z_stats['p50'] * 100
                    bg_range_pct = (bg_stats['p90'] - bg_stats['p10']) / bg_stats['p50'] * 100
                    
                    st.info(
                        f"**Uncertainty Ranges (P10-P90):**\n\n"
                        f"- **Z-factor**: {z_stats['p10']:.4f} to {z_stats['p90']:.4f} "
                        f"({z_range_pct:.1f}% spread around median)\n"
                        f"- **Bg**: {bg_stats['p10']:.6f} to {bg_stats['p90']:.6f} RV/SCV "
                        f"({bg_range_pct:.1f}% spread around median)\n"
                        f"- **μ**: {mu_stats['p10']:.4f} to {mu_stats['p90']:.4f} cP\n"
                        f"- **cg**: {cg_stats['p10']:.3e} to {cg_stats['p90']:.3e} 1/psi\n"
                        f"- **Gas Gradient**: {gg_stats['p10']:.4f} to {gg_stats['p90']:.4f} psi/ft\n"
                        f"- **m(p)**: {pp_stats['p10']:.3e} to {pp_stats['p90']:.3e} psia²/cP"
                    )
                    
                    # Key insights
                    high_uncertainty_props = []
                    for name, stats in [('Z-factor', z_stats), ('Bg', bg_stats), ('Viscosity', mu_stats)]:
                        if stats['cov'] > 10:
                            high_uncertainty_props.append(f"{name} (CoV={stats['cov']:.1f}%)")
                    
                    if high_uncertainty_props:
                        st.warning(f"⚠️ **High uncertainty detected in:** {', '.join(high_uncertainty_props)}. "
                                  f"Consider refining input measurements or using probabilistic analysis for critical decisions.")
                    else:
                        st.success("✅ **Low to moderate uncertainty** across all properties. Results are relatively well-constrained.")
                    
                    # Download results
                    results_df = pd.DataFrame({
                        'Iteration': range(1, n_iterations + 1),
                        'Pressure (psia)': pressures,
                        'Temperature (°F)': temperatures,
                        'Gas Gravity': gas_gravs,
                        'N₂ (mol frac)': mol_n2s,
                        'CO₂ (mol frac)': mol_co2s,
                        'H₂S (mol frac)': mol_h2ss,
                        'Z-factor': z_factors,
                        'Bg (RV/SCV)': bg_values,
                        'Viscosity (cP)': mu_values,
                        'Compressibility (1/psi)': cg_values,
                        'Gas Gradient (psi/ft)': gas_grad_values,
                        'm(p) (psia²/cP)': pp_values
                    })
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Monte Carlo Results (CSV)",
                        data=csv,
                        file_name=f"monte_carlo_uncertainty_{n_iterations}_iterations.csv",
                        mime="text/csv"
                    )
                    
                    st.success(f"✅ Monte Carlo simulation completed! {n_valid} of {n_iterations} iterations successful.")
                    
            except Exception as e:
                st.error(f"❌ Error during simulation: {str(e)}")
                import traceback
                st.code(traceback.format_exc())



if __name__ == "__main__":
    main()

