"""
Gas PVT Advanced Analysis - Streamlit Web Application

Advanced engineering workflows: Z-factor comparison, material balance,
wellbore hydraulics, hydrate prediction, gas quality, and batch processing.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import math as _m

from gas_pvt_correlations import (
    z_gas, bg_rv_per_scv, mu_gas, c_gas, gas_grad,
    p_crit, t_crit,
    z_gas_dak, z_gas_pmc, pmc_pseudo_criticals,
    gross_heating_value, net_heating_value, wobbe_index, specific_energy,
    dew_point_nemeth_kennedy, cricondentherm_estimate,
)
from material_balance import (
    pz_linear_regression, extrapolate_recovery, havlena_odeh_gas
)
from wellbore_hydraulics import cullender_smith, gas_velocity, erosional_velocity
from hydrate_prediction import (
    katz_hydrate_temperature, hydrate_envelope, subcooling,
    hammerschmidt_concentration, inhibitor_rate
)

# Import base pages and shared constants from core app
import app as _core
from app import (
    user_guide_and_background,
    single_point_calculator,
    pressure_profile,
    complete_pvt_table,
    water_gas_properties,
    critical_properties,
    uncertainty_analysis,
    SHELL_RED, SHELL_YELLOW, SHELL_DARK, SHELL_CSS,
    PLOT_FONT, SHELL_PECTEN_URL, SHELL_LINE_COLORS, AXIS_STYLE, LAYOUT_STYLE,
)


# ============================================================================
#  UNIT CONVERSION (same as core app)
# ============================================================================

class UnitConverter:
    @staticmethod
    def celsius_to_fahrenheit(temp_c):
        return temp_c * 9/5 + 32

    @staticmethod
    def fahrenheit_to_celsius(temp_f):
        return (temp_f - 32) * 5/9

    @staticmethod
    def psia_to_mpa(p):
        return p * 0.00689476

    @staticmethod
    def mpa_to_psia(p):
        return p / 0.00689476

    @staticmethod
    def psia_to_kpa(p):
        return p * 6.89476

    @staticmethod
    def kpa_to_psia(p):
        return p / 6.89476

    @staticmethod
    def psia_to_bar(p):
        return p * 0.0689476

    @staticmethod
    def bar_to_psia(p):
        return p / 0.0689476

    @staticmethod
    def psia_to_psig(p):
        return p - 14.696

    @staticmethod
    def psig_to_psia(p):
        return p + 14.696


uc = UnitConverter()


def get_user_units():
    return st.session_state.get('pressure_unit', 'psia'), st.session_state.get('temperature_unit', '°F')


def convert_pressure_to_psia(pressure, unit):
    if unit == 'psia': return pressure
    if unit == 'psig': return uc.psig_to_psia(pressure)
    if unit == 'MPa':  return uc.mpa_to_psia(pressure)
    if unit == 'kPa':  return uc.kpa_to_psia(pressure)
    if unit == 'bar':  return uc.bar_to_psia(pressure)
    return pressure


def convert_pressure_from_psia(pressure, unit):
    if unit == 'psia': return pressure
    if unit == 'psig': return uc.psia_to_psig(pressure)
    if unit == 'MPa':  return uc.psia_to_mpa(pressure)
    if unit == 'kPa':  return uc.psia_to_kpa(pressure)
    if unit == 'bar':  return uc.psia_to_bar(pressure)
    return pressure


def convert_temperature_to_fahrenheit(temp, unit):
    if unit == '°F': return temp
    if unit == '°C': return uc.celsius_to_fahrenheit(temp)
    return temp


def convert_temperature_from_fahrenheit(temp, unit):
    if unit == '°F': return temp
    if unit == '°C': return uc.fahrenheit_to_celsius(temp)
    return temp


# Constants and styling imported from app module above


# ============================================================================
#  PAGE FUNCTIONS
# ============================================================================

def zfactor_comparison():
    """Compare Hall-Yarborough, DAK, and Piper-McCain-Corredor Z-factor methods."""
    st.header("Z-factor Correlation Comparison")
    st.markdown(
        "Overlay **Hall-Yarborough (1973)**, **Dranchuk-Abou-Kassem (1975)**, "
        "and **Piper-McCain-Corredor (2012)** on the same plot to assess "
        "sensitivity to correlation choice."
    )

    pressure_unit, temperature_unit = get_user_units()
    st.info(f"📏 **Current Units:** Pressure = {pressure_unit}, Temperature = {temperature_unit}")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Input Parameters")
        default_p_min = 100.0 if pressure_unit == 'psia' else convert_pressure_from_psia(100.0, pressure_unit)
        default_p_max = 10000.0 if pressure_unit == 'psia' else convert_pressure_from_psia(10000.0, pressure_unit)
        p_min_in = st.number_input(f"P min ({pressure_unit})", value=default_p_min, step=100.0)
        p_max_in = st.number_input(f"P max ({pressure_unit})", value=default_p_max, step=100.0)
        default_t = 200.0 if temperature_unit == '°F' else convert_temperature_from_fahrenheit(200.0, temperature_unit)
        temp_in = st.number_input(f"Temperature ({temperature_unit})", value=default_t, step=10.0)
        gas_grav = st.number_input("Gas Specific Gravity", value=0.65, step=0.01, min_value=0.5, max_value=1.5)
        mol_n2 = st.number_input("N₂ (mol frac)", value=0.0, step=0.01, format="%.3f")
        mol_co2 = st.number_input("CO₂ (mol frac)", value=0.0, step=0.01, format="%.3f")
        mol_h2s = st.number_input("H₂S (mol frac)", value=0.0, step=0.01, format="%.3f")
        n_pts = st.slider("Points", 20, 200, 50)
        calc = st.button("📈 Compare", type="primary", use_container_width=True)

    with col2:
        if calc:
            p_min = convert_pressure_to_psia(p_min_in, pressure_unit)
            p_max = convert_pressure_to_psia(p_max_in, pressure_unit)
            temp_f = convert_temperature_to_fahrenheit(temp_in, temperature_unit)

            pc_s = p_crit(gas_grav, mol_n2, mol_co2, mol_h2s, 0, 0.85, 120)
            tc_s = t_crit(gas_grav, mol_n2, mol_co2, mol_h2s, 0, 0.85, 120)
            pc_pmc, tc_pmc = pmc_pseudo_criticals(gas_grav, mol_n2, mol_co2, mol_h2s)

            pressures = np.linspace(p_min, p_max, n_pts)
            z_hy, z_dak, z_pmc = [], [], []
            t_r = temp_f + 459.67
            for p in pressures:
                ppr_s = p / pc_s
                tpr_s = t_r / tc_s
                z_hy.append(z_gas(ppr_s, tpr_s))
                z_dak.append(z_gas_dak(ppr_s, tpr_s))
                ppr_p = p / pc_pmc
                tpr_p = t_r / tc_pmc
                z_pmc.append(z_gas_pmc(ppr_p, tpr_p))

            p_hy = [p for p, z in zip(pressures, z_hy) if not _m.isnan(z)]
            z_hy_clean = [z for z in z_hy if not _m.isnan(z)]
            p_dak = [p for p, z in zip(pressures, z_dak) if not _m.isnan(z)]
            z_dak_clean = [z for z in z_dak if not _m.isnan(z)]
            p_pmc = [p for p, z in zip(pressures, z_pmc) if not _m.isnan(z)]
            z_pmc_clean = [z for z in z_pmc if not _m.isnan(z)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=p_hy, y=z_hy_clean, mode='lines',
                                     name='Hall-Yarborough',
                                     line=dict(color=SHELL_RED, width=2.5)))
            fig.add_trace(go.Scatter(x=p_dak, y=z_dak_clean, mode='lines',
                                     name='Dranchuk-Abou-Kassem',
                                     line=dict(color='#FBCE07', width=2.5)))
            fig.add_trace(go.Scatter(x=p_pmc, y=z_pmc_clean, mode='lines',
                                     name='Piper-McCain-Corredor',
                                     line=dict(color='#009639', width=2.5)))
            fig.update_layout(
                title="Z-factor vs Pressure — Method Comparison",
                xaxis_title=f"Pressure ({pressure_unit})",
                yaxis_title="Z-factor",
                height=500, **LAYOUT_STYLE,
            )
            fig.update_xaxes(**AXIS_STYLE)
            fig.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig, use_container_width=True)

            n_dak_nan = sum(1 for z in z_dak if _m.isnan(z))
            n_pmc_nan = sum(1 for z in z_pmc if _m.isnan(z))
            if n_dak_nan > 0:
                st.warning(f"⚠️ DAK correlation did not converge for {n_dak_nan} points "
                          f"at high Ppr (outside valid envelope). These are omitted from the plot.")
            if n_pmc_nan > 0:
                st.warning(f"⚠️ PMC correlation did not converge for {n_pmc_nan} points.")

            def _fmt_z(z):
                return f"{z:.6f}" if not _m.isnan(z) else "–"
            def _fmt_dev(z_alt, z_ref):
                if _m.isnan(z_alt) or _m.isnan(z_ref) or z_ref == 0:
                    return "–"
                return f"{(z_alt - z_ref) / z_ref * 100:.3f}"

            dev_df = pd.DataFrame({
                "Pressure (psia)": [f"{p:.0f}" for p in pressures],
                "Z (HY)": [_fmt_z(z) for z in z_hy],
                "Z (DAK)": [_fmt_z(z) for z in z_dak],
                "Z (PMC)": [_fmt_z(z) for z in z_pmc],
                "DAK vs HY (%)": [_fmt_dev(zd, zh) for zh, zd in zip(z_hy, z_dak)],
                "PMC vs HY (%)": [_fmt_dev(zp, zh) for zh, zp in zip(z_hy, z_pmc)],
            })
            st.dataframe(dev_df, use_container_width=True, hide_index=True)
            csv = dev_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "z_comparison.csv", "text/csv")


def gas_quality_page():
    """Gas heating value, Wobbe index, and pipeline quality check."""
    st.header("Gas Quality & Wobbe Index")
    st.markdown("Estimate heating value, Wobbe Index, and compare against common pipeline tariff specs.")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Gas Composition")
        gas_grav = st.number_input("Gas Specific Gravity", value=0.65, step=0.01,
                                    min_value=0.5, max_value=1.5, key="gq_gg")
        mol_n2 = st.number_input("N₂ (mol frac)", value=0.0, step=0.01, format="%.3f", key="gq_n2")
        mol_co2 = st.number_input("CO₂ (mol frac)", value=0.0, step=0.01, format="%.3f", key="gq_co2")
        mol_h2s = st.number_input("H₂S (mol frac)", value=0.0, step=0.01, format="%.4f", key="gq_h2s")
        calc = st.button("🔥 Calculate", type="primary", use_container_width=True, key="gq_calc")

    with col2:
        if calc:
            ghv = gross_heating_value(gas_grav, mol_n2, mol_co2, mol_h2s)
            nhv = net_heating_value(ghv)
            wi = wobbe_index(ghv, gas_grav)
            se = specific_energy(ghv)

            results_df = pd.DataFrame({
                "Property": [
                    "Gross Heating Value (GHV)",
                    "Net Heating Value (NHV)",
                    "Wobbe Index",
                    "Specific Energy",
                    "Gas Specific Gravity",
                ],
                "Value": [f"{ghv:.1f}", f"{nhv:.1f}", f"{wi:.1f}", f"{se:.2f}", f"{gas_grav:.4f}"],
                "Units": ["BTU/SCF", "BTU/SCF", "BTU/SCF", "MJ/m³", "–"],
            })
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            st.markdown("### Pipeline Tariff Spec Check")
            specs = [
                ("GHV", ghv, 950, 1150, "BTU/SCF"),
                ("Wobbe Index", wi, 1310, 1390, "BTU/SCF"),
                ("CO₂", mol_co2 * 100, 0, 2, "mol%"),
                ("H₂S", mol_h2s * 1e6, 0, 4, "ppmv (equivalent)"),
                ("N₂", mol_n2 * 100, 0, 4, "mol%"),
            ]
            for name, val, lo, hi, unit in specs:
                if lo <= val <= hi:
                    st.success(f"✅ **{name}**: {val:.2f} {unit} — within typical spec ({lo}–{hi})")
                else:
                    st.warning(f"⚠️ **{name}**: {val:.2f} {unit} — outside typical spec ({lo}–{hi})")


def dew_point_page():
    """Screening-level dew point estimation."""
    st.header("Dew Point Estimation")
    st.markdown(
        "Screening-level estimate using Nemeth-Kennedy (1967) correlation. "
        "**Not a rigorous EOS calculation** — suitable for initial screening only."
    )

    col1, col2 = st.columns([1, 3])
    pressure_unit, temperature_unit = get_user_units()

    with col1:
        st.subheader("Composition")
        gas_grav = st.number_input("Gas Specific Gravity", value=0.75, step=0.01,
                                    min_value=0.55, max_value=1.2, key="dp_gg")
        mol_n2 = st.number_input("N₂ (mol frac)", value=0.0, step=0.01, format="%.3f", key="dp_n2")
        mol_co2 = st.number_input("CO₂ (mol frac)", value=0.02, step=0.01, format="%.3f", key="dp_co2")
        mol_h2s = st.number_input("H₂S (mol frac)", value=0.0, step=0.01, format="%.3f", key="dp_h2s")
        mol_c7 = st.number_input("C7+ (mol frac)", value=0.02, step=0.005, format="%.4f", key="dp_c7")
        default_tres = 200.0 if temperature_unit == '°F' else convert_temperature_from_fahrenheit(200.0, temperature_unit)
        t_res_in = st.number_input(f"Reservoir Temperature ({temperature_unit})", value=default_tres, step=10.0, key="dp_tres")
        calc = st.button("🔬 Estimate Dew Point", type="primary", use_container_width=True, key="dp_calc")

    with col2:
        if calc:
            t_res_f = convert_temperature_to_fahrenheit(t_res_in, temperature_unit)
            pd_est = dew_point_nemeth_kennedy(gas_grav, mol_n2, mol_co2, mol_h2s, mol_c7)
            ct_est = cricondentherm_estimate(gas_grav, mol_c7)

            pd_display = convert_pressure_from_psia(pd_est, pressure_unit)

            results_df = pd.DataFrame({
                "Property": [
                    "Estimated Dew Point Pressure",
                    "Estimated Cricondentherm",
                    "Reservoir Temperature",
                ],
                "Value": [
                    f"{pd_display:.0f}",
                    f"{ct_est:.0f}" if temperature_unit == '°F' else f"{(ct_est - 32)*5/9:.0f}",
                    f"{t_res_in:.0f}",
                ],
                "Units": [pressure_unit, temperature_unit, temperature_unit],
            })
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            if pd_est > 0:
                if t_res_f > ct_est:
                    st.success("✅ Reservoir temperature is **above** the cricondentherm — "
                              "single-phase gas expected, no liquid dropout in the reservoir.")
                else:
                    st.warning("⚠️ Reservoir temperature is **below** the cricondentherm — "
                              "retrograde condensation possible. Consider EOS characterisation.")
            else:
                st.info("ℹ️ Gas is very lean — no dew point predicted at these conditions.")

            st.caption("⚠️ This is a screening correlation with ±10–15% accuracy. "
                      "For investment decisions, use a tuned EOS (e.g., Peng-Robinson).")


def ogip_material_balance():
    """P/Z decline analysis and OGIP estimation."""
    st.header("OGIP / P÷Z Material Balance")
    st.markdown("Volumetric gas material balance — enter pressure decline data to estimate Original Gas In Place.")

    pressure_unit, temperature_unit = get_user_units()

    tab1, tab2 = st.tabs(["📝 Manual Entry", "📁 Upload CSV"])

    manual_data = None
    upload_data = None

    with tab1:
        st.markdown("Enter pressure vs cumulative production data below. First row should be initial conditions (Gp = 0).")
        default_data = pd.DataFrame({
            "Pressure (psia)": [3000.0, 2700.0, 2400.0, 2100.0, 1800.0],
            "Cum Production (BCF)": [0.0, 5.0, 12.0, 20.0, 30.0],
        })
        manual_data = st.data_editor(default_data, num_rows="dynamic", use_container_width=True, key="mb_editor")

    with tab2:
        uploaded = st.file_uploader("Upload CSV with 'Pressure' and 'Cum Production' columns",
                                     type=["csv"], key="mb_upload")
        if uploaded:
            upload_data = pd.read_csv(uploaded)
            st.dataframe(upload_data, use_container_width=True)

    data_source = upload_data if upload_data is not None else manual_data

    st.markdown("---")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Gas Properties")
        gas_grav = st.number_input("Gas Specific Gravity", value=0.65, step=0.01, key="mb_gg")
        mol_n2 = st.number_input("N₂ (mol frac)", value=0.0, step=0.01, format="%.3f", key="mb_n2")
        mol_co2 = st.number_input("CO₂ (mol frac)", value=0.0, step=0.01, format="%.3f", key="mb_co2")
        mol_h2s = st.number_input("H₂S (mol frac)", value=0.0, step=0.01, format="%.3f", key="mb_h2s")
        default_t = 200.0 if temperature_unit == '°F' else convert_temperature_from_fahrenheit(200.0, temperature_unit)
        temp_in = st.number_input(f"Temperature ({temperature_unit})", value=default_t, step=10.0, key="mb_t")
        default_pa = 500.0 if pressure_unit == 'psia' else convert_pressure_from_psia(500.0, pressure_unit)
        p_abandon_in = st.number_input(f"Abandonment Pressure ({pressure_unit})", value=default_pa, step=50.0, key="mb_pa")

        use_ho = st.checkbox("Apply Havlena-Odeh correction", key="mb_ho")
        if use_ho:
            cf = st.number_input("Formation compressibility (1/psi)", value=3.0e-6, format="%.2e", key="mb_cf")
            cw_val = st.number_input("Water compressibility (1/psi)", value=3.0e-6, format="%.2e", key="mb_cw")
            swi = st.number_input("Initial water saturation", value=0.25, step=0.05, key="mb_swi")

        calc = st.button("📊 Run Material Balance", type="primary", use_container_width=True, key="mb_calc")

    with col2:
        if calc and data_source is not None and len(data_source) >= 2:
            try:
                press_col = [c for c in data_source.columns if 'press' in c.lower()][0]
                gp_col = [c for c in data_source.columns if 'prod' in c.lower() or 'gp' in c.lower() or 'cum' in c.lower()][0]
            except IndexError:
                st.error("Could not find pressure and cumulative production columns. "
                         "Ensure column names contain 'Pressure' and 'Production' or 'Gp'.")
                return

            pressures = data_source[press_col].values.astype(float)
            gp = data_source[gp_col].values.astype(float)
            temp_f = convert_temperature_to_fahrenheit(temp_in, temperature_unit)
            p_abandon = convert_pressure_to_psia(p_abandon_in, pressure_unit)

            pc = p_crit(gas_grav, mol_n2, mol_co2, mol_h2s, 0, 0.85, 120)
            tc = t_crit(gas_grav, mol_n2, mol_co2, mol_h2s, 0, 0.85, 120)
            t_r = temp_f + 459.67
            z_vals = np.array([z_gas(p / pc, t_r / tc) for p in pressures])

            if use_ho:
                ogip, slope, intercept = havlena_odeh_gas(
                    pressures, z_vals, gp, cf, cw_val, swi)
                method = "Havlena-Odeh"
            else:
                ogip, slope, intercept = pz_linear_regression(pressures, z_vals, gp)
                method = "P/Z Linear"

            z_abandon = z_gas(p_abandon / pc, t_r / tc)
            gp_eur, rf = extrapolate_recovery(slope, intercept, p_abandon, z_abandon)

            st.subheader("Results")
            res_df = pd.DataFrame({
                "Parameter": ["OGIP", "EUR (at abandonment)", "Recovery Factor", "Method"],
                "Value": [f"{ogip:.2f}", f"{gp_eur:.2f}", f"{rf*100:.1f}%", method],
                "Units": ["BCF", "BCF", "", ""],
            })
            st.dataframe(res_df, use_container_width=True, hide_index=True)

            pz = pressures / z_vals
            gp_line = np.linspace(0, ogip * 1.05, 100)
            pz_line = intercept + slope * gp_line

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=gp, y=pz, mode='markers+lines',
                                     name='Data', marker=dict(size=10, color=SHELL_RED),
                                     line=dict(color=SHELL_RED, width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=gp_line, y=np.maximum(pz_line, 0),
                                     mode='lines', name='Regression',
                                     line=dict(color='#009639', width=2.5)))
            pz_ab = p_abandon / z_abandon
            fig.add_trace(go.Scatter(x=[gp_eur], y=[pz_ab], mode='markers',
                                     name=f'Abandon ({p_abandon:.0f} psia)',
                                     marker=dict(size=14, color='#FBCE07', symbol='diamond')))
            fig.update_layout(
                title=f"P/Z vs Cumulative Production — OGIP = {ogip:.1f} BCF",
                xaxis_title="Cumulative Production (BCF)",
                yaxis_title="P/Z (psia)",
                height=500, **LAYOUT_STYLE,
            )
            fig.update_xaxes(**AXIS_STYLE)
            fig.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig, use_container_width=True)

            detail_df = pd.DataFrame({
                "Pressure (psia)": [f"{p:.1f}" for p in pressures],
                "Z-factor": [f"{z:.6f}" for z in z_vals],
                "P/Z (psia)": [f"{pz_v:.1f}" for pz_v in pz],
                "Gp (BCF)": [f"{g:.2f}" for g in gp],
            })
            st.dataframe(detail_df, use_container_width=True, hide_index=True)


def wellbore_traverse_page():
    """Cullender-Smith wellbore pressure traverse."""
    st.header("Wellbore Pressure Traverse")
    st.markdown("Calculate bottomhole pressure from wellhead conditions (or vice versa) "
                "using the **Cullender-Smith** method for dry gas wells.")

    pressure_unit, temperature_unit = get_user_units()
    st.info(f"📏 **Current Units:** Pressure = {pressure_unit}, Temperature = {temperature_unit}")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Well Parameters")
        solve_dir = st.radio("Solve for:", ["BHP (given WHP)", "WHP (given BHP)"], key="wt_dir")
        solve_for = "bhp" if "BHP" in solve_dir else "whp"

        default_p = 2000.0 if pressure_unit == 'psia' else convert_pressure_from_psia(2000.0, pressure_unit)
        p_known_in = st.number_input(f"Known Pressure ({pressure_unit})", value=default_p, step=100.0, key="wt_p")
        default_t = 80.0 if temperature_unit == '°F' else convert_temperature_from_fahrenheit(80.0, temperature_unit)
        t_known_in = st.number_input(f"Temperature at known end ({temperature_unit})", value=default_t, step=10.0, key="wt_t")

        depth_ft = st.number_input("TVD (ft)", value=10000.0, step=500.0, min_value=100.0, key="wt_d")
        q_mscfd = st.number_input("Gas Rate (MSCF/d)", value=5000.0, step=500.0, min_value=0.0, key="wt_q")
        d_inches = st.number_input("Tubing ID (inches)", value=2.441, step=0.1, key="wt_di")
        roughness = st.number_input("Roughness (inches)", value=0.0006, format="%.4f", key="wt_rough")

        gas_grav = st.number_input("Gas Specific Gravity", value=0.65, step=0.01, key="wt_gg")
        t_grad = st.number_input("Temperature Gradient (°F/ft)", value=0.015, step=0.001, format="%.4f", key="wt_tg")

        calc = st.button("🛢️ Run Traverse", type="primary", use_container_width=True, key="wt_calc")

    with col2:
        if calc:
            p_known = convert_pressure_to_psia(p_known_in, pressure_unit)
            t_known = convert_temperature_to_fahrenheit(t_known_in, temperature_unit)
            pc_val = p_crit(gas_grav, 0, 0, 0, 0, 0.85, 120)
            tc_val = t_crit(gas_grav, 0, 0, 0, 0, 0.85, 120)

            result = cullender_smith(
                p_known, t_known, depth_ft, q_mscfd, d_inches,
                gas_grav, pc_val, tc_val,
                roughness=roughness, t_gradient=t_grad,
                n_segments=100, solve_for=solve_for,
            )

            solved_p = result.get('bhp', result.get('whp'))
            solved_label = "BHP" if solve_for == "bhp" else "WHP"
            solved_display = convert_pressure_from_psia(solved_p, pressure_unit)
            st.metric(f"Calculated {solved_label}", f"{solved_display:.1f} {pressure_unit}")

            fig = make_subplots(rows=1, cols=3,
                                subplot_titles=("Pressure vs Depth", "Temperature vs Depth", "Velocity vs Depth"),
                                horizontal_spacing=0.12)

            depths = result['depths']
            fig.add_trace(go.Scatter(x=result['pressures'], y=depths, mode='lines',
                                     line=dict(color=SHELL_RED, width=2.5), name='Pressure'),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=result['temperatures'], y=depths, mode='lines',
                                     line=dict(color='#FBCE07', width=2.5), name='Temperature'),
                          row=1, col=2)
            fig.add_trace(go.Scatter(x=result['velocities'], y=depths, mode='lines',
                                     line=dict(color='#009639', width=2.5), name='Gas Velocity'),
                          row=1, col=3)
            fig.add_trace(go.Scatter(x=result['erosional'], y=depths, mode='lines',
                                     line=dict(color=SHELL_RED, width=1.5, dash='dash'),
                                     name='Erosional Limit'),
                          row=1, col=3)

            fig.update_xaxes(title_text=f"Pressure ({pressure_unit})", row=1, col=1)
            fig.update_xaxes(title_text=f"Temperature ({temperature_unit})", row=1, col=2)
            fig.update_xaxes(title_text="Velocity (ft/s)", row=1, col=3)
            for c in range(1, 4):
                fig.update_yaxes(title_text="Depth (ft)", autorange="reversed", row=1, col=c)

            traverse_layout = {**LAYOUT_STYLE, "margin": dict(t=60, b=10, r=40)}
            fig.update_layout(height=500, showlegend=True, **traverse_layout)
            fig.update_xaxes(**AXIS_STYLE)
            fig.update_yaxes(**AXIS_STYLE)
            for ann in fig.layout.annotations:
                ann.y = ann.y + 0.04
                ann.font = dict(size=14, family="Futura Medium, Futura, sans-serif", color="#333")
            st.plotly_chart(fig, use_container_width=True)

            max_v = max(result['velocities'])
            min_erosional = min(result['erosional'])
            if max_v > min_erosional:
                st.warning(f"⚠️ Maximum velocity ({max_v:.1f} ft/s) exceeds erosional limit "
                          f"({min_erosional:.1f} ft/s) — consider larger tubing or lower rate.")
            else:
                st.success(f"✅ Maximum velocity ({max_v:.1f} ft/s) is within erosional limit "
                          f"({min_erosional:.1f} ft/s).")


def hydrate_prediction_page():
    """Hydrate equilibrium curve and inhibitor dosing."""
    st.header("Hydrate Formation Prediction")
    st.markdown("Katz gravity chart correlation for hydrate equilibrium, "
                "with Hammerschmidt inhibitor dosing calculator.")

    pressure_unit, temperature_unit = get_user_units()

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Operating Conditions")
        gas_grav = st.number_input("Gas Specific Gravity", value=0.65, step=0.01,
                                    min_value=0.55, max_value=1.0, key="hy_gg")

        default_p = 2000.0 if pressure_unit == 'psia' else convert_pressure_from_psia(2000.0, pressure_unit)
        default_t = 60.0 if temperature_unit == '°F' else convert_temperature_from_fahrenheit(60.0, temperature_unit)
        op_p_in = st.number_input(f"Operating Pressure ({pressure_unit})", value=default_p, step=100.0, key="hy_p")
        op_t_in = st.number_input(f"Operating Temperature ({temperature_unit})", value=default_t, step=5.0, key="hy_t")

        st.markdown("---")
        st.subheader("Inhibitor Dosing")
        inhibitor = st.selectbox("Inhibitor Type", ["Methanol", "MEG", "DEG", "TEG"], key="hy_inh")
        water_rate = st.number_input("Water Rate (bpd)", value=100.0, step=10.0, min_value=0.0, key="hy_wtr")

        calc = st.button("🧊 Analyze", type="primary", use_container_width=True, key="hy_calc")

    with col2:
        if calc:
            op_p = convert_pressure_to_psia(op_p_in, pressure_unit)
            op_t = convert_temperature_to_fahrenheit(op_t_in, temperature_unit)

            t_hyd = katz_hydrate_temperature(op_p, gas_grav)
            sc = subcooling(op_t, op_p, gas_grav)

            t_env, p_env = hydrate_envelope(gas_grav, p_min=100, p_max=10000)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_env, y=p_env, mode='lines',
                                     name='Hydrate Equilibrium',
                                     line=dict(color=SHELL_RED, width=3),
                                     fill='tozerox',
                                     fillcolor='rgba(221,29,33,0.1)'))
            fig.add_trace(go.Scatter(x=[op_t], y=[op_p], mode='markers',
                                     name='Operating Point',
                                     marker=dict(size=16, color='#009639', symbol='star')))

            fig.update_layout(
                title="Hydrate Equilibrium Envelope",
                xaxis_title=f"Temperature ({temperature_unit})",
                yaxis_title=f"Pressure ({pressure_unit})",
                yaxis_type="log",
                height=500, **LAYOUT_STYLE,
            )
            fig.update_xaxes(**AXIS_STYLE)
            fig.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig, use_container_width=True)

            if sc > 0:
                st.error(f"❄️ **HYDRATE RISK**: Operating {sc:.1f}°F below hydrate equilibrium "
                        f"temperature ({t_hyd:.1f}°F at {op_p:.0f} psia).")

                conc = hammerschmidt_concentration(sc, inhibitor.lower())
                inh_rate = inhibitor_rate(conc, water_rate, inhibitor.lower())

                st.markdown("### Inhibitor Requirements")
                inh_df = pd.DataFrame({
                    "Parameter": [
                        "Required Temperature Depression",
                        f"{inhibitor} Concentration",
                        f"{inhibitor} Injection Rate",
                    ],
                    "Value": [f"{sc:.1f}", f"{conc:.1f}", f"{inh_rate:.1f}"],
                    "Units": ["°F", "wt%", "gal/day"],
                })
                st.dataframe(inh_df, use_container_width=True, hide_index=True)
            else:
                st.success(f"✅ **SAFE**: Operating {abs(sc):.1f}°F above hydrate equilibrium "
                          f"temperature ({t_hyd:.1f}°F at {op_p:.0f} psia).")


def batch_upload_page():
    """Batch PVT calculation from uploaded CSV/Excel file."""
    st.header("Batch Upload — CSV / Excel")
    st.markdown("Upload a file with pressure/temperature data to generate PVT properties for every row.")

    pressure_unit, temperature_unit = get_user_units()

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Upload & Settings")
        uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], key="batch_file")

        gas_grav = st.number_input("Gas Specific Gravity", value=0.65, step=0.01, key="batch_gg")
        mol_n2 = st.number_input("N₂ (mol frac)", value=0.0, step=0.01, format="%.3f", key="batch_n2")
        mol_co2 = st.number_input("CO₂ (mol frac)", value=0.0, step=0.01, format="%.3f", key="batch_co2")
        mol_h2s = st.number_input("H₂S (mol frac)", value=0.0, step=0.01, format="%.3f", key="batch_h2s")
        psc = st.number_input("Psc (psia)", value=14.696, step=0.1, key="batch_psc")
        tsc = st.number_input("Tsc (°F)", value=60.0, step=1.0, key="batch_tsc")

        calc = st.button("⚡ Run Batch", type="primary", use_container_width=True, key="batch_calc")

    with col2:
        if uploaded:
            try:
                if uploaded.name.endswith('.xlsx'):
                    raw_df = pd.read_excel(uploaded)
                else:
                    raw_df = pd.read_csv(uploaded)
                st.markdown(f"**Loaded {len(raw_df)} rows, {len(raw_df.columns)} columns.**")
                st.dataframe(raw_df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
        else:
            st.info("Upload a CSV or Excel file to get started. Required columns: **Pressure**, **Temperature**. "
                    "Optional columns: gas_gravity, N2, CO2, H2S, TDS.")
            return

        if calc:
            col_map = {}
            for c in raw_df.columns:
                cl = c.lower().strip()
                if 'press' in cl or cl == 'p':
                    col_map['pressure'] = c
                elif 'temp' in cl or cl == 't':
                    col_map['temperature'] = c
                elif 'grav' in cl or cl == 'sg' or cl == 'gamma':
                    col_map['gas_gravity'] = c
                elif cl in ('n2', 'nitrogen'):
                    col_map['n2'] = c
                elif cl in ('co2', 'carbon dioxide'):
                    col_map['co2'] = c
                elif cl in ('h2s', 'hydrogen sulfide'):
                    col_map['h2s'] = c

            if 'pressure' not in col_map or 'temperature' not in col_map:
                st.error("Could not find Pressure and Temperature columns. "
                         "Ensure column names contain 'Pressure'/'P' and 'Temperature'/'T'.")
                return

            results = []
            progress = st.progress(0, text="Processing...")
            n = len(raw_df)

            for idx, row in raw_df.iterrows():
                p = float(row[col_map['pressure']])
                t = float(row[col_map['temperature']])
                gg = float(row.get(col_map.get('gas_gravity', ''), gas_grav)) if 'gas_gravity' in col_map else gas_grav
                n2 = float(row.get(col_map.get('n2', ''), mol_n2)) if 'n2' in col_map else mol_n2
                co2 = float(row.get(col_map.get('co2', ''), mol_co2)) if 'co2' in col_map else mol_co2
                h2s = float(row.get(col_map.get('h2s', ''), mol_h2s)) if 'h2s' in col_map else mol_h2s

                pc_val = p_crit(gg, n2, co2, h2s, 0, 0.85, 120)
                tc_val = t_crit(gg, n2, co2, h2s, 0, 0.85, 120)
                t_r = t + 459.67
                ppr = p / pc_val
                tpr = t_r / tc_val

                z = z_gas(ppr, tpr)
                bg = bg_rv_per_scv(p, t, psc, tsc, pc_val, tc_val)
                mu = mu_gas(p, t, pc_val, tc_val, gg, n2, co2, h2s)
                cg = c_gas(p, pc_val, tpr)
                gg_val = gas_grad(p, t, gg, pc_val, tc_val)

                results.append({
                    "Pressure (psia)": p,
                    "Temperature (°F)": t,
                    "Z-factor": z,
                    "Bg (RV/SCV)": bg,
                    "μg (cP)": mu,
                    "cg (1/psi)": cg,
                    "Gas Gradient (psi/ft)": gg_val,
                })

                progress.progress((idx + 1) / n, text=f"Row {idx+1}/{n}")

            progress.empty()
            out_df = pd.DataFrame(results)
            st.success(f"✅ Processed {len(out_df)} rows.")
            st.dataframe(out_df, use_container_width=True, hide_index=True)

            csv = out_df.to_csv(index=False)
            st.download_button("📥 Download Results CSV", csv, "batch_pvt_results.csv", "text/csv")


# ============================================================================
#  MAIN
# ============================================================================

def main():
    # Page configuration & styling
    st.set_page_config(
        page_title="Shell Gas PVT — Complete Analysis",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(SHELL_CSS, unsafe_allow_html=True)

    # Header
    st.markdown(f"""
    <div class="shell-banner">
        <img src="{SHELL_PECTEN_URL}" alt="Shell">
        <div>
            <h1>Gas PVT Analysis Tool (Prototype)</h1>
            <p>Core PVT calculations + advanced engineering workflows</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown(
        f'<div style="text-align:center;margin-bottom:1rem;">'
        f'<img src="{SHELL_PECTEN_URL}" width="100">'
        f'</div>',
        unsafe_allow_html=True
    )

    st.sidebar.title("⚙️ Unit Settings")

    st.sidebar.selectbox(
        "Pressure Units:",
        ['psia', 'psig', 'MPa', 'kPa', 'bar'],
        index=0, key='pressure_unit'
    )
    st.sidebar.selectbox(
        "Temperature Units:",
        ['°F', '°C'],
        index=0, key='temperature_unit'
    )

    st.sidebar.markdown("---")
    st.sidebar.title("📊 Analysis Type")

    analysis_type = st.sidebar.radio(
        "Select Analysis:",
        [
            "📚 User Guide & Background",
            "Single Point Calculator",
            "Pressure Profile",
            "Complete PVT Table",
            "Water-Gas Properties",
            "Critical Properties",
            "Uncertainty Analysis",
            "---",
            "Z-factor Comparison",
            "Gas Quality / Wobbe",
            "Dew Point Estimation",
            "---- ",
            "OGIP / P÷Z Material Balance",
            "Wellbore Pressure Traverse",
            "Hydrate Prediction",
            "----- ",
            "Batch Upload (CSV/Excel)",
        ],
        label_visibility="collapsed",
    )

    page_map = {
        # Core pages (from app.py)
        "📚 User Guide & Background": user_guide_and_background,
        "Single Point Calculator": single_point_calculator,
        "Pressure Profile": pressure_profile,
        "Complete PVT Table": complete_pvt_table,
        "Water-Gas Properties": water_gas_properties,
        "Critical Properties": critical_properties,
        "Uncertainty Analysis": uncertainty_analysis,
        # Advanced pages
        "Z-factor Comparison": zfactor_comparison,
        "Gas Quality / Wobbe": gas_quality_page,
        "Dew Point Estimation": dew_point_page,
        "OGIP / P÷Z Material Balance": ogip_material_balance,
        "Wellbore Pressure Traverse": wellbore_traverse_page,
        "Hydrate Prediction": hydrate_prediction_page,
        "Batch Upload (CSV/Excel)": batch_upload_page,
    }

    page_fn = page_map.get(analysis_type)
    if page_fn:
        page_fn()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "Complete gas PVT analysis toolkit:\n\n"
        "**Core:** Z-factor (HY), viscosity, Bg, compressibility, "
        "water content, critical properties, uncertainty\n\n"
        "**Advanced:** Z-factor comparison (HY/DAK/PMC), "
        "material balance, Cullender-Smith traverse, "
        "hydrate prediction, gas quality/Wobbe, batch processing\n\n"
        "Based on SPE Monograph Vol. 20 by Whitson & Brulé"
    )
    st.sidebar.caption("Built with Shell engineering standards")


if __name__ == "__main__":
    main()
