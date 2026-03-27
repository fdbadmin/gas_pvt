"""
Uncertainty Analysis Helper Functions

Provides Monte Carlo sampling, sensitivity analysis, and visualization
utilities for PVT uncertainty analysis.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


def generate_samples(mean: float, std: float, n: int, dist_type: str = "normal") -> np.ndarray:
    """
    Generate random samples from a specified distribution.

    Args:
        mean: Mean (or center) of distribution
        std: Standard deviation (or half-width for uniform)
        n: Number of samples
        dist_type: 'normal', 'uniform', or 'triangular'

    Returns:
        Array of n samples
    """
    if std <= 0:
        return np.full(n, mean)

    if dist_type == "normal":
        return np.random.normal(mean, std, n)
    elif dist_type == "uniform":
        half_range = std * np.sqrt(3)  # match variance: std = range/(2*sqrt(3))
        return np.random.uniform(mean - half_range, mean + half_range, n)
    elif dist_type == "triangular":
        # Symmetric triangular with same std
        half_range = std * np.sqrt(6)
        return np.random.triangular(mean - half_range, mean, mean + half_range, n)
    else:
        return np.random.normal(mean, std, n)


def calculate_tornado_data(inputs_dict: dict, outputs_dict: dict,
                           extra_params: dict) -> pd.DataFrame:
    """
    Calculate sensitivity (Spearman rank correlation) of each input on each output.

    Args:
        inputs_dict: Dict of input_name -> array of sampled values
        outputs_dict: Dict of output_name -> array of computed values
        extra_params: Reserved for future use

    Returns:
        DataFrame with columns: Input, Output, Correlation, Impact
    """
    rows = []
    for out_name, out_vals in outputs_dict.items():
        valid = np.isfinite(out_vals)
        for in_name, in_vals in inputs_dict.items():
            in_valid = in_vals[valid]
            out_valid = out_vals[valid]
            if len(in_valid) < 10 or np.std(in_valid) == 0:
                corr = 0.0
            else:
                corr, _ = stats.spearmanr(in_valid, out_valid)
                if np.isnan(corr):
                    corr = 0.0
            rows.append({
                "Input": in_name,
                "Output": out_name,
                "Correlation": corr,
                "Impact": abs(corr)
            })
    return pd.DataFrame(rows)


def create_box_plots(data_dict: dict, labels: list) -> go.Figure:
    """
    Create a grid of box plots for the given data.

    Args:
        data_dict: Dict of label -> array of values
        labels: List of keys to plot (order)

    Returns:
        Plotly Figure
    """
    n = len(labels)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=labels,
                        vertical_spacing=0.15, horizontal_spacing=0.12)

    for idx, label in enumerate(labels):
        r = idx // cols + 1
        c = idx % cols + 1
        vals = data_dict[label]
        fig.add_trace(
            go.Box(y=vals, name=label, marker_color="#1f77b4",
                   boxmean='sd', showlegend=False),
            row=r, col=c
        )

    fig.update_layout(height=350 * rows, title_text="Distribution Box Plots")
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")
    return fig


def create_tornado_chart(sensitivity_df: pd.DataFrame, output_name: str,
                         top_n: int = 8) -> go.Figure:
    """
    Create a horizontal tornado (bar) chart showing sensitivity of inputs
    on a single output property.

    Args:
        sensitivity_df: DataFrame from calculate_tornado_data
        output_name: Which output column to visualise
        top_n: Max number of inputs to show

    Returns:
        Plotly Figure
    """
    subset = sensitivity_df[sensitivity_df["Output"] == output_name].copy()
    subset = subset.sort_values("Impact", ascending=True).tail(top_n)

    colors = ["#d62728" if c < 0 else "#2ca02c" for c in subset["Correlation"]]

    fig = go.Figure(go.Bar(
        x=subset["Correlation"],
        y=subset["Input"],
        orientation="h",
        marker_color=colors,
        text=[f"{c:.3f}" for c in subset["Correlation"]],
        textposition="outside"
    ))

    fig.update_layout(
        title=f"Sensitivity to {output_name} (Spearman ρ)",
        xaxis_title="Rank Correlation",
        yaxis_title="Input Parameter",
        height=50 * top_n + 200,
        xaxis=dict(range=[-1.05, 1.05])
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")
    return fig


def create_distribution_plot_with_cdf(data: np.ndarray, title: str,
                                       n_bins: int = 40) -> go.Figure:
    """
    Create a histogram with an overlaid CDF curve.

    Args:
        data: 1-D array of values
        title: Plot title / property name
        n_bins: Number of histogram bins

    Returns:
        Plotly Figure with dual y-axes (histogram + CDF)
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Histogram
    fig.add_trace(
        go.Histogram(x=data, nbinsx=n_bins, name="Frequency",
                     marker_color="#1f77b4", opacity=0.7),
        secondary_y=False
    )

    # CDF
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    fig.add_trace(
        go.Scatter(x=sorted_data, y=cdf, mode="lines", name="CDF",
                   line=dict(color="#ff7f0e", width=2)),
        secondary_y=True
    )

    # P10 / P50 / P90 lines
    for pct, label in [(10, "P10"), (50, "P50"), (90, "P90")]:
        val = np.percentile(data, pct)
        fig.add_vline(x=val, line_dash="dash", line_color="gray",
                      annotation_text=f"{label}={val:.4g}")

    fig.update_layout(title=title, height=400, bargap=0.05)
    fig.update_xaxes(title_text=title, showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(title_text="Frequency", secondary_y=False, showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(title_text="Cumulative Probability", secondary_y=True,
                     range=[0, 1.05])
    return fig
