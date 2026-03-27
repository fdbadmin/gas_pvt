"""
PDF Report Generator – Shell-branded PVT report output.

Uses fpdf2 (pure Python, no system dependencies) to produce a
professional letter-size PDF with Shell branding.

The module also provides a helper to render Plotly figures to
static PNG bytes via kaleido (optional; falls back to placeholder
if kaleido is unavailable).

Dependencies:
    pip install fpdf2 kaleido
"""

import io
import datetime
from typing import List, Optional, Tuple

import pandas as pd

try:
    from fpdf import FPDF
    _HAS_FPDF = True
except ImportError:
    _HAS_FPDF = False

try:
    import plotly.io as pio
    _HAS_KALEIDO = True
except ImportError:
    _HAS_KALEIDO = False


# Shell brand colours
_RED = (221, 29, 33)
_YELLOW = (251, 206, 7)
_DARK = (51, 51, 51)
_LIGHT_BG = (255, 248, 225)


def _fig_to_png_bytes(fig, width: int = 900, height: int = 500) -> Optional[bytes]:
    """Convert a Plotly figure to PNG bytes via kaleido."""
    if not _HAS_KALEIDO:
        return None
    try:
        return pio.to_image(fig, format="png", width=width, height=height, scale=2)
    except Exception:
        return None


class PVTReport(FPDF if _HAS_FPDF else object):
    """Shell-branded PDF report."""

    def __init__(self, title: str = "Gas PVT Analysis Report"):
        if not _HAS_FPDF:
            raise ImportError("fpdf2 is required for PDF generation. "
                              "Install with: pip install fpdf2")
        super().__init__(orientation='P', unit='mm', format='Letter')
        self.report_title = title
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(15, 15, 15)

    # ------------------------------------------------------------------
    def header(self):
        # Red stripe
        self.set_fill_color(*_RED)
        self.rect(0, 0, 216, 8, 'F')
        # Yellow accent line
        self.set_fill_color(*_YELLOW)
        self.rect(0, 8, 216, 2, 'F')

        self.set_y(14)
        self.set_font("Helvetica", 'B', 18)
        self.set_text_color(*_DARK)
        self.cell(0, 10, self.report_title, ln=True, align='L')

        self.set_font("Helvetica", '', 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5,
                  f"Generated {datetime.datetime.now().strftime('%d %B %Y  %H:%M')}",
                  ln=True)

        self.set_draw_color(*_RED)
        self.line(15, self.get_y() + 2, 201, self.get_y() + 2)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}  |  Shell Gas PVT Tool",
                  align='C')

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def add_section(self, title: str):
        """Add a section heading."""
        self.set_font("Helvetica", 'B', 14)
        self.set_text_color(*_RED)
        self.cell(0, 10, title, ln=True)
        self.set_text_color(*_DARK)
        self.ln(2)

    def add_subsection(self, title: str):
        self.set_font("Helvetica", 'B', 11)
        self.set_text_color(*_DARK)
        self.cell(0, 8, title, ln=True)
        self.ln(1)

    def add_text(self, text: str):
        self.set_font("Helvetica", '', 10)
        self.set_text_color(*_DARK)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def add_key_value(self, pairs: List[Tuple[str, str]]):
        """Add a list of (label, value) pairs in two columns."""
        self.set_font("Helvetica", '', 10)
        col_w = 90
        for label, value in pairs:
            self.set_text_color(100, 100, 100)
            self.cell(col_w, 6, label, border=0)
            self.set_text_color(*_DARK)
            self.set_font("Helvetica", 'B', 10)
            self.cell(col_w, 6, str(value), ln=True)
            self.set_font("Helvetica", '', 10)
        self.ln(3)

    def add_dataframe(self, df: pd.DataFrame, col_widths: Optional[List[float]] = None):
        """Render a DataFrame as a table."""
        n_cols = len(df.columns)
        if col_widths is None:
            available = 186  # mm (letter - margins)
            col_widths = [available / n_cols] * n_cols

        # Header row
        self.set_font("Helvetica", 'B', 9)
        self.set_fill_color(*_LIGHT_BG)
        self.set_text_color(*_DARK)
        for j, col in enumerate(df.columns):
            self.cell(col_widths[j], 7, str(col), border=1, fill=True, align='C')
        self.ln()

        # Data rows
        self.set_font("Helvetica", '', 9)
        for _, row in df.iterrows():
            if self.get_y() > 250:
                self.add_page()
            for j, col in enumerate(df.columns):
                self.cell(col_widths[j], 6, str(row[col]), border=1, align='C')
            self.ln()
        self.ln(3)

    def add_plot(self, fig, width_mm: float = 180, caption: str = ""):
        """Add a Plotly figure as a PNG image."""
        png = _fig_to_png_bytes(fig)
        if png is None:
            self.add_text(f"[Plot: {caption or 'chart'} – kaleido not available for rendering]")
            return
        # Write PNG to a temp buffer and embed
        buf = io.BytesIO(png)
        if self.get_y() > 180:
            self.add_page()
        self.image(buf, x=15, w=width_mm)
        if caption:
            self.set_font("Helvetica", 'I', 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 5, caption, ln=True, align='C')
            self.set_text_color(*_DARK)
        self.ln(4)

    def add_interpretation(self, flags: List[str], severities: List[str]):
        """Add interpretation flags with colour coding."""
        if not flags:
            self.add_text("No flags raised — all properties within expected ranges.")
            return
        for flag, sev in zip(flags, severities):
            if sev == "warning":
                self.set_text_color(200, 120, 0)
                prefix = "⚠ "
            elif sev == "error":
                self.set_text_color(*_RED)
                prefix = "✗ "
            else:
                self.set_text_color(0, 128, 0)
                prefix = "✓ "
            self.set_font("Helvetica", '', 9)
            self.cell(0, 5, f"{prefix}{flag}", ln=True)
        self.set_text_color(*_DARK)
        self.ln(3)

    # ------------------------------------------------------------------
    def build_pdf(self) -> bytes:
        """Return the PDF as bytes."""
        self.alias_nb_pages()
        return self.output()


def generate_pvt_report(title: str,
                        input_params: List[Tuple[str, str]],
                        results_df: pd.DataFrame,
                        figures: Optional[list] = None,
                        flags: Optional[List[str]] = None,
                        severities: Optional[List[str]] = None,
                        extra_text: str = "") -> bytes:
    """
    One-call convenience function to produce a complete PVT report PDF.

    Args:
        title: Report main title
        input_params: List of (label, value) tuples for input parameters
        results_df: DataFrame of computed results
        figures: Optional list of (plotly_figure, caption) tuples
        flags: Optional interpretation flag strings
        severities: Optional corresponding severity strings
        extra_text: Optional additional text/notes

    Returns:
        PDF file as bytes
    """
    rpt = PVTReport(title)
    rpt.add_page()

    rpt.add_section("Input Parameters")
    rpt.add_key_value(input_params)

    rpt.add_section("Results")
    rpt.add_dataframe(results_df)

    if figures:
        rpt.add_section("Plots")
        for fig, caption in figures:
            rpt.add_plot(fig, caption=caption)

    if flags:
        rpt.add_section("Interpretation")
        rpt.add_interpretation(flags, severities or ["info"] * len(flags))

    if extra_text:
        rpt.add_section("Notes")
        rpt.add_text(extra_text)

    return rpt.build_pdf()
