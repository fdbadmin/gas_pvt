# Gas PVT Analysis - Streamlit App

**Industry-standard correlations for reservoir engineering calculations**

This repository contains everything needed to run the Gas PVT Analysis web application.

---

## 📋 Files Included

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application |
| `gas_pvt_correlations.py` | Gas PVT correlation functions (Hall-Yarborough, Lucas, etc.) |
| `water_pvt_correlations.py` | Water-gas interaction correlations |
| `uncertainty_enhanced.py` | Helper functions for uncertainty analysis and visualizations |
| `requirements.txt` | Python package dependencies |
| `config.toml` | Streamlit configuration (theme, server settings) |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/fdbadmin/gas_pvt.git
cd gas_pvt
```

### 2. Set up Python virtual environment (recommended)

```bash
python3 -m venv .venv

# Activate virtual environment
# Mac/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Streamlit config (optional)

```bash
mkdir -p .streamlit
cp config.toml .streamlit/config.toml
```

### 5. Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 📦 Required Packages

- **streamlit** - Web application framework
- **pandas** - Data manipulation
- **numpy** - Numerical computations
- **plotly** - Interactive visualizations
- **scipy** - Statistical distributions and KDE for uncertainty analysis

---

## 🎯 Features

### Analysis Modules:

1. **Single Point Calculator** - Calculate PVT properties at specific conditions
2. **Pressure Profile** - Generate pressure vs. depth profiles
3. **Complete PVT Table** - Comprehensive 14-property tables with:
   - Temperature range option (plot multiple temperature curves)
   - Interactive plots for all properties
   - Downloadable CSV data
4. **Water-Gas Properties** - Water content and gas solubility calculations
5. **Critical Properties** - Pseudo-critical property estimation
6. **Uncertainty Analysis** - Advanced Monte Carlo simulation with:
   - Mean ± Std Dev or Mean ± Range inputs
   - Distribution types: Normal, Uniform, Triangular
   - Box plots with P10/P50/P90
   - Tornado charts for sensitivity ranking
   - Coefficient of Variation (CoV) analysis with color coding

### Key Capabilities:

- **Multiple unit systems** - psia/psig/MPa/kPa/bar, °F/°C
- **Automated interpretation** - Engineering flags and warnings
- **Composition handling** - N₂, CO₂, H₂S corrections (Wichert-Aziz)
- **Interactive plots** - Hover for values, zoom, pan, export
- **Data export** - Download results as CSV

---

## 🔧 Correlations Used

- **Z-factor**: Hall-Yarborough (1973)
- **Viscosity**: Lucas (1980) with Standing corrections
- **Pseudo-critical**: Sutton (1985)
- **Acid gas corrections**: Wichert-Aziz (1972)
- **Water content**: McCarthy-Boyd-Reid (GPSA)
- **Critical properties**: Riazi-Daubert

---

## 📖 Usage Tips

- Check the **User Guide & Background** section in the app for detailed documentation
- Use **Uncertainty Analysis** when you have measurement uncertainty or need sensitivity analysis
- Use **Temperature Range** in Complete PVT Table to compare thermal effects
- All calculations use absolute pressure (psia) internally with automatic conversion

---

## 🐛 Troubleshooting

**Issue**: Port 8501 already in use  
**Solution**: Stop other Streamlit apps or use a different port:
```bash
streamlit run app.py --server.port 8502
```

**Issue**: Import errors  
**Solution**: Ensure all dependencies are installed:
```bash
pip install --upgrade -r requirements.txt
```

**Issue**: Calculations fail at extreme conditions  
**Solution**: Check if inputs are within correlation valid ranges:
- Pressure: 100-15,000 psia (Ppr < 15)
- Temperature: Above critical temperature preferred

---

## 📝 License

This tool implements industry-standard correlations from published SPE papers and the GPSA Engineering Data Book.

---

## 📧 Support

For questions about PVT correlations or usage, refer to:
- SPE Monograph Vol. 20: *Phase Behavior* by Whitson & Brulé
- GPSA Engineering Data Book (15th Edition)
- McCain: *The Properties of Petroleum Fluids*

---

**Version**: 2.0 (March 2026)  
**Enhanced Features**: Temperature ranges, advanced uncertainty analysis with distributions and sensitivity charts
