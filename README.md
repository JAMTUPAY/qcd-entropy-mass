# Universal Entropy-Mass Relation in QCD

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## Abstract

We report the discovery of a universal linear relationship between hadron masses and entanglement entropy loss during RG flow in QCD. Using continuum-extrapolated lattice QCD c-function data, we find that all light hadrons follow:

**m = |ΔS_RG| × [c₀ + a_B·B + α_S·S + β_J·J]**

where |ΔS_RG| ≈ 9.81 k_B is the universal entropy lost from 3 GeV to 0.2 GeV.

## Repository Structure

qcd-entropy-mass/
├── paper/
│   ├── main.tex             # LaTeX source
│   └── figures/             # All figures
├── code/
│   └── (analysis code)      # To be added
├── data/
│   └── lattice_cfunction.csv # c-function data
└── simple_extract_figures.py # Figure generation

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate figures
python simple_extract_figures.py
Key Results

R² = 0.851 across 13 hadrons
All coefficients significant beyond 5σ

Author
Johann Anton Michael Tupay
Email: jamtupayl@icloud.com
License
MIT License - see LICENSE file

Save it (`Control+X`, `Y`, `Enter`).

## Step 11: Add README and commit everything

```bash
git add README.md
git commit -m "Initial commit: QCD entropy-mass discovery paper and analysis"
