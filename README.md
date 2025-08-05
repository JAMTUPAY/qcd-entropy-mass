# Universal Entropy-Mass Relation in QCD

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16743904.svg)](https://doi.org/10.5281/zenodo.16743904)#


## Abstract

We report the discovery of a universal linear relationship between hadron masses and entanglement entropy loss during RG flow in QCD. Using continuum-extrapolated lattice QCD c-function data, we find that all light hadrons follow:

**m = |ΔS_RG| × [c₀ + a_B·B + α_S·S + β_J·J]**

where |ΔS_RG| ≈ 9.81 k_B is the universal entropy lost from 3 GeV to 0.2 GeV.

## Repository Structure

qcd-entropy-mass/
├── paper/
│   ├── Universal_Entropy_Mass_Relation_in_QCD.pdf  # Final paper
│   ├── Universal_Entropy_Mass_Relation_in_QCD.tex  # Original LaTeX
│   ├── main.tex                                    # Clean LaTeX version
│   └── figures/                                    # All figures (PDF & PNG)
├── code/
│   ├── qcd_entropy_code.py      # Complete analysis code
│   └── simple_extract_figures.py # Figure generation script
├── data/
│   └── lattice_cfunction.csv    # c-function data
└── requirements.txt             # Python dependencies

## Quick Start

### Reproduce the analysis
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete analysis
python code/qcd_entropy_code.py

# Or just generate figures
python simple_extract_figures.py

Key Results

R² = 0.851 across 13 hadrons (ground + excited states)
All coefficients significant beyond 5σ
Universal entropy loss: |ΔS_RG| = 9.81 ± 0.29 k_B

Fit Coefficients (MeV/k_B)

c₀ (base cost): 83.5 ± 1.7
a_B (baryon): 15.0 ± 2.4
α_S (strangeness): 11.4 ± 1.2
β_J (spin): 25.3 ± 2.2

Citation
If you use this work, please cite:

bibtex@article{Tupay2025,
  title={Universal Entropy-Mass Relation in QCD: Discovery from Lattice c-Function},
  author={Tupay, Johann Anton Michael},
  journal={arXiv preprint arXiv:2XXX.XXXXX},
  year={2025}
}

Author
Johann Anton Michael Tupay
Email: jamtupayl@icloud.com
Location: London, United Kingdom

License
MIT License - see LICENSE file

