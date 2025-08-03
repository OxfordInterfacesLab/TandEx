# TandEx
Tandem Explorer: Multi-diode circuit model to calculate of 2J tandem solar cell efficiency (PVK+Si)

**Modeling the Practical Efficiency Limits of Tandem Solar Cells**

The Jupyter notebook (`tandex_0X.ipynb`) provides a comprehensive modeling framework to evaluate the practical efficiency limits of **perovskite–silicon tandem solar cells**, particularly accounting for the effects of **transparent conductive electrodes (TCEs)** on performance.

---

## 📘 Overview

Tandem solar cells, especially those combining perovskite and silicon materials, have the potential to exceed the single-junction Shockley–Queisser limit. This notebook models their practical limitations using a **multi-diode circuit approach**.

The study incorporates:
- Electrical losses from TCE sheet resistance
- Optical losses from TCE transmittance
- Radiative dark current modeling
- Quantum efficiency (EQE) calculations
- Integration of AM1.5G solar spectrum

It also includes visualizations and parameter sweeps to explore design trade-offs.

---

## 📐 Theory Summary

### 1. **TCE-Related Losses**
- **Electrical**: Series resistance is modeled via lateral conduction losses, leading to:

Rs = (R_sheet_TCE · l) / (6w)

or

ρs = (R_sheet_TCE · l²) / 12


**Optical:** Optical shading loss is calculated using weighted average transmittance (WAT):

WAT = ∫[ T_x(λ) · AM1.5G(λ) dλ ] / ∫[ AM1.5G(λ) dλ ]

---

### 2. Radiative Dark Current and Jsc

**Radiative dark current:**

J01 = q · ∫[ BB(λ) · EQE(λ) dλ ]

where

BB(λ) = (2πc / λ⁴) · 1 / (exp(hc / λkT) - 1)

**Short-circuit current for the perovskite sub-cell:**

Jsc_P = q · ∫[ AM1.5G(λ) · EQE_P(λ) dλ ]

with

EQE_P(λ) = T_x_front · (1 - exp(-α(λ) · W))

Here:
- `T_x_front` is the transmittance through the front layer stack (TCE + ARC + ETL + buffers)
- `W` is the absorber thickness
- `α(λ)` is the wavelength-dependent absorption coefficient

---

## ⚙️ Installation

Clone this repository and set up the Python environment.

### Requirements

Install dependencies using `pip` or `conda`. The following packages are required:

- numpy
- scipy
- matplotlib
- pandas
- PySpice (for circuit simulation)
- time, gc, ctypes (standard libraries)

