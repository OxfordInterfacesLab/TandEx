# TandEx
Tandem Explorer: Multi-diode circuit model to calculate of 2J tandem solar cell efficiency (PVK+Si)

**Modeling the Practical Efficiency Limits of Tandem Solar Cells**

This repository contains a Jupyter notebook (`tandex_01.ipynb`) that provides a comprehensive modeling framework to evaluate the practical efficiency limits of **perovskite–silicon tandem solar cells**, particularly accounting for the effects of **transparent conductive electrodes (TCEs)** on performance.

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
  
  \[
  R_s = \frac{R_{\text{sheet}}^{\text{TCE}} \cdot l}{6w}
  \quad \text{or} \quad
  \rho_s = \frac{R_{\text{sheet}}^{\text{TCE}} \cdot l^2}{12}
  \]

- **Optical**: Optical shading loss modeled using weighted average transmittance (WAT):

  \[
  WAT = \frac{\int T_x(\lambda) \cdot AM1.5G(\lambda) \, d\lambda}{\int AM1.5G(\lambda) \, d\lambda}
  \]

### 2. **Radiative Dark Current and Jsc**
- Radiative dark current (J₀₁):

  \[
  J_{01} = q \int BB(\lambda) \cdot EQE(\lambda) \, d\lambda
  \]

- Short-circuit current (Jsc) for perovskite:

  \[
  J_{sc,P} = q \int AM1.5G(\lambda) \cdot EQE_P(\lambda) \, d\lambda
  \]

  Where:

  \[
  EQE_P(\lambda) = T_{x,\text{front}} \cdot \left(1 - e^{-\alpha(\lambda) W}\right)
  \]

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

### Using pip

```bash
pip install numpy scipy matplotlib pandas
pip install PySpice
