#libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve, least_squares
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.interpolate import RegularGridInterpolator

from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
import gc

# this code requires the ngspice library to be installed
import ctypes
NgSpiceShared.library_path = r'C:\Spice64_dll\dll-vs\ngspice.dll'
ctypes.cdll.LoadLibrary("C:/Spice64_dll/dll-vs/ngspice.dll")

# Constants
k = 1.380649e-23  # Boltzmann constant in J/K
q = 1.60217662e-19  # Electron charge in C
h 	= 	6.62606957e-34 	# Js Planck's constant
c	= 	2.99792458e8	# m/s speed of light

epsrox = 3.9
eps0 = 8.85e-14  # F/cm
epsrs = 11.9
T = 300  # K
Vt = (k * T) / q  # eV
hc = 1240  # eV*nm



# functions

def Si_bulk_lifetime(Ndop, Delta_n, waferT, ni, Temp):
    # Niewelt model for intrinsic lifetime calculation
    # https://doi.org/10.1016/j.solmat.2021.111467
    
    if Ndop > 0:
        p_0 = Ndop
        n_0 = ni**2 / p_0
    else:
        n_0 = -Ndop
        p_0 = ni**2 / n_0

    if n_0 > p_0:
        n_d = n_0 + Delta_n
        p_d = Delta_n
    else:
        n_d = Delta_n
        p_d = p_0 + Delta_n

    Blow = 4.76e-15  # [cm^3s^-1] Radiative recombination coefficient
    Bmin = 0.2 + (0 - 0.2) / (1 + (Temp / 320)**2.5)
    b1 = 2 * (1.5e18 + (1e7 - 1.5e18) / (1 + (Temp / 550)**3.0))
    b3 = 2 * (4e18 + (1e9 - 4e18) / (1 + (Temp / 365)**3.54))
    Brel = Bmin + (1.00 - Bmin) / (1 + ((n_d + p_d) / b1)**0.54 + ((n_d + p_d) / b3)**1.25)

    nieff = ni**2 / Brel
    if n_0 > p_0:
        n0eff = n_0
        p0eff = nieff / n_0
    else:
        p0eff = p_0
        n0eff = nieff / p_0

    n_deff = p0eff + Delta_n
    p_deff = n0eff + Delta_n

    B = Brel * Blow  # Radiative recombination probability

    # Photon recycling
    fPR = 0.9835 + 0.006841 * np.log10(waferT) - 4.554e-9 * Delta_n**0.4612

    geeh = 1 + (4.38 - 1) / (1 + ((n_d + p_d) / 4e17)**2)
    gehh = 1 + (4.88 - 1) / (1 + ((n_d + p_d) / 4e17)**2)

    R_Auger = (3.41e-31 * geeh * ((n_d**2 * p_d) - (n0eff**2 * p0eff)) +
               1.17e-31 * gehh * ((n_d * p_d**2) - (n0eff * p0eff**2)))

    t_Aug = Delta_n / R_Auger

    R_rad = ((n_d * p_d - nieff) * (B * (1 - fPR)))

    t_Rad = Delta_n / R_rad

    return t_Aug, t_Rad, np.sqrt(nieff), p0eff, n0eff, n_deff, p_deff 

def calculate_efficiency(voltages, currents, plotting=False):
# Calculate power at each voltage point
    powers = voltages * currents

    # Find max power point (MPP)
    max_power_idx = np.argmax(powers)
    v_mpp = voltages[max_power_idx]
    i_mpp = currents[max_power_idx]
    p_mpp = powers[max_power_idx]

    efficiency = (p_mpp) * 1e3
    
    # Find Jsc (iJ at iV closest to 0 V)
    Jsc_index = np.argmin(np.abs(voltages))
    Jsc = currents[Jsc_index]
    
    # Find where current crosses zero
    if np.any(currents < 0):
        f_voc = interp1d(currents, voltages)
        Voc = float(f_voc(0))
    else:
        Voc = np.nan  # No crossing found

    FF = efficiency*1e-3/(Voc*Jsc)


    if plotting == True:
        fig, ax1 = plt.subplots(figsize=(6, 4))
        color = 'tab:blue'
        ax1.set_xlabel('Voltage (V)')
        ax1.set_ylabel('Current Density (mA/cm2)', color=color)
        ax1.plot(voltages, currents*1e3, color=color, label='Current')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, np.max(currents*1e3) * 1.1)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Power (mW/cm2)', color=color)
        ax2.plot(voltages, powers*1e3, color=color, label='Power')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, np.max(powers*1e3) * 1.1)

        # Mark the max power point
        ax2.plot(v_mpp, p_mpp, 'ko')
        ax2.annotate(f'{efficiency:.2f}%', xy=(v_mpp, 1e3*p_mpp),
                    xytext=(v_mpp*0.9, 1e3*p_mpp*1.01),
                    
                    fontsize=10, color='black')

        plt.title('Solar Cell I-V and Power Curve')
        fig.tight_layout()
        plt.show()

    return efficiency, Jsc, Voc, FF

def pseudo_efficiency_Si( l, Waf_thick, J0d, R_tco, dop_type, Ndop, ni, Jsc_Si):

    # Sample range for excess carrier concentration (Delta n)
    delta_n = np.logspace(12, 16.4, 1000)  # in cm^-3
    # J0d = 2e-15 # Dark saturation current density, A/cm2

    # Calculate intrinsic lifetimes
    t_Aug, t_Rad, n_i_eff, _, _, _, _ = Si_bulk_lifetime(dop_type * Ndop, delta_n, Waf_thick, ni, T)

    # Calculate effective lifetime including J0 term
    tau_eff = 1 / (1 / t_Aug + 1 / t_Rad + (J0d * (Ndop + delta_n) / (Waf_thick * q * n_i_eff**2)))  # Combined intrinsic lifetime

    # Calculate iJ using intrinsic lifetime
    iJ =  ( Jsc_Si - (q * Waf_thick * delta_n / tau_eff) ) * 1e3  # Convert to mA/cm^2

    # Calculate iV
    iV = (k * T / q) * np.log((delta_n * (Ndop + delta_n)) / (n_i_eff**2)) - (iJ * 1e-3 * R_tco * l**2 / 6)

    # Calculate implied efficiency
    pseudo_efficiency = iV * iJ  # Simplified efficiency calculation

    # Find Jsc (iJ at iV closest to 0 V)
    Jsc_index = np.argmin(np.abs(iV))
    iJsc = iJ[Jsc_index]

    # Find Voc (iV where iJ crosses zero)
    if np.any(iJ < 0):
        f_voc = interp1d(iJ, iV)
        iVoc = float(f_voc(0))
    else:
        iVoc = np.nan

    iFF=100* np.max(pseudo_efficiency) / (iVoc * iJsc)
    
    print(f"Pseudo J-V curve results:")
    print(f"Implied Efficiency: {np.max(iV * iJ):.4f}, Jsc = {iJsc:.4f} mA/cm^2,Voc = {1e3*iVoc:.4f} mV,FF = {iFF:.4f} %")
    
    return iV, iJ, iJsc, iVoc, iFF


def Jsc_pvk_func(lambda_vec, AM15g, pvk_thick):
    # Calculate the short-circuit current density for the perovskite cell
    
    file_path = "./data/CsFaPbIB_1,72_Tejada2018.csv"
    df_pvk = pd.read_csv(file_path)
    
    # pvk_thick= 600e-7  # cm
    interp1d_pvk_alpha = interp1d(df_pvk['lambda_k'], df_pvk['alpha'], kind='linear', fill_value='extrapolate')
    
    Abs_pvk=1-np.exp(-interp1d_pvk_alpha(lambda_vec) * pvk_thick)
    
    Jsc_pvk = q * np.trapezoid(AM15g * (Abs_pvk), lambda_vec)  # cm^-1 * cm = unitless
    
    AM15g_after_pvk = AM15g * np.exp(-interp1d_pvk_alpha(lambda_vec) * pvk_thick)


    return Jsc_pvk, AM15g_after_pvk, Abs_pvk

def Jsc_Si_fun(Waf_thick, lambda_vec, AM15g):
    
    # Load the Si n_k properties
    file_path = "./data/Si_opt.csv"
    Si_nk = pd.read_csv(file_path)
    Si_nk = Si_nk.rename(columns={
        'λ,n (nm)': 'lambda_n',
        'n': 'n',
        'λ,k (nm)': 'lambda_k',
        'k': 'k',
        'α (cm⁻¹)': 'alpha'
    })
    

    a=0.935; b=0.67
    

    a_W_lambda = interp1d(Si_nk['lambda_k'], Waf_thick*Si_nk['alpha'], kind='linear', fill_value='extrapolate')
    n_lambda = interp1d(Si_nk['lambda_n'], Si_nk['n'], kind='linear', fill_value='extrapolate')

    var=a_W_lambda(lambda_vec)
    Zfact=(2+a*(var)**b)/(1+a*(var)**b)

    exp_term = np.exp(-2* Zfact * a_W_lambda(lambda_vec) )
    Ax= (1 - exp_term) / (1 - (1 - 1 / n_lambda(lambda_vec)**2) * exp_term)

    Jsc_Si = q * np.trapezoid(AM15g*Ax , lambda_vec) # in Amps/cm2.
    # print("Isc_Si =", 1e3*Isc_Si2/(w * (l / 2)), "mA/cm^2")

    return Jsc_Si, Ax



def jv_diode_model(Vcell, I01_Si, I0Aug_Si, n_def, I0n_Si, Isc_Si, Rsh_Si, Rser, w,l):
    
 
    top_cell= [0, 0, 0,  0, 0, 0]
    bottom_cell = [Isc_Si, Rsh_Si, I01_Si, I0Aug_Si* w * (l / 2) , I0n_Si* w * (l / 2), n_def]

    circuit=make_circuit(top_cell, bottom_cell, Rser)

    # Set up simulator
    simulator = circuit.simulator()
    currents = []

    for v in Vcell:
        circuit['VLOAD'].dc_value = v  # 
        analysis = simulator.operating_point()
        i_out = float(analysis.branches['vload'][0])
        currents.append(i_out)
        
    voltages = np.array(Vcell)
    currents = np.array(currents) / (w * (l / 2))

    eff, Jsc, Voc =calculate_efficiency(voltages, currents, False)

    print(f"Efficiency= {eff:.4f}, Jsc = {1e3*Jsc:.5f} mA/cm^2, Voc = {Voc:.5f} V, FF = {0.1*eff/(Voc*Jsc):.5f} % ")
    
    return currents, eff, Jsc, Voc


def find_optimal_pitch(J0_Si, J0d, J0Aug_Si, Rsh_Si, Jsc_Si, l_values, R_tco, AAF):
    """
    Finds the optimal finger pitch l for maximum efficiency.

    Returns:
        l_opt (float): Optimal pitch (cm)
        app_Voc (float): Open-circuit voltage (V)
        app_eff (np.ndarray): Efficiency array (%)
        app_FF (np.ndarray): Fill factor array
        app_Jsc (np.ndarray): Short-circuit current array (A/cm^2)
    """

    def Voc_function(V):
        term1 = (J0_Si + J0d) * np.exp(q * V / (k * T))
        term2 = J0Aug_Si * np.exp(q * V / ((2/3) * k * T))
        term3 = V / (Rsh_Si)
        return term1 + term2 + term3 - Jsc_Si

    sol = root_scalar(Voc_function, bracket=[0, 0.8], method='brentq')
    app_Voc = sol.root

    app_Jsc = Jsc_Si * AAF
    app_neff = 2 * 1 * (2/3) / (1 + 2/3)
    app_Voceff = app_Voc * q / (k * T) / 0.8  # Convert to eV
    app_FF = ((app_Voceff - np.log(app_Voceff + 0.72)) / (app_Voceff + 1)) * \
             (1 - (((l_values ** 2 / 12)) * R_tco) * Jsc_Si * AAF / app_Voc - app_Voc / (Rsh_Si * Jsc_Si * AAF))
    app_eff = app_FF * app_Voc * app_Jsc / 0.1 * 100

    l_opt = l_values[np.argmax(app_eff)]
    return l_opt, app_Voc, app_eff, app_FF, app_Jsc


def make_circuit(top_cell, bottom_cell, Rser):
    circuit = Circuit('Solar Cell Model')

    Isc_pvk, Rsh_pvk, I0_pvk, I0def_pvk, Rdef_pvk, I02_pvk = top_cell

    Isc_Si, Rsh_Si, I01_Si, I0Aug_Si, I0d_Si, n_def = bottom_cell

    node1='n1'
    node2='n2'
    node22='n22'
    node3='n3'
    gnd=circuit.gnd

    circuit.V('LOAD', node1, gnd, -0.01) # Load voltage source to measure output
    circuit.R('Ser', node1, node2, Rser)
    
    circuit.I('pvk', node2, node3, Isc_pvk)
    circuit.R('sh_pvk', node2, node3, Rsh_pvk)

    circuit.model('Dpvk', 'D', IS=I0_pvk , N=1)
    circuit.D(10, node2, node3, model='Dpvk')
    
    if I0def_pvk>0:
        circuit.model('DpvkAug', 'D', IS=I0def_pvk , N=1)
        circuit.D(12, node2, node22, model='DpvkAug')
        circuit.R('def_pvk', node22, node3, Rdef_pvk)
    
    
    if I02_pvk>0:
        circuit.model('DpvkRec', 'D', IS=I02_pvk , N=2)
        circuit.D(13, node2, node3, model='DpvkRec')


    circuit.I('Si', node3, gnd, Isc_Si)
    circuit.R('sh_Si', node3, gnd, Rsh_Si)

    circuit.model('Dsi', 'D', IS=I01_Si , N=1)
    circuit.D(21, node3, gnd, model='Dsi')
    circuit.model('DsiAug', 'D', IS=I0Aug_Si , N=2/3)
    circuit.D(22, node3, gnd, model='DsiAug')
    circuit.model('DsiRec', 'D', IS=I0d_Si , N=n_def)
    circuit.D(23, node3, gnd, model='DsiRec')

    return circuit

def simulate_circuit(top_cell, bottom_cell, Rser, w, l, Vcell):
    """
    Simulate the tandem cell circuit and return voltages and current densities.

    Args:
        top_cell: List of top cell parameters.
        bottom_cell: List of bottom cell parameters.
        Rser: Series resistance (ohms.cm^2).
        w: Cell width (cm).
        l: Finger pitch (cm).
        Vcell: Array of voltages to simulate.
        t_func: Module containing make_circuit.

    Returns:
        voltages: np.ndarray of voltages.
        currents: np.ndarray of current densities (A/cm^2).
    """
    Rseries= Rser / (w * (l / 2))
    
    circuit = Circuit('Solar Cell Model')

    Isc_pvk, Rsh_pvk, I0_pvk, I0def_pvk, Rdef_pvk, I02_pvk = top_cell

    Isc_Si, Rsh_Si, I01_Si, I0Aug_Si, I0d_Si, n_def = bottom_cell

    node1='n1'
    node2='n2'
    node22='n22'
    node3='n3'
    gnd=circuit.gnd

    circuit.V('LOAD', node1, gnd, -0.01) # Load voltage source to measure output
    circuit.R('Ser', node1, node2, Rseries)
    
    circuit.I('pvk', node2, node3, Isc_pvk)
    circuit.R('sh_pvk', node2, node3, Rsh_pvk)

    circuit.model('Dpvk', 'D', IS=I0_pvk , N=1)
    circuit.D(10, node2, node3, model='Dpvk')
    
    if I0def_pvk>0:
        circuit.model('DpvkAug', 'D', IS=I0def_pvk , N=1)
        circuit.D(12, node2, node22, model='DpvkAug')
        circuit.R('def_pvk', node22, node3, Rdef_pvk)
    
    
    if I02_pvk>0:
        circuit.model('DpvkRec', 'D', IS=I02_pvk , N=2)
        circuit.D(13, node2, node3, model='DpvkRec')


    circuit.I('Si', node3, gnd, Isc_Si)
    circuit.R('sh_Si', node3, gnd, Rsh_Si)

    circuit.model('Dsi', 'D', IS=I01_Si , N=1)
    circuit.D(21, node3, gnd, model='Dsi')
    circuit.model('DsiAug', 'D', IS=I0Aug_Si , N=2/3)
    circuit.D(22, node3, gnd, model='DsiAug')
    circuit.model('DsiRec', 'D', IS=I0d_Si , N=n_def)
    circuit.D(23, node3, gnd, model='DsiRec')
 
    simulator = circuit.simulator()
    currents = []

    for v in Vcell:
        circuit['VLOAD'].dc_value = v
        analysis = simulator.operating_point()
        i_out = float(analysis.branches['vload'][0])
        currents.append(i_out)

    voltages = np.array(Vcell)
    currents = np.array(currents) / (w * (l / 2))
    return voltages, currents

def simulate_pvk_sj(top_cell, Rser, w, l, Vcell):
    Rseries= Rser / (w * (l / 2))
    
    circuit = Circuit('Solar Cell Model')

    Isc_pvk, Rsh_pvk, I0_pvk, I0def_pvk, Rdef_pvk, I02_pvk = top_cell

    
    circuit.V('LOAD', 'n1', circuit.gnd, -0.01) # Load voltage source to measure output
    circuit.R('Ser', 'n1', 'n2', Rseries)
    circuit.I('pvk', 'n2', circuit.gnd, Isc_pvk)
    circuit.R('sh_pvk', 'n2', circuit.gnd, Rsh_pvk)

    circuit.model('Dpvk', 'D', IS=I0_pvk , N=1)
    circuit.D(10, 'n2', circuit.gnd, model='Dpvk')
    
    if I0def_pvk>0:
        circuit.model('DpvkAug', 'D', IS=I0def_pvk , N=1)
        circuit.D(12, 'n2', 'n22', model='DpvkAug')
        circuit.R('def_pvk', 'n22', circuit.gnd, Rdef_pvk)
    
    if I02_pvk>0:
        circuit.model('DpvkRec', 'D', IS=I02_pvk , N=2)
        circuit.D(13, 'n2', circuit.gnd, model='DpvkRec')

 
    simulator = circuit.simulator()
    currents = []

    for v in Vcell:
        circuit['VLOAD'].dc_value = v
        analysis = simulator.operating_point()
        i_out = float(analysis.branches['vload'][0])
        currents.append(i_out)

    voltages = np.array(Vcell)
    currents = np.array(currents) / (w * (l / 2))
    return voltages, currents


def simulate_Si_sj(bottom_cell, Rser, w, l, Vcell):
    Rseries= Rser / (w * (l / 2))
    
    circuit = Circuit('Solar Cell Model')

    Isc_Si, Rsh_Si, I01_Si, I0Aug_Si, I0d_Si, n_def = bottom_cell

    node1='n1'
    node2='n2'
    node22='n22'
    node3='n2'
    gnd=circuit.gnd

    circuit.V('LOAD', node1, gnd, -0.01) # Load voltage source to measure output
    circuit.R('Ser', node1, node2, Rseries)
    
    circuit.I('Si', node3, gnd, Isc_Si)
    circuit.R('sh_Si', node3, gnd, Rsh_Si)

    circuit.model('Dsi', 'D', IS=I01_Si , N=1)
    circuit.D(21, node3, gnd, model='Dsi')
    circuit.model('DsiAug', 'D', IS=I0Aug_Si , N=2/3)
    circuit.D(22, node3, gnd, model='DsiAug')
    circuit.model('DsiRec', 'D', IS=I0d_Si , N=n_def)
    circuit.D(23, node3, gnd, model='DsiRec')

  
    simulator = circuit.simulator()
    currents = []

    for v in Vcell:
        circuit['VLOAD'].dc_value = v
        analysis = simulator.operating_point()
        i_out = float(analysis.branches['vload'][0])
        currents.append(i_out)

    voltages = np.array(Vcell)
    currents = np.array(currents) / (w * (l / 2))
    return voltages, currents

def plot_pvk_iv_curve(voltages, currents, eff, Jsc, Voc, FF, ERE):
    
    
    fig, ax1 = plt.subplots(figsize=(5, 4))
    plt.rc('font', family='Arial', size=16)

    # Left y-axis: linear current density
    ax1.plot(voltages, currents*1e3, label='3-diode model, J-V')
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Current Density (mA/cm$^2$)')
    ax1.legend(loc='lower left')
    ax1.grid(False)
    ax1.set_xlim((0, 1.3))
    ax1.set_ylim((-0.01, 23))

    # Right y-axis: log(current)
    ax2 = ax1.twinx()
    ax2.plot(voltages, (np.abs(currents-Jsc)*1e3), color='tab:red', linestyle='--')
    ax2.set_ylabel('| $J - J_{sc}|$ (mA/cm$^2$)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_yscale('log')
    ax2.minorticks_on()

    cell_text = [
        [f"{Voc:.3f} V"],
        [f"{Jsc*1e3:.2f} mA/cm$^2$"],
        [f"{FF*100:.2f} %"],
        [f"{eff:.2f} %"],
        [f"{100*ERE:.2f} %"]
    ]
    row_labels = [" $V_{oc}$  ", "  $J_{sc}$", "  FF", r"  $\eta$", " ERE"]

    # Add table as inset
    table = ax1.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=None,
        loc='upper right',
        cellLoc='center',
        bbox=[0.2, 0.4, 0.3, 0.35]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.2)

    plt.show()

def plot_si_jv(iV, iJ, voltages, currents, Voc, Jsc, FF, eff):
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.rc('font', family='Arial', size=16)
    ax.plot(iV, iJ, 'o', label='iVoc vs iJ')
    ax.plot(voltages, currents*1e3, label='3-diode model, J-V')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Current Density (mA/cm$^2$)')
    ax.legend(loc='lower center')
    ax.grid(False)
    ax.set_xlim((0.35, 0.8))
    ax.set_ylim((-0.01, 45))

    # Right y-axis: log(current)
    ax2 = ax.twinx()
    ax2.plot(voltages, (np.abs(currents-Jsc)*1e3), color='tab:red', linestyle='--')
    ax2.set_ylabel('| $J - J_{sc}|$ (mA/cm$^2$)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_yscale('log')
    ax2.minorticks_on()


    cell_text = [
        [f"{Voc*1e3:.2f} mV"],
        [f"{Jsc*1e3:.2f} mA/cm$^2$"],
        [f"{100*FF:.2f} %"],
        [f"{eff:.2f} %"]
    ]
    row_labels = [" $V_{oc}$  ", "  $J_{sc}$", "  FF", r"  $\eta$"]

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=None,
        loc='upper right',
        cellLoc='center',
        bbox=[0.15, 0.55, 0.3, 0.35]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.2)

    plt.show()
    
    
## Transfer Matrix Method (TMM) for optical modeling


def I_mat(n1,n2):
	# transfer matrix at an interface
	r = (n1-n2)/(n1+n2)
	t = (2*n1)/(n1+n2)
	ret = np.array([[1,r],[r,1]],dtype=complex)
	ret = ret / t
	return ret

def L_mat(n,d,l):
	# propagation matrix
	# n = complex dielectric constant
	# d = thickness
	# l = wavelength
	xi = (2*np.pi*d*n)/l
	L = np.array( [ [ np.exp(complex(0,-1.0*xi)),0] , [0,np.exp(complex(0,xi))] ] )
	return L


def get_ntotal_from_excel(mat_name, lambdas, header=0, sheet_name=0):
	"""
	Read n and k values from a single Excel file for a given material index.

	Parameters:
	- excel_path: path to the Excel file
	- material_index: 0-based index of the material (0 = first material, 1 = second, etc.)
	- lambdas: array of wavelengths to interpolate over
	- header: row number to use as column headers (default = 0)
	- sheet_name: Excel sheet to read from (default = first sheet)

	Returns:
	- ntotal: list of complex refractive indices at the specified wavelengths
	"""
	# mat_name = 'SiO2'  # example material name

	excel_path = './data/Optical_Properties.xlsx'

	df = pd.read_excel(excel_path, header=0, sheet_name=0)
	col_index = next( (i for i, col in enumerate(df.columns) if mat_name in col),-1)

	if col_index == -1:
		raise ValueError(f"Material '{mat_name}' not found in the Excel file.")
	else:
		# Each material takes 2 columns: wavelength, n, k
		
		wavelengths = df.iloc[:, 0]
		n_values = df.iloc[:, col_index ]
		k_values = df.iloc[:, col_index +1]

		# Drop any rows with missing values
		valid = wavelengths.notnull() & n_values.notnull() & k_values.notnull()
		wavelengths = wavelengths[valid].to_numpy(dtype=float)
		n_values = n_values[valid].to_numpy(dtype=float)
		k_values = k_values[valid].to_numpy(dtype=float)

		# Interpolation
		int_n = interp1d(wavelengths, n_values, kind='linear', bounds_error=False, fill_value="extrapolate")
		int_k = interp1d(wavelengths, k_values, kind='linear', bounds_error=False, fill_value="extrapolate")

		n_interp = int_n(lambdas)
		k_interp = int_k(lambdas)

		ntotal = [complex(n, k) for n, k in zip(n_interp, k_interp)]
	return ntotal

def TMM_Tx_Ax(layers, t, lambdas, x_step, front_layers):
	"""
	Calculate the transmission through a stack of layers using the Transfer Matrix Method (TMM).
	
	Parameters:
	layers: list of layer names
	thicknesses: list of layer thicknesses in nm
	lambdas: array of wavelengths in nm
	x_step: grid spacing for cross section simulation in nm
	"""
   
	# initialize an array
	n = np.zeros((len(layers),len(lambdas)),dtype=complex)

	# load index of refraction for each material in the stack
	for i,l in enumerate(layers):
		ni = np.array(get_ntotal_from_excel(l,lambdas))
		n[i,:] = ni

	# calculate incoherent power transmission through substrate

	T_glass = abs((4.0*1.0*n[0,:])/((1+n[0,:])**2))
	R_glass = abs((1-n[0,:])/(1+n[0,:]))**2

	# calculate transfer marices, and field at each wavelength and position
	t[0] 		= 0
	t_cumsum	= np.cumsum(t)
	x_pos		= np.arange((x_step/2.0),sum(t),x_step)
	# get x_mat
	comp1	= np.kron(np.ones( (len(t),1) ),x_pos)
	comp2	= np.transpose(np.kron(np.ones( (len(x_pos),1) ),t_cumsum))
	x_mat 	= sum(comp1>comp2,0) 	# might need to get changed to better match python indices

	R		= lambdas*0.0
	T		= lambdas*0.0
	E		= np.zeros( (len(x_pos),len(lambdas)),dtype=complex )

	# start looping
	for ind,l in enumerate(lambdas):
		# calculate the transfer matrices for incoherent reflection/transmission at the first interface
		S = I_mat(n[0,ind],n[1,ind])
		for matind in np.arange(1,len(t)-1):
			mL = L_mat( n[matind,ind] , t[matind] , lambdas[ind] )
			mI = I_mat( n[matind,ind] , n[matind+1,ind])
			S  = np.asarray(np.asmatrix(S)*np.asmatrix(mL)*np.asmatrix(mI))
		R[ind] = abs(S[1,0]/S[0,0])**2
		T[ind] = abs((2/(1+n[0,ind])))/np.sqrt(1-R_glass[ind]*R[ind])
		
		# good up to here
		# calculate all other transfer matrices
		for material in np.arange(1,len(t)):
			xi = 2*np.pi*n[material,ind]/lambdas[ind]
			dj = t[material]

			x_indices	= np.nonzero(x_mat == material)
			x			= x_pos[x_indices]-t_cumsum[material-1]               
			# Calculate S_Prime
			S_prime		= I_mat(n[0,ind],n[1,ind])
			for matind in np.arange(2,material+1):
				mL = L_mat( n[matind-1,ind],t[matind-1],lambdas[ind] )
				mI = I_mat( n[matind-1,ind],n[matind,ind] )
				S_prime  = np.asarray( np.asmatrix(S_prime)*np.asmatrix(mL)*np.asmatrix(mI) )
			# Calculate S_dprime (double prime)
			S_dprime	= np.eye(2)
			for matind in np.arange(material,len(t)-1):
				mI	= I_mat(n[matind,ind],n[matind+1,ind])
				mL	= L_mat(n[matind+1,ind],t[matind+1],lambdas[ind])
				S_dprime = np.asarray( np.asmatrix(S_dprime) * np.asmatrix(mI) * np.asmatrix(mL) )
			# Normalized Electric Field Profile
			num = T[ind] * (S_dprime[0,0] * np.exp( complex(0,-1.0)*xi*(dj-x) ) + S_dprime[1,0]*np.exp(complex(0,1)*xi*(dj-x)))
			den = S_prime[0,0]*S_dprime[0,0]*np.exp(complex(0,-1.0)*xi*dj) + S_prime[0,1]*S_dprime[1,0]*np.exp(complex(0,1)*xi*dj)
			E[x_indices,ind] = num / den

	# overall Reflection from device with incoherent reflections at first interface
	Reflection = R_glass+T_glass**2*R/(1-R_glass*R)

	# Absorption coefficient in 1/cm
	a = np.zeros( (len(t),len(lambdas)) )
	for matind in np.arange(1,len(t)):
		a[matind,:] = ( 4 * np.pi * np.imag(n[matind,:]) ) / ( lambdas * 1.0e-7 )

	Absorption = np.zeros( (len(t),len(lambdas)) )
	for matind in np.arange(1,len(t)):
		Pos 		= np.nonzero(x_mat == matind)
		AbsRate 	= np.tile( (a[matind,:] * np.real(n[matind,:])) , (len(Pos),1)) * (abs(E[Pos,:])**2)
		Absorption[matind,:] = np.sum(AbsRate,1)*x_step*1.0e-7
	
	TotalTx=( 1 - Reflection - np.sum(Absorption[front_layers,:], axis=0))
 	
	return TotalTx, Absorption, Reflection, E, x_pos, n


def optic_electronic_efficiency(Si_Waf_thick, pvk_layer_id, R_tco, BB300K, layers, thicknesses, 
                                lambda_vec, AM15g, x_step, l_values, front_layers,
                                top_cell_density, bottom_cell_density, w, d, Vcell, RserInt):
    """
    Calculate the optical and electronic efficiency of a solar cell stack using TMM.
    
    Parameters:
    Si_Waf_thick: Thickness of the silicon wafer in cm.
    pvk_layer_id: Index of the perovskite layer in the layers list.
    R_tco: Series resistance of the TCO in ohms.cm^2.
    BB300K: Black body radiation at 300K.
    layers: List of layer names.
    thicknesses: List of layer thicknesses in nm.
    lambda_vec: Array of wavelengths in nm.
    AM15g: AM1.5 global spectrum.
    x_step: Grid spacing for cross section simulation in nm.
    l_values: Array of finger pitches in cm.
    front_layers: List of indices for front layers in the stack.
    top_cell_density: Density parameters for the top cell.
    bottom_cell_density: Density parameters for the bottom cell.
    w: Width of the cell in cm.
    d: Contact width in um.
    Vcell: Array of voltages to simulate.
    RserInt: Internal series resistance in ohms.cm^2.
    """

    TotalTx, Absorption, Reflection, E, x_pos, n= TMM_Tx_Ax(layers, thicknesses, lambda_vec, x_step, front_layers)
 
    Jsc_pvk= q* np.trapezoid(AM15g*Absorption[pvk_layer_id,:], lambda_vec) # A/cm2
    EQE_pvk=Absorption[pvk_layer_id, :]
    FilteredAM15g = AM15g * TotalTx  # Filtered AM15g spectrum after the top cell
    Jsc_Si, AxSi = Jsc_Si_fun(Si_Waf_thick, lambda_vec, FilteredAM15g)
    EQE_Si=FilteredAM15g*AxSi/AM15g
    
    J0_pvk=q * np.trapezoid(BB300K*EQE_pvk , lambda_vec)
    J0_Si=q * np.trapezoid(BB300K*EQE_Si , lambda_vec)
    
    # Unpack all values from top_cell and bottom_cell
    _, Rsh_pvk , J01d_pvk,  J0def_pvk, Rdef_pvk, J02_pvk = top_cell_density
    _, Rsh_Si, _, J0Aug_Si, J0n_Si, n_def = bottom_cell_density

    AAF = 1 - d / l_values  # Active Area Fraction, assuming d = 20 um contact width
    l, _, _, _, _ = find_optimal_pitch(J0_Si, J0n_Si, J0Aug_Si, Rsh_Si, Jsc_Si, l_values, R_tco, AAF)
    AAF = 1 - d / l  

    # Re-calculate the currents, based on the optimal pitch. Units of Amperes
    # perovskite top cell
    Isc_pvk = -AAF * w * (l / 2) *Jsc_pvk
    I0_pvk =  w * (l / 2) * J0_pvk
    I01d_pvk= ( w * (l / 2)) * J01d_pvk

    # silicon bottom cell
    Isc_Si = -AAF * w * (l / 2) *Jsc_Si
    I01_Si =  w * (l / 2) * J0_Si

    Rser = R_tco * (l**2/ (12)) + RserInt # series resistance in ohms.cm^2 [TCO + external]

    top_cell= [Isc_pvk, Rsh_pvk / ( w * (l / 2)), I0_pvk + I01d_pvk,  J0def_pvk * w * (l / 2), Rdef_pvk/ ( w * (l / 2)), J02_pvk* w * (l / 2)]
    bottom_cell = [Isc_Si, Rsh_Si/(w*(l/2)), I01_Si, J0Aug_Si* w * (l / 2) , J0n_Si* w * (l / 2), n_def]

    voltages, currents = simulate_circuit(top_cell, bottom_cell, Rser, w, l, Vcell)

    eff, Jsc, Voc, FF =calculate_efficiency(voltages, currents, False)

    return eff, Jsc, Voc, FF, TotalTx, Absorption, Reflection, E, x_pos, n



def Si_res(delta_n_q, ndop_q):
    """
    Interpolates resistivity values from an Excel file based on Delta_n and Ndop.
    Args:
        delta_n_q: Array of Delta_n values (cm^-3).
        ndop_q: Single value of Ndop values (cm^-3). [does not support array]

    """
    
    # Load Excel, skipping the first row (Ndop header), and keeping it separately
    df = pd.read_excel("./data/resistivity_Si.xlsx")

    # Extract Delta_n (column 0)
    delta_n_vals = df.iloc[:, 0].values

    # Extract Ndop values from header row (columns 1 onwards)
    ndop_vals = df.columns[1:6].astype(float)

    # Extract Rho matrix
    rho_matrix = df.iloc[:, 1:6].values  # shape: (n_delta_n, n_ndop)

    interp_func = RegularGridInterpolator(
        (delta_n_vals, ndop_vals),
        rho_matrix,
        bounds_error=False,  # optional: allows extrapolation
        fill_value=None      # extrapolates if bounds_error=False
    )

    # delta_n_q = np.linspace(1e10, 1.0e17, 100)
    ndop_q2 = ndop_q * np.ones_like(delta_n_q)

    # Combine into shape (100, 2): one row per point
    query_points = np.column_stack((delta_n_q, ndop_q2))

    # Interpolate
    rho_vals = interp_func(query_points)
    return rho_vals

    