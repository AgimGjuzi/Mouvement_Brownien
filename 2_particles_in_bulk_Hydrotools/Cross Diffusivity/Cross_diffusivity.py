#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from collections import defaultdict
from scipy.optimize import curve_fit
import os

from matplotlib import rc
import seaborn as sns
custom_params = {
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.markeredgecolor": "k",
    "lines.markeredgewidth": 1,
    "figure.dpi": 200,
    "text.usetex": False,
    "font.family": "serif",
    "axes.linewidth": 2,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.minor.width": 2,
    "ytick.minor.width": 2,
}
sns.set_theme(context="notebook", style="ticks", rc=custom_params)

# ---- Settings ----
dt = 0.01               # time step (s)
a = 1.0                 # particle radius (μm)
eta = 1e-3              # viscosity (Pa·s)
kT = 0.0041419464       # thermal energy (pN·μm)
D0 = kT / (6 * np.pi * eta * a)  # self-diffusivity
coord = 0              # x = 0, y = 1, z = 2

# ---- Load coordinate file (xyz positions) ----
def load_coordinates(filename):
    frames = []
    with open(filename) as f:
        while True:
            header = f.readline()
            if not header:
                break
            n = int(header.strip())
            coords = [list(map(float, f.readline().split()[:3])) for _ in range(n)]
            frames.append(coords)
    return np.array(frames)  # shape: (nframes, nparticles, 3)

# ---- Compute cross diffusivity D12^‖ ----
def cross_diffusivity(pos1, pos2):
    disp1 = np.diff(pos1)
    disp2 = np.diff(pos2)
    return 0.5 * np.mean(disp1 * disp2) / dt

# ---- Find all output files ----
files = sorted(glob.glob("outputs_trap/**/*.sphere_array.config", recursive=True))
if not files:
    raise RuntimeError("No simulation files found")

# ---- Group files by r value ----
groups = defaultdict(list)
for f in files:
    match = re.search(r"r_(\d+\.\d+)", f)
    if match:
        r_val = float(match.group(1))
        groups[r_val].append(f)

# ---- Compute D12 and errors for each r ----
r_vals = []
D12_vals = []
err_D12 = []  # error on D12 (std dev across runs)
err_r = []    # error on r (std dev across runs)

for r_nom, file_list in sorted(groups.items()):
    D_list = []  # stores D12 from each run
    r_list = []  # stores average r from each run

    for f in file_list:
        xyz = load_coordinates(f)
        pos1 = xyz[:, 0, coord]
        pos2 = xyz[:, 1, coord]

        # Calculate D12^‖ from each run
        D = cross_diffusivity(pos1, pos2)
        D_list.append(D)

        # Compute mean inter-particle distance for this run
        r = np.linalg.norm(xyz[:, 0, :] - xyz[:, 1, :], axis=1)
        r_list.append(np.mean(r))

    # Average values over runs
    r_vals.append(np.mean(r_list))
    D12_vals.append(np.mean(D_list))

    # Error bars:
    # → std deviation across runs
    err_D12.append(np.std(D_list) if len(D_list) > 1 else 0.0)  # error on D12^‖
    err_r.append(np.std(r_list) if len(r_list) > 1 else 0.0)    # error on r

# Convert to arrays
r_vals = np.array(r_vals)
D12_vals = np.array(D12_vals)
err_D12 = np.maximum(err_D12, 1e-12)  # avoid zero division
err_r = np.array(err_r)

# ---- Self-diffusivity (D11) from sample file ----
xyz_sample = load_coordinates(files[0])
q_sample = xyz_sample[:, 1, coord]
D11 = 0.5 * np.mean(np.diff(q_sample)**2) / dt

# ---- Theory: analytical D12^‖ in bulk ----
def D_theory(r):
    return D0 * ((3 * a) / (2 * r) - (a / r)**3)

r_plot = np.linspace(r_vals.min(), r_vals.max(), 300)

# ---- Fit: D12‖ = A / r^n ----
def model(r, A, n):
    return A * r**(-n)

popt, pcov = curve_fit(model, r_vals, D12_vals,
                       sigma=err_D12, absolute_sigma=True,
                       p0=[np.max(D12_vals), 3.0],
                       bounds=([0, 0], [np.inf, 10]))
A_fit, n_fit = popt
A_err, n_err = np.sqrt(np.diag(pcov))
scale = np.min(D12_vals)/np.min( D_theory(r_plot))
# ---- Plot ----
plt.figure(figsize=(8, 5))
plt.errorbar(r_vals, D12_vals, xerr=err_r, yerr=err_D12,
             fmt='o', markersize=8, capsize=4, label="Simulation")

plt.plot(r_plot,scale *  D_theory(r_plot), 'b--', label="Théorie")

plt.plot(r_plot, model(r_plot, A_fit, n_fit), 'k-', lw=1.5,
         label=fr"Fit: $A/r^n$, $n$ = {n_fit:.2f} ± {n_err:.2f}")

plt.xlabel("$r$ ($\mu$m)")
plt.ylabel("Diffusivité croisée $D_{12}^\parallel$ ($\mu$m$^2$/s)")
plt.legend()
plt.tight_layout()

# ---- Save plot ----
os.makedirs("Cross Diffusivity", exist_ok=True)
plt.savefig("Cross Diffusivity/cross_diffusivity_vs_r_fixed.png", dpi=300)
