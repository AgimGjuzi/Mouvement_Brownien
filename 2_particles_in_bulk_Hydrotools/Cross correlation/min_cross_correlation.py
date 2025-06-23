import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from collections import defaultdict
from scipy.optimize import curve_fit

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

# === Settings ===
dt = 0.01                # time step
max_lag = 0.8            # max lag time
max_lag_frames = int(max_lag / dt)
kBT = 0.00414            # thermal energy
eta = 1e-3               # viscosity
a = 1.0                  # particle radius
k_spring = 0.1           # trap stiffness
coord = 0                # x direction

# === Find files grouped by r ===
files = glob.glob("outputs_trap/**/*.sphere_array.config", recursive=True)
r_groups = defaultdict(list)

for f in files:
    m = re.search(r"r_(\d+\.\d+)", f)
    if m:
        r = float(m.group(1))
        r_groups[r].append(f)

if not r_groups:
    raise RuntimeError("No files found.")

# === Helper functions ===
def load_coords(filename):
    data = []
    with open(filename) as f:
        while True:
            header = f.readline()
            if not header:
                break
            n = int(header.strip())
            frame = [list(map(float, f.readline().split()[:3])) for _ in range(n)]
            data.append(frame)
    return np.array(data)

def cross_correlation(x1, x2):
    x1 -= np.mean(x1)
    x2 -= np.mean(x2)
    lags = np.arange(-max_lag_frames, max_lag_frames + 1)
    corr = []
    for lag in lags:
        if lag < 0:
            corr.append(np.mean(x1[:lag] * x2[-lag:]))
        elif lag > 0:
            corr.append(np.mean(x1[lag:] * x2[:-lag]))
        else:
            corr.append(np.mean(x1 * x2))
    return np.array(corr)

# === Collect data: min(C12) vs a/r ===
a_over_r = []
min_corrs = []
err_a_over_r = []
err_min_corrs = []

for r_val, flist in sorted(r_groups.items()):
    mins = []
    r_means = []

    for f in flist:
        coords = load_coords(f)
        x1 = coords[:, 0, coord]
        x2 = coords[:, 1, coord]
        r = np.linalg.norm(coords[:, 0, :] - coords[:, 1, :], axis=1)

        corr = cross_correlation(x1, x2)
        mins.append(np.min(corr))
        r_means.append(np.mean(r))

    r_avg = np.mean(r_means)
    r_std = np.std(r_means)
    a_r = a / r_avg
    a_r_err = a * r_std / (r_avg ** 2)

    a_over_r.append(a_r)
    min_corrs.append(np.mean(mins))
    err_a_over_r.append(a_r_err)
    err_min_corrs.append(np.std(mins))

# === Fit to linear model: min(C12) = -A * a/r ===
def model(x, A): return -A * x

a_over_r = np.array(a_over_r)
min_corrs = np.array(min_corrs)
err_a_over_r = np.array(err_a_over_r)
err_min_corrs = np.array(err_min_corrs)

popt, pcov = curve_fit(model, a_over_r, min_corrs, sigma=err_min_corrs, absolute_sigma=True)
A_fit = popt[0]
A_err = np.sqrt(pcov[0, 0])
A_theory = (3 * kBT) / (2 * k_spring * np.e)

# === Plot result ===
plt.figure(figsize=(8, 5))
plt.errorbar(a_over_r, min_corrs, xerr=err_a_over_r, yerr=err_min_corrs,
             fmt='o', label="Simulation", capsize=4)
plt.plot(a_over_r, model(a_over_r, A_fit), 'b--', label=f"Fit: A = {A_fit:.3f}")
plt.plot(a_over_r, model(a_over_r, A_theory), 'k-', label=f"Théorie: A = {A_theory:.3f}")
plt.xlabel("a / r")
plt.ylabel("Minimum de la Corrélation croisée  ($\mu m^2$)")
plt.legend()
plt.tight_layout()
plt.savefig("Cross Correlation/min_cross_corr_vs_ar.png")

