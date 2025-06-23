#!/usr/bin/env python3
"""
Analyse two-particle trajectories, measure the
cross-diffusivity D12‖(r) and fit A / r^n.

Modifié pour tracer D12‖ en fonction de r au lieu de a/r.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob, re, warnings
from collections import defaultdict
from scipy.optimize import curve_fit, OptimizeWarning

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
# ───────── user-tuneable constants ─────────
particle1, particle2 = 0, 1
coord_index          = 0    # 0 = x
dt                   = 0.01    # s  (frame interval)
a                    = 1.0     # μm  (sphere radius for a/r)

# ───────── helper functions ─────────
def load_coordinates(fname):
    """Return ndarray [nframes, nparticles, 3] from a .config file."""
    frames = []
    with open(fname) as f:
        while True:
            header = f.readline()
            if not header:
                break
            n = int(header.strip())
            frames.append([list(map(float, f.readline().split()[:3])) for _ in range(n)])
    return np.asarray(frames)


def cross_diffusivity(q1, q2, dt):
    """D12‖  =  ⟨Δq1 Δq2⟩ / (2 dt)"""
    d1, d2 = np.diff(q1), np.diff(q2)
    return 0.5 * np.mean(d1 * d2) / dt


def power_law_signed_r(r, A, n, sign):
    """Signed power law for r."""
    return sign * A * r**(-n)


# ───────── gather files grouped by nominal r ─────────
file_list = sorted(glob.glob("outputs/**/*.sphere_array.config", recursive=True))
r_groups  = defaultdict(list)
for f in file_list:
    if m := re.search(r"r_(\d+\.\d+)", f):
        r_groups[float(m.group(1))].append(f)

if not r_groups:
    raise RuntimeError("❌ No trajectory files (pattern r_*.sphere_array.config) found.")

# ───────── crunch every r ─────────
r_vals, D12, err_r, err_y, h_vals = [], [], [], [], []

for r_nom, flist in sorted(r_groups.items()):
    Dvals, r_means = [], []

    for f in flist:
        xyz = load_coordinates(f)
        q1  = xyz[:, particle1, coord_index]
        q2  = xyz[:, particle2, coord_index]
        Dvals.append(cross_diffusivity(q1, q2, dt))

        disp = xyz[:, particle1, :] - xyz[:, particle2, :]
        r_means.append(np.mean(np.linalg.norm(disp, axis=1)))

    # ensemble average for this r
    D12_mean = np.mean(Dvals)
    D12.append(D12_mean)
    err_y.append(np.std(Dvals) if len(Dvals) > 1 else 0.0)

    r_mean, eta_r = np.mean(r_means), np.std(r_means)
    r_vals.append(r_mean)
    err_r.append(eta_r)
    z_series = xyz[:, particle1, 2]
    h_vals.append(np.mean(z_series))

r_vals = np.asarray(r_vals)
D12    = np.asarray(D12)
err_r  = np.asarray(err_r)
err_y  = np.maximum(np.asarray(err_y), 1e-12)   # prevent zero σ

# ───────── fit power law D12 = ± A / r^n ─────────
sign_branch = np.sign(D12[np.argmax(np.abs(D12))])
if sign_branch == 0:
    sign_branch = 1.0

def model_r(r, A, n):
    return power_law_signed_r(r, A, n, sign_branch)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=OptimizeWarning)
    popt, pcov = curve_fit(
        model_r, r_vals, D12,
        sigma=err_y, absolute_sigma=True,
        p0=[np.abs(D12).max(), 3.0],
        bounds=([0.0, 0.0], [np.inf, 10.0]),
        maxfev=1000000,
    )

A_fit, n_fit = popt
A_err, n_err = np.sqrt(np.diag(pcov))
n_err_txt = rf"{n_err:.2f}" if np.isfinite(n_err) else r"\infty"

# ───────── Theoretical prediction ─────────
kT =  0.0041419464
eta = 1.0e-3
a = 1
D0 = kT / (6*np.pi*eta*a)

def D_collective_parallel(r, a, h):
    xi = 4 * (h**2) / (r**2)
    return D0 * ( 3/2 * a/r * (1 - (1 + xi + 3/2*xi**2)/(1 + xi)**(5/2)))


h_phys = np.mean(h_vals)
r_plot = np.linspace(r_vals.min(), r_vals.max(), 400)
D_theo = D_collective_parallel(r_vals, a, h_phys)
D_theo_curve = D_collective_parallel(r_plot, a, h_phys)

# ───────── Plot ─────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(r_vals, D12, xerr=err_r, yerr=err_y,
            fmt='o',markersize = 10, capsize=4, ecolor='gray',
            label=rf"simulation $D_{{12}}^{{\parallel}}$")

ax.plot(r_plot, model_r(r_plot, *popt), 'k-', lw=1.2,
        label=rf"fit: $A/r^n$,  $n={n_fit:.2f}\,\pm\,{n_err_txt}$")

ax.plot(r_plot, D_theo_curve, 'b--', label ='Theory')

ax.set_xlabel(r"$r\;(\mu\mathrm{m})$")
ax.set_ylabel(r"$D_{12}^{\parallel}\;(\mu\mathrm{m}^{2}\,\mathrm{s}^{-1})$")
ax.set_title('Cross_diffusivity near a wall')
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("cross_diffusivity_vs_r.png", dpi=300)

print(D12)
