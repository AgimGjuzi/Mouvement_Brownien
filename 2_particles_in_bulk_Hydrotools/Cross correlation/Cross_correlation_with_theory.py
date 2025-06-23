import numpy as np
import matplotlib.pyplot as plt
import glob, re
from collections import defaultdict
from matplotlib import cm

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

# === Simulation parameters ===
dt = 0.01
max_lag = 1.5
max_lag_frames = int(max_lag / dt)
kBT = 0.0041419464
eta = 1e-3
a = 1.0
k = 0.1
coord = 0  # x-axis
p1, p2 = 0, 1

# === Find all config files and group them by r ===
file_list = glob.glob("outputs_trap/**/*.sphere_array.config", recursive=True)
r_groups = defaultdict(list)
for f in file_list:
    match = re.search(r"r_(\d+\.\d+)", f)
    if match:
        r_val = float(match.group(1))
        r_groups[r_val].append(f)

if not r_groups:
    raise RuntimeError("❌ No files found.")

# === Load coordinates from file ===
def load_coords(fname):
    coords = []
    with open(fname) as f:
        while True:
            header = f.readline()
            if not header:
                break
            n = int(header.strip())
            frame = [list(map(float, f.readline().split()[:3])) for _ in range(n)]
            coords.append(frame)
    return np.array(coords)

# === Cross-correlation function ===
def cross_corr(x1, x2):
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
    return lags * dt, np.array(corr)

# === Rotne–Prager theory function ===
def theory_corr(t, a, r, eta, k, kBT):
    tau = 6 * np.pi * eta * a / k
    eps = (3 * a / (4 * r)) * (2 - (4 * a**2) / (3 * r**2))
    t = np.abs(t)
    return 0.5 * (kBT / k) * (np.exp(-t * (1 + eps) / tau) - np.exp(-t * (1 - eps) / tau))

# === Plotting ===
plt.figure(figsize=(6, 4))
colors = cm.GnBu(np.linspace(0.4, 0.9, len(r_groups)))

for color, (r, files) in zip(colors, sorted(r_groups.items())):
    corrs, rs = [], []

    for f in files:
        coords = load_coords(f)
        x1 = coords[:, p1, coord]
        x2 = coords[:, p2, coord]
        r_vals = np.linalg.norm(coords[:, p1] - coords[:, p2], axis=1)
        rs.append(np.mean(r_vals))
        _, c = cross_corr(x1, x2)
        corrs.append(c)

    r_mean = np.mean(rs)
    avg_corr = np.mean(corrs, axis=0)
    lags = np.arange(-max_lag_frames, max_lag_frames + 1) * dt
    t_theory = np.linspace(-max_lag, max_lag, 500)
    theory = theory_corr(t_theory, a, r_mean, eta, k, kBT)

    # Plot simulation
    plt.plot(lags[::4], avg_corr[::4], 'o', markersize=8, color=color, label=f"Sim r={r_mean:.2f}")

    # Plot theory
    plt.plot(t_theory, theory, 'k-')

# === Labels and save ===
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("$t$ (s)")
plt.ylabel("$Corrélation croisée$ (μm²)")
plt.text(1, -0.006, 'b)', fontsize=20)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("Cross Correlation/cross_corr_with_epsilon_rpy.png", dpi=300)
print("✅ Saved: cross_corr_with_epsilon_rpy.png")
