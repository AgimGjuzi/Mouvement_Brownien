import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from collections import defaultdict

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
time_step = 0.01       # Time step between frames (s)
max_lag = 0.8          # Max lag time (s)
max_lag_frames = int(max_lag / time_step)
coord_index = 0        # 0 = x-axis

# Physical constants
kBT = 0.0041419464
eta = 1e-3
a = 1.0
k_spring = 0.1

# === Choose r value you want to plot ===
target_r = 5.00  # 👈 Change this to the r value you want (e.g., 8.00, 15.00)

# === Find all files and group by r ===
file_list = glob.glob("outputs_trap/**/*.sphere_array.config", recursive=True)
r_groups = defaultdict(list)

for filename in file_list:
    match = re.search(r"r_(\d+\.\d+)", filename)
    if match:
        r_val = float(match.group(1))
        r_groups[r_val].append(filename)

if target_r not in r_groups:
    raise ValueError(f"No files found for r = {target_r:.2f}")

files = r_groups[target_r]

# === Load coordinates from file ===
def load_coordinates(filename):
    frames = []
    with open(filename) as f:
        while True:
            header = f.readline()
            if not header:
                break
            n = int(header.strip())
            frame = [list(map(float, f.readline().split()[:3])) for _ in range(n)]
            frames.append(frame)
    return np.array(frames)

# === Compute autocorrelation of a signal ===
def compute_autocorrelation(pos, max_lag_frames):
    pos -= np.mean(pos)
    lags = np.arange(-max_lag_frames, max_lag_frames + 1)
    corr = []
    for lag in lags:
        if lag < 0:
            corr.append(np.mean(pos[:lag] * pos[-lag:]))
        elif lag > 0:
            corr.append(np.mean(pos[lag:] * pos[:-lag]))
        else:
            corr.append(np.mean(pos * pos))
    return lags * time_step, np.array(corr)

# === Compute autocorrelation for all files, then average ===
autocorrelations = []

for file in files:
    coords = load_coordinates(file)
    q = coords[:, 0, coord_index]  # Particle 0, x-axis
    _, ac = compute_autocorrelation(q, max_lag_frames)
    autocorrelations.append(ac)

# Compute average autocorrelation
autocorr_avg = np.mean(autocorrelations, axis=0)
lags = np.arange(-max_lag_frames, max_lag_frames + 1) * time_step

# === Compute theory curve ===
tau = 6 * np.pi * eta * a / k_spring
C0 = kBT / k_spring
theory = C0 * np.exp(-np.abs(lags) / tau)

# === Plot ===
plt.figure(figsize=(6, 4))
plt.plot(lags[::5], autocorr_avg[::5], 'o', label="Simulation", markersize=8)
plt.plot(lags, theory, 'k--', label="Théorie")
plt.xlabel("$t$ (s)")
plt.ylabel("$Autocorrélation$ ($\mu m^2$)")
plt.legend()
plt.tight_layout()
plt.savefig(f"Cross Correlation/autocorrelation_avg_r_{target_r:.2f}.png")

