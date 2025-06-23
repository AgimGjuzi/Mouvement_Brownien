import matplotlib.pyplot as plt
import numpy as np

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
# Reuse your function to read positions
def read_positions(file_path, a=1.5):
    positions = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        Np = int(lines[0].strip())
        block_size = Np + 1
        num_blocks = len(lines) // block_size

        for block in range(num_blocks):
            start = block * block_size + 1
            for i in range(Np):
                parts = lines[start + i].strip().split()
                if len(parts) >= 3:
                    x, y, z = map(float, parts[:3])
                    if z > a:
                        positions.append([x, y, z])

    return Np, np.array(positions)
# Load data
file_path = 'run_blobs.sphere_array.config'
Np, positions = read_positions(file_path)
Nstep = int(len(positions) / Np)
frames_data = [np.array(positions[i * Np:(i + 1) * Np], dtype=float) for i in range(Nstep)]

# Extract Z for particle 0 over time
z_traj = [frame[0][2] for frame in frames_data]
time = np.arange(Nstep)

# Plot
plt.figure(figsize=(8,5))
plt.plot(time, z_traj)
plt.xlabel("$t$ (s)")
plt.ylabel("$z$ ($\mathrm{\mu m }$)")
plt.savefig('Trajectoire_z_zoomed.png')
