import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import rc

# === Plotting Settings ===
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["lines.markeredgecolor"] = "k"
mpl.rcParams["lines.markeredgewidth"] = 1.5
mpl.rcParams["figure.dpi"] = 200
rc("font", family="serif")
rc("xtick", labelsize="medium")
rc("ytick", labelsize="medium")
rc("axes", labelsize="large")

# === Read Data ===
def read_positions(file_path):
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
                    positions.append([x, y, z])

    return Np, np.array(positions)

# === Load and Organize Data ===
file_path = 'r_8.00_run1_blobs.sphere_array.config'
Np, positions = read_positions(file_path)
Nstep = len(positions) // Np
frames_data = [positions[i * Np:(i + 1) * Np] for i in range(Nstep)]

# === Extract and Plot Z Trajectories ===
z_traj1 = [frame[0][2] for frame in frames_data]
z_traj2 = [frame[1][2] for frame in frames_data]
time = np.arange(Nstep)

plt.figure()
plt.plot(time, z_traj1)
plt.plot(time, z_traj2)
plt.xlabel("Time step")
plt.ylabel("z position")
plt.title("Z Trajectory of Particle 1")
plt.savefig('Trajectoire_z.png')
