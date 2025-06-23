import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["lines.markeredgecolor"] = "k"
mpl.rcParams["lines.markeredgewidth"] = 1.5
mpl.rcParams["figure.dpi"] = 200
from matplotlib import rc
rc("font", family="serif")
rc("xtick", labelsize="medium")
rc("ytick", labelsize="medium")
rc("axes", labelsize="large")

a = 1.5

# Load trajectory from config file (1 particle, 2-line blocks)
def load_config_file(file_path):
    positions = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for i in range(1, len(lines), 2):
        parts = lines[i].strip().split()
        if len(parts) >= 3:
            x, y, z = map(float, parts[:3])
            if z > a:
                positions.append([x, y, z])

    return np.array(positions)



def movmin(z, window):
    result = np.empty_like(z)
    start_pt = 0
    end_pt = int(np.ceil(window / 2))

    for i in range(len(z)):
        if i < int(np.ceil(window / 2)):
            start_pt = 0
        if i > len(z) - int(np.ceil(window / 2)):
            end_pt = len(z)
        result[i] = np.min(z[start_pt:end_pt])
        start_pt += 1
        end_pt += 1

    return result


def main():
    file_path = 'run_blobs.sphere_array.config'
    print("Loading trajectory from:", file_path)
    pos = load_config_file(file_path)
    print(f"Loaded {len(pos)} steps.")

    m = movmin(pos[:, 2], 1000)

    # Compute histogram and peak
    counts, bin_edges = np.histogram(m, bins=100, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    z_pic = bin_centers[np.argmax(counts)]
    print("Peak of PDF z (most probable height):", z_pic)

    # Plot histogram and peak line
    plt.figure()
    plt.hist(m, bins=50, density=True, alpha=0.4, label="PDF z")
    plt.axvline(z_pic, color='red', linestyle='--', label=f'Z_pic ≈ {z_pic - a:.3f} µm')
    plt.xlabel("z ($\mathrm{\mu m }$)")
    plt.ylabel('PDF')
    plt.legend()
    plt.savefig("pdf.minima.png")


if __name__ == "__main__":
    main()