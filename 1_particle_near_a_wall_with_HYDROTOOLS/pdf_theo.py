import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

custom_params = {
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.markeredgecolor": "k",
    "lines.markeredgewidth": 1,
    "figure.dpi": 300,
    "text.usetex": False,
    "font.family": "serif",
    "axes.linewidth": 2,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.minor.width": 2,
    "ytick.minor.width": 2,
}
sns.set_theme(context="notebook", style="ticks", rc=custom_params)


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

t = np.concatenate([
    np.arange(1, 10, 1),
    np.arange(10, 100, 10),
    np.arange(100, 1000, 100),
    np.arange(1000, 10000, 1000)])



dt =  0.01
kT = 0.0041419464
repulsion_strength_wall = 0.01988134272
B = repulsion_strength_wall / kT
eta0 = 1e-3
a =  1.5
g = 0.00781499
l_b = kT/g
l_d =  0.021

z = np.linspace(a + 1e-9, 8, 1000)


def _Peq(z):
    if z <= a:  
        return 0
    else:
        z_surf = z - a 
        return np.exp(-(B * np.exp(-z_surf / l_d) + z_surf / l_b))
def Peq(z):
    z = np.asarray(z)
    P = np.vectorize(_Peq)(z)
    if P.ndim > 0:
        P = P / np.trapz(P, z)  # Normalisation
    return P

bins2 = np.geomspace(a + 1e-9, 8, 50)

def calcul_pdf(data, bins = 50, density = True):
    pdf, bins_edge = np.histogram(data, bins = bins, density = density)
    bins_center = (bins_edge[0:-1] + bins_edge[1:]) / 2
    return pdf, bins_center

def main():
    file_path = 'run_blobs.sphere_array.config'
    pos = load_config_file(file_path)
    pdf_z, bins = calcul_pdf(pos[:, 2], bins2)

    
    plt.figure(figsize=(8,5))
    plt.semilogy(bins, pdf_z,'o', label="Simulation",markersize = 10)
    plt.plot(z , Peq(z), color = "k",linewidth=2, zorder =10)
    plt.xlabel("$z$ ($\mathrm{\mu m}$)", labelpad=0.5)
    plt.ylabel("$P_{\mathrm{eq}}$ ($\mathrm{\mu m  ^{-1}}$)", labelpad=0.5)
    plt.legend()
    plt.savefig("pdf_semilog.png")
    
    plt.figure(figsize=(8,5))
    plt.plot(bins, pdf_z,'o', label="Simulation",markersize = 10)
    plt.plot(z , Peq(z), color = "k",linewidth=2, zorder =10, label = "Courbe théorique")
    plt.xlabel("$z$ ($\mathrm{\mu m}$)", labelpad=0.5)
    plt.ylabel("$P_{\mathrm{eq}}$ ($\mathrm{\mu m  ^{-1}}$)", labelpad=0.5)
    plt.text(7.5 , 0.25, 'a)' , fontsize = 20)
    plt.legend(frameon=False)
    plt.savefig("pdf.png")


if __name__ == "__main__":
    main()
