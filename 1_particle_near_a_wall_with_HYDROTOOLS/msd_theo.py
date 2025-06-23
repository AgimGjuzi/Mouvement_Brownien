import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
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
    np.arange(1000, 50000, 2000)])
z = np.linspace(a + 1e-6, 5, 10000)

dt =  0.01
kT = 0.0041419464
repulsion_strength_wall = 0.01988134272
B = repulsion_strength_wall / kT
eta0 = 1e-3
a =  1.5
g =  0.00781499
l_b = kT/g
l_d =  0.021

D0 = kT / (6 * np.pi * eta0 * a)

def Dz_z(z):
    result =D0* ((
                1
                - (9 / 8) * (a / (z ))
                + (1 / 2) * (a / (z )) ** 3
                - (1 / 8) * (a / (z)) ** 5
                ))
    return result

def Dz_pade(z):
    etaz = eta0 * (6 * z ** 2 + 9 * a * z + 2 * a ** 2) / (6 * z ** 2 + 2 * a * z)
    return kT / (6 * np.pi * etaz * a)

def Dx_f(z):
    result = D0* ((1
                - (9 / 16) * (a / (z+a))
                + (1 / 8) * (a / (z+a))** 3
                - (45/236) * (a / (z+a))** 4
                - (1 / 16) * (a / (z+a))** 5
                ))
    return result

def Dxy_z(z):
    result =D0* ((
                1
                - (9 / 16) * (a / (z))
                + (1 / 8) * (a / (z)) ** 3
                - (1 / 16) * (a / (z)) ** 5
                ))
    return result

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



P = Peq(z)
Dx_vals = Dxy_z(z)
Dz_vals = Dz_z(z)
D_x_mean = simps(Dx_vals * P, z)
D_z_mean = simps(Dz_vals * P, z)
MSD_x_m = 2 * D_x_mean * t * dt 
MSD_z_m = 2 * D_z_mean * t * dt 

print(f"⟨Dx(z)⟩ = {D_x_mean/(D0)} ")
print(f"⟨Dz(z)⟩ = {D_z_mean/(D0)} ")

msd_th = simps((z - a)**2 * P, z)
print(f"MSD from wall: {msd_th:.4f}")

# 1D MSD function
def MSD(x, t):
    msd = np.zeros(len(t))
    for n, i in enumerate(t):
        msd[n] = np.nanmean((x[:-i] - x[i:]) ** 2)
    return msd

bins2 = np.geomspace(a + 1e-9, 8, 50)

def main():
   
    file_path = 'run_blobs.sphere_array.config'
   
    pos = load_config_file(file_path)

    msd_x = MSD(pos[:, 0], t)
    msd_y = MSD(pos[:, 1], t)
    msd_z = MSD(pos[:, 2], t)
   

    plateau = np.mean(msd_z[t > 2e3])
    print('Plateau : ', plateau)

    plt.figure(figsize=(8,5))

    plt.loglog(t, msd_x,'o',markersize = 10, label="MSD x")
    plt.loglog(t, msd_y,'o',markersize = 10, label="MSD y")
    plt.loglog(t, msd_z,'o',markersize = 10, label="MSD z")
    plt.plot(t,MSD_x_m, color = "k",linewidth=2, zorder =10, label = 'Courbe théorique')
    plt.plot(t,MSD_z_m,color = "k",linewidth=2, zorder =10)
    plt.plot(t,[msd_th]* len(t),'--',color = "k",linewidth=2, zorder =10, label = 'Plateau')
    

    plt.xlabel("$t$ (s)")
    plt.ylabel("$MSD$ ($\mathrm{\mu m  ^{2}}$)")
    plt.text(3e4,2e-3, 'b)', fontsize = 20)
    plt.legend(frameon = False)
    plt.savefig("msd_théorique.png")





if __name__ == "__main__":
    main()
