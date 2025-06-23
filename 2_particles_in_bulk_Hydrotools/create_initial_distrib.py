import sys
import numpy as np

if __name__ == '__main__':
    r = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    a = 1
    Np = 2

    pos = np.array([
        [0, 0, 0],
        [r, 0, 0]
    ])

    quat = np.concatenate((np.ones((Np, 1)), np.zeros((Np, 3))), axis=1)
    to_save = np.concatenate((pos, quat), axis=1)

    np.savetxt("sphere_array.clones", to_save, header=str(Np), comments='')
