import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from tqdm import tqdm
from numba import jit, prange
import os

plt.rcParams['animation.ffmpeg_path'] = r'C:\ProgramData\chocolatey\bin\ffmpeg.exe'  # Update this path if necessary

L = 400
D = 1
lambda_ = 5 / 100
gamma = 0.1 / 10000
rho_0 = 0.75
batch = 10_000_000_000

N = int(rho_0 * L * L)

lattice = np.zeros((L, L), dtype=np.int8)

dt = 1 / (2 * D + lambda_ + gamma)
time = 0

def fill_lattice():
    print("Filling lattice")
    
    lattice = np.zeros((L, L), dtype=int)

    filled_spaces = np.random.choice(L * L, N, replace=False)
    
    rows = filled_spaces // L
    cols = filled_spaces % L
    
    spins = np.random.choice([-1, 1], size=N)
    
    lattice[rows, cols] = spins

    print("Lattice filled")
    return lattice

@jit(nopython=True, parallel=True)
def update(lattice):
    for i in prange(batch):
        col = np.random.randint(0, L)
        row = np.random.randint(0, L)

        r = np.random.random()
        if r < D * dt:
            # Diffuse upwards
            lattice[(row-1) % L, col], lattice[row, col] = lattice[row, col], lattice[(row-1) % L, col]
        
        elif r < 2 * D * dt:
            # Diffuse left
            lattice[row, (col+1) % L], lattice[row, col] = lattice[row, col], lattice[row, (col+1) % L]

        elif r < (2 * D + lambda_) * dt:
            if lattice[row, col] == 1 and lattice[row, (col+1) % L] == 0:
                lattice[row, col], lattice[row, (col+1) % L] = lattice[row, (col+1) % L], lattice[row, col]
            elif lattice[row, col] == -1 and lattice[row, (col-1) % L] == 0:
                lattice[row, col], lattice[row, (col-1) % L] = lattice[row, (col-1) % L], lattice[row, col]
        else:
            lattice[row, col] = -lattice[row, col]

def get_next_filename(base_path, base_name, extension):
    i = 1
    while True:
        filename = f"{base_path}{base_name}{i}{extension}"
        if not os.path.exists(filename):
            return filename
        i += 1

fig = plt.figure()
lattice = fill_lattice()
cmaplist = [0 for i in range(3)]
cmaplist[0] = (1.0, .0, .0, 1.0)
cmaplist[1] = (1.0, 1.0, 1.0, 1.0)
cmaplist[2] = (.0, .0, 1.0, 1.0)

# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, 3)

time = 0
for i in tqdm(range(10)):
    print(np.sum(lattice), np.sum(np.abs(lattice)))
    update(lattice)
    plt.imshow(lattice, cmap=cmap, interpolation='nearest')
    time += dt * batch
    plt.title(f"Time: {time:.2f}")
    base_path = 'd:\\jason\\2024\\BTP\\normal_profile_new\\'
    filename = get_next_filename(base_path, 'normal_profile_', '.png')
    plt.savefig(filename.format(i), dpi=300)
    # ims.append([im])

    plt.figure()
    # vertical_profile = np.sum(lattice, axis=1)/L
    vertical_profile = np.sum(np.abs(lattice), axis=0) / L
    plt.plot(vertical_profile)
    plt.xlabel('Row')
    plt.ylabel('Sum of Spins')
    plt.ylim(0, 1.2)
    plt.title(f"Time: {time:.2f}")
    base_path = 'd:\\jason\\2024\\BTP\\vertical_profile_new\\'
    filename = get_next_filename(base_path, 'vertical_profile_', '.png')
    plt.savefig(filename, dpi=300)
    plt.close()

# plt.colorbar()
# base_path = 'd:\\jason\\2024\\BTP\\'
# time = 0  # Initialize time variable
# for i in range(len(ims)):
#     ims[i][0].set_array(lattice)
#     time += dt * batch
#     # plt.title(f"Time: {time:.2f}")
#     filename = get_next_filename(base_path, 'vertical_profile_', '.png')
#     plt.savefig(filename.format(i), dpi=300)

# ani.save(filename, writer='ffmpeg')