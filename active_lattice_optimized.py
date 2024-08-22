import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from tqdm import tqdm
from numba import jit, prange
import os

plt.rcParams['animation.ffmpeg_path'] = r'C:\ProgramData\chocolatey\bin\ffmpeg.exe'  # Update this path if necessary

L = 4000
D = 1
lambda_ = 40 / 100
gamma = 10 / 10000
rho_0 = 0.65
batch = 1_000_000

N = int(rho_0 * L * L)

lattice = np.zeros((L, L), dtype=np.int8)

dt = 1 / (2 * D + lambda_ + gamma)
time = 0

def fill_lattice():
    global time
    time = 0
    print("Filling lattice")
    
    lattice = np.zeros((L, L), dtype=np.int8)

    filled_spaces = np.random.choice(L * L, N, replace=False)
    
    rows = filled_spaces // L
    cols = filled_spaces % L
    
    spins = np.random.choice([-1, 1], size=N)
    
    lattice[rows, cols] = spins

    print("Lattice filled")
    return lattice

@jit(nopython=True, fastmath=True)
def update(lattice, rows, cols, rs):
    # cols = np.random.randint(0, L, batch).astype(np.uint16)
    # rows = np.random.randint(0, L, batch).astype(np.uint16)
    # rs = np.random.random(batch)
    for i in range(batch):

        col = cols[i]
        row = rows[i]

        r = rs[i]
        if r < D * dt:
            # Diffuse upwards   
            lattice[(row-1) % L, col], lattice[row, col] = lattice[row, col], lattice[(row-1) % L, col]
        
        elif r < 2 * D * dt:
            # Diffuse left
            lattice[row, (col+1) % L], lattice[row, col] = lattice[row, col], lattice[row, (col+1) % L]

        elif r < (2 * D + lambda_) * dt:
            if lattice[row, col] == 1 and lattice[row, (col+1) % L] == 0:
                lattice[row, col], lattice[row, (col+1) % L] = 0, 1
            elif lattice[row, col] == -1 and lattice[row, (col-1) % L] == 0:
                lattice[row, col], lattice[row, (col-1) % L] = 0, -1
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
ims = []
lattice = fill_lattice()
cmaplist = [0 for i in range(3)]
cmaplist[0] = (1.0, .0, .0, 1.0)
cmaplist[1] = (1.0, 1.0, 1.0, 1.0)
cmaplist[2] = (.0, .0, 1.0, 1.0)


# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, 3)

for i in tqdm(range(100)):
    rows = np.random.randint(0, L, batch, dtype=np.uint16)
    cols = np.random.randint(0, L, batch, dtype=np.uint16)
    rs = np.random.random(batch)
    update(lattice, rows, cols, rs)
    im = plt.imshow(lattice, cmap=cmap, interpolation='nearest')
    # print(np.sum(np.abs(lattice)))
    ims.append([im])

plt.colorbar()
filename = get_next_filename( 'd:\\jason\\2024\BTP\\', '2_particle_', '.mp4')
ani = animation.ArtistAnimation(fig, ims, interval= 10, blit=True, repeat_delay=1000)

ani.save(filename, writer='ffmpeg')