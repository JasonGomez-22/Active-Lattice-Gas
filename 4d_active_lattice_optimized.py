import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from tqdm import tqdm
from numba import jit, prange
import os


plt.rcParams['animation.ffmpeg_path'] = r'C:\ProgramData\chocolatey\bin\ffmpeg.exe'  # Update this path if necessary

L = 800
D = 1
lambda_ = 40 / 100
gamma = 10 / 10000
rho_0 = 0.63
batch = 30_000_000

# mapping = {
#     -1: 2,
#     2: 1,
#     1: -2,
#     -2: -1,
#     0: 0
# }

mapping = np.array([0, -2, 1, -1, 2])   

N = int(rho_0 * L * L)

lattice = np.zeros((L, L), dtype=np.int8)

dt = 1 / (2 * D + lambda_ + gamma)
# time = 0

def fill_lattice():
    # time = 0
    lattice = np.zeros((L, L), dtype=np.int8)
    
    filled_spaces = np.random.choice(L * L, N, replace=False)
    for index in range(N):
        col = filled_spaces[index] % L
        row = filled_spaces[index] // L
        spin = np.random.choice([-2, -1, 1, 2])
        lattice[row][col] = spin

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
                lattice[row, col], lattice[row, (col+1) % L] = 0, 1
            elif lattice[row, col] == -1 and lattice[row, (col-1) % L] == 0:
                lattice[row, col], lattice[row, (col-1) % L] = 0, -1
            elif lattice[row, col] == 2 and lattice[(row-1) % L, col] == 0:
                lattice[row, col], lattice[(row-1) % L, col] = 0, 2
            elif lattice[row, col] == -2 and lattice[(row+1) % L, col] == 0:
                lattice[row, col], lattice[(row+1) % L, col] = 0, -2
        else:
            lattice[row, col] = mapping[lattice[row, col]]


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
cmaplist = [0 for i in range(5)]
cmaplist[0] = (1.0, .0, .0, 1.0)
cmaplist[1] = (.0, 1.0, .0, 1.0)
cmaplist[2] = (1.0, 1.0, 1.0, 1.0)
cmaplist[3] = (1.0, 1.0, .0, 1.0)
cmaplist[4] = (.0, .0, 1.0, 1.0)

# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, 5)

for i in tqdm(range(1_000)):
    update(lattice)
    im = plt.imshow(lattice, cmap=cmap, interpolation='nearest')
    ims.append([im])

plt.colorbar()
filename = get_next_filename( 'd:\\jason\\2024\BTP\\', '4_particle_', '.mp4')
ani = animation.ArtistAnimation(fig, ims, interval= 10, blit=True, repeat_delay=1000)

ani.save(filename, writer='ffmpeg')