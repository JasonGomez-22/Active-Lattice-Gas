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
batch = 10_000_000

N = int(rho_0 * L * L)

lattice = np.zeros((L, L), dtype=np.int8)

dt = 1 / (2 * D + lambda_ + gamma)
# time = 0

def fill_lattice():
    # time = 0
    print("Filling lattice")
    
    lattice = np.zeros((L, L), dtype=np.int8)

    filled_spaces = np.random.choice(L * L, N, replace=False)
    
    rows = filled_spaces // L
    cols = filled_spaces % L
    
    spins = np.random.choice([-1, 1], size=N)
    
    lattice[rows, cols] = spins

    print("Lattice filled")
    return lattice

@jit(nopython=True, parallel=False, fastmath=True)
def update(lattice):
    cols = np.random.randint(0, L, batch)
    rows = np.random.randint(0, L, batch)
    r = np.random.random(batch)

    # Diffuse upwards
    mask_up = r < D * dt
    lattice[(rows[mask_up]-1) % L, cols[mask_up]], lattice[rows[mask_up], cols[mask_up]] = lattice[rows[mask_up], cols[mask_up]], lattice[(rows[mask_up]-1) % L, cols[mask_up]]

    # Diffuse left
    mask_left = (r >= D * dt) & (r < 2 * D * dt)
    lattice[rows[mask_left], (cols[mask_left]+1) % L], lattice[rows[mask_left], cols[mask_left]] = lattice[rows[mask_left], cols[mask_left]], lattice[rows[mask_left], (cols[mask_left]+1) % L]

    # Reaction
    mask_react = (r >= 2 * D * dt) & (r < (2 * D + lambda_) * dt)
    mask_react_right = mask_react & (lattice[rows, cols] == 1) & (lattice[rows, (cols+1) % L] == 0)
    mask_react_left = mask_react & (lattice[rows, cols] == -1) & (lattice[rows, (cols-1) % L] == 0)
    lattice[rows[mask_react_right], cols[mask_react_right]], lattice[rows[mask_react_right], (cols[mask_react_right]+1) % L] = 0, 1
    lattice[rows[mask_react_left], cols[mask_react_left]], lattice[rows[mask_react_left], (cols[mask_react_left]-1) % L] = 0, -1

    # Flip spin
    mask_flip = r >= (2 * D + lambda_) * dt
    lattice[rows[mask_flip], cols[mask_flip]] = -lattice[rows[mask_flip], cols[mask_flip]]

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

for i in tqdm(range(1_000)):
    update(lattice)
    im = plt.imshow(lattice, cmap=cmap, interpolation='nearest')
    # print(np.sum(np.abs(lattice)))
    ims.append([im])

plt.colorbar()
filename = get_next_filename( 'd:\\jason\\2024\BTP\\', '2_particle_', '.mp4')
ani = animation.ArtistAnimation(fig, ims, interval= 10, blit=True, repeat_delay=1000)

ani.save(filename, writer='ffmpeg')