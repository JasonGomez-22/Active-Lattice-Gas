import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from tqdm import tqdm
from numba import cuda
import numba.cuda.random as curand
import os

plt.rcParams['animation.ffmpeg_path'] = r'C:\ProgramData\chocolatey\bin\ffmpeg.exe'  # Update this path if necessary

L = 4000
D = 1
lambda_ = 40 / 100
gamma = 10 / 10000
rho_0 = 0.65
batch = 100_000_000

N = int(rho_0 * L * L)

lattice = np.zeros((L, L), dtype=np.int8)

dt = 1 / (2 * D + lambda_ + gamma)

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

@cuda.jit
def update(lattice, dt, D, lambda_, L, batch, rng_states):
    i = cuda.grid(1)
    if i < batch:
        col = int(curand.xoroshiro128p_uniform_float32(rng_states, i) * L)
        row = int(curand.xoroshiro128p_uniform_float32(rng_states, i) * L)

        r = curand.xoroshiro128p_uniform_float32(rng_states, i)
        if r < D * dt:
            lattice[(row-1) % L, col], lattice[row, col] = lattice[row, col], lattice[(row-1) % L, col]
        
        elif r < 2 * D * dt:
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

@cuda.jit
def curand_init_kernel(rng_states, seed):
    idx = cuda.grid(1)
    if idx < rng_states.size:
        curand.init_xoroshiro128p_states(rng_states, seed)

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

# Allocate memory on the GPU
d_lattice = cuda.to_device(lattice)

threads_per_block = 256
blocks_per_grid = (batch + (threads_per_block - 1)) // threads_per_block

# Initialize random states
rng_states = curand.create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=1)

for i in tqdm(range(10_000)):
    update[blocks_per_grid, threads_per_block](d_lattice, dt, D, lambda_, L, batch, rng_states)
    print(np.sum(d_lattice.copy_to_host()), np.sum(np.abs(d_lattice.copy_to_host())))
    lattice = d_lattice.copy_to_host()
    im = plt.imshow(lattice, cmap=cmap, interpolation='nearest')
    ims.append([im])

plt.colorbar()
filename = get_next_filename('d:\\jason\\2024\\BTP\\', '2_particle_', '.mp4')
ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=1000)

ani.save(filename, writer='ffmpeg')