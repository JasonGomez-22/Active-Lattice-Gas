import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from tqdm import tqdm
from numba import jit, prange
import cProfile
import pstats

# Set the cache directory
os.environ['NUMBA_CACHE_DIR'] = 'numba_cache'

L = 4000
D = 1
lambda_ = 40 / 100
gamma = 10 / 10000
rho_0 = 0.65
batch = 100_000_000

dt = 1 / (2 * D + lambda_ + gamma)

@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True)
def update(lattice, rows, cols, rs, batch, L, D, dt, lambda_):
    for i in prange(batch):
        col = cols[i]
        row = rows[i]
        r = rs[i]
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

def fill_lattice():
    return np.zeros((L, L), dtype=np.int8)

def main():
    fig = plt.figure()
    ims = []
    lattice = fill_lattice()
    cmaplist = [0 for i in range(3)]
    cmaplist[0] = (1.0, .0, .0, 1.0)
    cmaplist[1] = (1.0, 1.0, 1.0, 1.0)
    cmaplist[2] = (.0, .0, 1.0, 1.0)

    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 3)

    for i in tqdm(range(100)):
        rows = np.random.randint(0, L, batch, dtype=np.uint16)
        cols = np.random.randint(0, L, batch, dtype=np.uint16)
        rs = np.random.random(batch)
        update(lattice, rows, cols, rs, batch, L, D, dt, lambda_)
        im = plt.imshow(lattice, cmap=cmap, interpolation='nearest')
        ims.append([im])

    plt.colorbar()
    filename = get_next_filename('d:\\jason\\2024\\BTP\\', '2_particle_', '.mp4')
    ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=1000)
    ani.save(filename, writer='ffmpeg')

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats('profile_results.prof')
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()