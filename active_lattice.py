import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import time as tt
from tqdm import tqdm
from numba import jit

plt.rcParams['animation.ffmpeg_path'] = r'C:\ProgramData\chocolatey\bin\ffmpeg.exe'  # Update this path if necessary

L = 400
D = 1
lambda_ = 40/100
gamma = 10/10000
rho_0 = 0.65

N = int(rho_0*L*L)

lattice = np.array([[0 for col in range(L)] for row in range(L)])

dt = 1/(2*D+lambda_+gamma)
# time = 0

def fill_lattice():
    # time = 0
    lattice = np.array([[0 for col in range(L)] for row in range(L)])
    
    filled_spaces = np.random.choice(L*L, N, replace = False)
    for index in range(N):
        col = filled_spaces[index] % L
        row = filled_spaces[index] // L
        spin = np.random.choice([-1,1])
        lattice[row][col] = spin

    return lattice

@jit(nopython=True)
def update(lattice):
    #Choosing particle

    col = np.random.randint(0,L)
    row = np.random.randint(0,L)


    r = np.random.random()
    if r < D*dt:
        #Diffuse upwards

        lattice[(row-1)%L][col], lattice[row][col] = lattice[row][col], lattice[(row-1)%L][col]
    
    elif r < 2*D*dt:
        #Diffuse left
        
        lattice[row][(col+1)%L], lattice[row][col] = lattice[row][col], lattice[row][(col+1)%L]

    
    elif r < (2*D + lambda_)*dt:

        if lattice[row][col] == 1 and lattice[row][(col+1)%L] == 0:
            lattice[row][col], lattice[row][(col+1)%L] = 0,1

        elif lattice[row][col] == -1 and lattice[row][(col-1)%L] == 0:
            lattice[row][col], lattice[row][(col-1)%L] = 0,-1

    else:
        lattice[row][col] = -lattice[row][col]

    # time += dt/N


fig = plt.figure()
ims = []
lattice=fill_lattice()
cmaplist = [0 for i in range(5)]
cmaplist[0] = (1.0, .0, .0, 1.0)
cmaplist[1] = (.0, 1.0, .0, 1.0)
cmaplist[2] = (1.0, 1.0, 1.0, 1.0)
cmaplist[3] = (1.0, 1.0, .0, 1.0)
cmaplist[4] = (.0, .0, 1.0, 1.0)

# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, 5)

for i in tqdm(range(1_000_000_000)):
    update(lattice)
    if i % 1_000_000 == 0:
        im = plt.imshow(lattice, cmap=cmap, interpolation='nearest')

        ims.append([im])

plt.colorbar()
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

ani.save(r'd:\jason\2024\BTP\test1.mp4', writer='ffmpeg')
