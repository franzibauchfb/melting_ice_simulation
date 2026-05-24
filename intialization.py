#Import
import numpy as np 
import math

def lattice(n_particles, L, dim):
    n_side = int(np.ceil(n_particles **(1/dim)))
    abstand = L / n_side
    achsenpunkte = np.linspace(abstand/2, L-abstand/2, n_side)
    grid = np.array(np.meshgrid(achsenpunkte,achsenpunkte,achsenpunkte))
    r_start = grid.reshape(3,-1)[:, :n_particles]
    return (r_start)


def lh_crystal(nx,ny,nz):
    a=4.521
    b= a*((3)**(0.5))
    c= a*((8/3)**(0.5))
    basis = np.array([
        [0,      0,      0],
        [1/2,    0.5,    0],
        [0,      1/3,    1/16],
        [0.5,    5/6,    1/16],
        [0,      2/3,    1/2],
        [1/2,    1/6,    0.5],
        [0,      1/6,    0.5 + 1/16],
        [0.5,    2/3,    0.5 + 1/16]
    ])
    grundbaustein= basis*np.array([a,b,c])
    gitter_liste= []
    for i in range(nx):
       for j in range(ny):
            for k in range(nz):
                verschiebung = np.array([i*a, j*b, k*c])
                zelle = grundbaustein + verschiebung
                gitter_liste.append(zelle)
    gitter = np.vstack(gitter_liste).T
    L = np.array([nx*a,ny*b,nz*c])
    return(gitter,L)


def lammps_coord(filename,L):
    data = np.loadtxt(filename, skiprows=4068)
    data = data[data[:,0].argsort()]
    coords = L[0]*data[:,2:5].T 
    return coords

