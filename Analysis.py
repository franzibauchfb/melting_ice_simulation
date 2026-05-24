#Import
import numpy as np 
import math

def observable_average(M,n_steps,O):
    OM= O[n_steps-M:n_steps]
    O_average=(1/n_steps)*np.sum(OM)
    return(O_average)

def binning(O, k): #k ist Binanzahl
    n_total = np.size(O)
    min = np.min(O)
    max = np.max(O)
    bins = np.linspace(min,max,k)
    intervall= (max-min)/k
    count= np.zeros(k)
    for i in range(n_total):
        Oi = O[i]
        for j in range(k-1):
            if bins[j]<=Oi<=bins[j+1]:
                count[j]+=1
            if Oi==bins[j+1]:
                count[j]+=1
    histogramm = np.array([bins,count])
    return (histogramm)

def write_xyz(filename, r, step, mode='a'):
    n_particles = r.shape[1]
    with open(filename, mode) as f:
        f.write(f"{n_particles}\n")
        f.write(f"Atoms. Step: {step}\n")
        for i in range(n_particles):
            # Wir nennen die Atome einfach 'O' für Sauerstoff (mW-Wasser)
            f.write(f"O {r[0,i]:.5f} {r[1,i]:.5f} {r[2,i]:.5f}\n")