#Import
import numpy as np 
import math

from mw_force_2 import force_mW, minimum_image_distance
from temperature_measurement import instantaneous_temp
from time_integration import rescale_factor, velocity_verlet_integration_with_rescale
from Analysis import observable_average, binning, write_xyz
from intialization import lattice, lh_crystal, lammps_coord
from graphics import eng_plot,temp_plot, temp_histogramm

#Paramters
m = 18.015
delta_t=0.1
n_steps = 10000  
bath_temp = 25
tau = 500.0  
n_particles = 32
filename = "dump_py"
open(filename, 'w').close()


#Initalisation
lattic = lh_crystal (1,2,2)
r_start = lattic[0]
L = lattic[1]+np.array([4.521*0.5,4.521*0.5*(3)**0.5,4.521*0.5*(8/3)**0.5]) #PBCs brauchen etwas mehr Platz als die Gitterkonstante
center = np.mean(r_start, axis=1).reshape(3, 1)
L_vec = L.reshape(3, 1) 
r_start = r_start - center + (L_vec / 2)

# Richtiger Anfangsimpuls basierend auf Boltzmann-Verteilung
# Für Zieltemperatur: KE = (3/2)*N*k_B*T
k_B = 8.617333262e-5  # eV/K
KE_target = (3/2) * n_particles * k_B * bath_temp
ENERGY_CONVERSION = 103.6427
p_rms = np.sqrt(2*m * KE_target / (n_particles * ENERGY_CONVERSION))
p_start = np.random.randn(3, n_particles) * (p_rms / np.sqrt(3))

# Remove center-of-mass momentum from the initial velocity distribution
p_start -= np.mean(p_start, axis=1, keepdims=True)


#time_integration
res = velocity_verlet_integration_with_rescale(r_start,p_start,delta_t,n_steps,L,m,bath_temp,tau, filename)
trajectory = res[0]
impulses = res[1]
pot_eng = res[2]
kin_eng = res[3]
tot_eng = kin_eng + pot_eng
inst_temp = res[4]

#Analysis
temp_average = observable_average(n_steps,n_steps,inst_temp)
temp_distribution= binning(inst_temp[0:n_steps],25)

#Graphics
eng_plot(pot_eng,kin_eng,tot_eng)
temp_plot(inst_temp,temp_average)
temp_histogramm(temp_distribution)

