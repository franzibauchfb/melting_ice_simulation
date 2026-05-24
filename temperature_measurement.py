#Import
import numpy as np 
import math

def instantaneous_temp(pt,n_particles,m):
    impulses_squared=(np.linalg.norm(pt,axis=0))**2
    # Convert kinetic energy from amu*(Å/fs)^2 to eV
    ENERGY_CONVERSION = 103.6427
    kin_eng = np.sum(impulses_squared)/(2*m) * ENERGY_CONVERSION
    inst_temp = 2/(3*(n_particles-1)*8.617333262e-5)*kin_eng
    return(inst_temp,kin_eng)