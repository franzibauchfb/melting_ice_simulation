#Import
import numpy as np 
import math

from mw_force_2 import force_mW
from temperature_measurement import instantaneous_temp
from Analysis import write_xyz

# Conversion factors for metal units (eV, Å, fs, amu)
FORCE_TO_ACCELERATION = 0.009648533289  # Å/fs^2 per (eV/Å) per amu


def rescale_factor(inst_temp,bath_temp,tau, delta_t):
    if inst_temp < 1e-10:
        return 1
    factor = math.sqrt(1+((2*delta_t)/tau)*((bath_temp/inst_temp)-1))
    return(factor)

def velocity_verlet_integration_with_rescale (r,p,delta_t,n_steps,L,m,bath_temp,tau, filename):
    n_particles = len(r[0]) #Teilchenanzahl
    r_var=r #um Variable von Anfangsbedingung zu unterscheiden
    p_var=p
    res_force_old =force_mW(r_var,L)
    F_old=res_force_old[0]
    trajectory=np.zeros((n_steps,3,n_particles))
    impulses=np.zeros((n_steps,3,n_particles))
    pot_eng=np.zeros(n_steps)
    kin_eng=np.zeros(n_steps)
    inst_temp=np.zeros(n_steps)
    trajectory[0]=r_var #Anfangsbedingungen festhalten
    impulses[0]=p_var
    for k in range(n_steps-1):
        r_var = r_var + (delta_t/m)*p_var + (delta_t**2/2)*(F_old * FORCE_TO_ACCELERATION / m)  # neue Position (Velocity Verlet)
        r_var %= L.reshape(3,1)
        res_force_new = force_mW(r_var,L) 
        F_new= res_force_new[0] #neue Kraft
        pot_eng_k= res_force_new[1] 
        pot_eng[k]=pot_eng_k #Potentielle Energie speichern

        p_var = p_var + (delta_t/2)*(F_old + F_new) * FORCE_TO_ACCELERATION #neuer Impuls
        F_old=F_new

        trajectory [k+1]=r_var #speichern des Timesteps Orte
        impulses[k+1]=p_var
        if k % 50 == 0:
            write_xyz(filename,r_var,k,mode='a')
        res_inst_temp=instantaneous_temp(p_var,n_particles,m)
        inst_temp_k = res_inst_temp[0]
        inst_temp[k]=inst_temp_k
        kin_eng_k = res_inst_temp[1]
        kin_eng[k] = kin_eng_k #Kinetische Energie speichern
        rescale= rescale_factor(inst_temp_k,bath_temp, tau,delta_t) #Thermostat
        p_var=rescale*p_var
        
    return (trajectory,impulses,pot_eng,kin_eng,inst_temp)
