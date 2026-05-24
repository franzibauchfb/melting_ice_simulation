import numpy as np 
import math

epsilon = 0.2683
sigma = 2.3925
a= 1.8
lamba = 23.15
gamma = 1.2
A = 7.049556277
B = 0.6022245584
p=4
q=0
costheta= -1/3

def minimum_image_distance (ri,rj,L):
    rij = rj - ri
    rij = rij - L * np.round(rij / L)
    return rij

def triplet_potential(ri,rj,rk,L):
    rij_vec = minimum_image_distance(ri,rj,L)
    rik_vec = minimum_image_distance(ri,rk,L)
    rij = np.linalg.norm(rij_vec)
    rik = np.linalg.norm(rik_vec)
    if rij >= a*sigma or rik >=a*sigma:
        phi_3=0
        Fj=0
        Fk=0
    else:
        costhetaijk=(np.dot(rij_vec,rik_vec)/(rij*rik))
        phi_3=lamba*epsilon*(costhetaijk-costheta)**2*math.exp(gamma*sigma/(rij-a*sigma))*math.exp(gamma*sigma/(rik-a*sigma))
        fstrichdurchfij= -gamma*sigma/(rij-a*sigma)**2
        fstrichdurchfik= -gamma*sigma/(rik-a*sigma)**2
        phi_partial_cos= 2*lamba*epsilon*(costhetaijk-costheta)*math.exp(sigma*gamma/(rij-a*sigma))*math.exp(gamma*sigma/(rik-a*sigma))
        Fj= -(-phi_3*(fstrichdurchfij/rij)*rij_vec - phi_partial_cos/(rij*rik)*rik_vec + phi_partial_cos*costhetaijk/rij**2*rij_vec)
        Fk= -(-phi_3*(fstrichdurchfik/rik)*rik_vec - phi_partial_cos/(rij*rik)*rij_vec + phi_partial_cos*costhetaijk/rik**2*rik_vec)
    return(phi_3,Fj,Fk)

def force_mW(r,L):
    n= r.shape[1]
    forces= np.zeros((3,n))
    pot_eng=0
    #pair
    for i in range(n):
        for j in range(i + 1, n):
            rij_vec = minimum_image_distance(r[:, i], r[:, j], L)
            rij = np.linalg.norm(rij_vec)
            if rij < a*sigma:
            # 1. Potentielle Energie
            # Formel: A * eps * [B*(sig/r)^4 - 1] * exp(sig/(r-a*sig))
                term_pow = (sigma/rij)**4
                exp_factor = math.exp(sigma/(rij - a*sigma))
    
                phi_2 = A * epsilon * (B * term_pow - 1.0) * exp_factor
                pot_eng += phi_2

            # 2. Kraft Fij (Ableitung von phi_2)
            # Hilfsvariablen
                exp_part = math.exp(sigma / (rij - a * sigma))
                bracket = B * (sigma / rij)**4 - 1.0  # Nutze q=0 -> (sigma/rij)**0 = 1

            # Die korrekte Ableitung (Kettenregel)
            # dV/dr = A * eps * [ exp * (-4*B*sigma^4/r^5) + bracket * exp * (-sigma/(r-a*sigma)^2) ]
                deriv = A * epsilon * exp_part * (
                        -4.0 * B * (sigma**4) / (rij**5) 
                        - (B * (sigma/rij)**4 - 1.0) * (sigma / (rij - a*sigma)**2))
            # Kraft ist der negative Gradient: F = -dV/dr * (r_vec/r)
                F_scalar = -deriv 
                Fij = F_scalar * (rij_vec / rij)

                forces[:, i] -= Fij
                forces[:, j] += Fij
    #triplet
    for i in range(n):
        neighbors=[]
        for j in range(n):
            if i==j: continue
            rij_vec = minimum_image_distance(r[:,i],r[:,j],L) 
            rij = np.linalg.norm(rij_vec)
            if rij < a*sigma:
                neighbors.append(j)
        for idx_j in range(len(neighbors)):
            for idx_k in range(idx_j + 1, len(neighbors)):
                j = neighbors[idx_j]
                k = neighbors[idx_k]
                rik_vec = minimum_image_distance(r[:,i],r[:,k],L) 
                rik = np.linalg.norm(rik_vec)
                if rik < a*sigma:
                    fijk = triplet_potential(r[:, i], r[:, j], r[:,k], L)
                    forces[:,i] += fijk[1]+fijk[2]
                    forces[:,j] -= fijk[1]
                    forces[:,k] -= fijk[2] 
                    pot_eng += fijk[0]
    return(forces,pot_eng)
