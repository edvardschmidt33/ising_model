import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm.auto import tqdm
from numba import njit
import os
os.makedirs("./data", exist_ok=True)
os.makedirs("./figs", exist_ok=True)
from sim import main, equilibriate, MonterCarlo_move, Mag, init_lattice, to_list


@njit(cache = True)
def dE(s, i, j, L, J, B):
    t = s[i - 1 if i>0 else L-1, j]
    b = s[i + 1 if i<L-1 else 0, j]
    l = s[i, j - 1 if j>0 else L-1]
    r = s[i, j + 1 if j<L-1 else 0]

    return  2 * s[i, j] * (J*(t + b + l + r) + B)



@njit(cache = True, fastmath = True)
def Energy(s, L, J, B):
    energy = 0
    for i in range(L):
        for j in range(L):
            S = s[i, j]
            nn = s[(i+1)%L, j] + s[i,(j+1)%L] + s[(i-1)%L, j] + s[i,(j-1)%L]
            energy += nn*S   
    
    energy = energy/4
    energy += -B*np.sum(s)
    return energy #add/remove J

@njit(cache = True, fastmath = True)
def MonterCarlo_move(s, L, J, B):
    for _ in range(L*L):       
        i = np.random.randint(0,L)
        j = np.random.randint(0,L)
        ediff = dE(s,i,j,L, J, B)
        if ediff <= 0:
            s[i,j] = -s[i, j]
        elif np.random.random() < np.exp(-ediff):
            s[i,j] = -s[i, j]
    return s


def equilibriate(s, L, J, B ,max_sweeps,stable_blocks = 5, block_size = 100, tol = 1e-4):
    N = L*L
    prev_block_avg = None
    stable_count = 0

    count = 0
    while count < max_sweeps:
        E_block = 0
        for _ in range(block_size):
            MonterCarlo_move(s,L, J, B)
            E_block += Energy(s, L, J, B) / N
            
        block_avg = E_block/block_size

        if prev_block_avg is not None:
            if abs(block_avg - prev_block_avg) < tol:
                stable_count += 1
                if stable_count >= stable_blocks:
                    print(f'Achieved stable count at {count}')
                    break
            else:
                stable_count = 0
        
        prev_block_avg = block_avg
        count += block_size
    
    return count


def main(B):
    for n, J in tqdm(enumerate(J_list), total=len(J_list), desc="J loop"):   
        s = init_lattice(L)
        E_j = M_j = 0
        E2_j = M2_j = 0
        M4_j = 0
        equilibriate(s, L, -J, B, eq_limit)
        
        for _ in range(mc_sweeps):
            MonterCarlo_move(s, L, -J, B)
            E_sample = Energy(s, L, -J, B)
            M_sample = Mag(s)
        
            E_j += E_sample
            M_j += abs(M_sample)
            E2_j += E_sample*E_sample
            M2_j += M_sample*M_sample
            M4_j += M_sample**4
        T = T_list[n]
        E[n] = E_j/(mc_sweeps*L*L)
        M[n] = M_j/(mc_sweeps*L*L)
        CV[n] = (E2_j - E_j*E_j/mc_sweeps)/(mc_sweeps*L*L*T*T) 
        X[n] = (M2_j- M_j*M_j/mc_sweeps)/(mc_sweeps*L*L*T) 



if __name__ == '__main__':
       ### Define algorithm parameters ###

    temp_points = 100      # Number of temperature points
    L = 32                 # Lattice size
    eq_limit = 50000       # Maximum iterations before equilibrum
    mc_sweeps  = 10000     # Sweeps in Monte Carlo-sampling
    B = 0                  # Magnetic field
    
    T_list = np.linspace(0.5, 5, temp_points)        #change to real Boltzman constant if necessary
    J_list = 1/(T_list)


    s = init_lattice(L)
    E, M = np.zeros(temp_points), np.zeros(temp_points)
    CV ,X = np.zeros(temp_points), np.zeros(temp_points)
    U = np.zeros(temp_points)
    
    main(B)

    results = {
            'E': to_list(E),
            'M': to_list(M),
            'CV': to_list(CV),
            'X': to_list(X),
            'U': to_list(U),
            'T_list': to_list(T_list),
            'J_list': to_list(J_list),
            'L': int(L),
            'temp_points': int(temp_points),
            'mc_sweeps': int(mc_sweeps),
            'eq_limit': int(eq_limit),
            }
    
    with open(f"./data/results_task2_L{L}_negJ.json", "w") as f:
        json.dump(results, f, indent=4)

    print('Results saved in .json')
