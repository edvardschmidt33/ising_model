
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from numba import njit

import json
import argparse

np.random.seed(0)

import os
os.makedirs("./data", exist_ok=True)
os.makedirs("./figs", exist_ok=True)


def init_lattice(L):
    return np.random.choice([-1,1], size = (L,L))

@njit(cache = True)
def dE(s, i, j, L, J):
    t = s[i - 1 if i>0 else L-1, j]
    b = s[i + 1 if i<L-1 else 0, j]
    l = s[i, j - 1 if j>0 else L-1]
    r = s[i, j + 1 if j<L-1 else 0]

    return J * 2 * (t + b + l + r) * s[i, j]

@njit(cache = True, fastmath = True)
def MonterCarlo_move(s, L, J):
    for _ in range(L*L):       
        i = np.random.randint(0,L)
        j = np.random.randint(0,L)
        ediff = dE(s,i,j,L, J)
        if ediff <= 0:
            s[i,j] = -s[i, j]
        elif np.random.random() < np.exp(-ediff):
            s[i,j] = -s[i, j]
    return s



@njit(cache=True, fastmath=True)
def Energy(s, L, J):
    energy = 0.0
    for i in range(L):
        for j in range(L):
            S = s[i, j]
            energy -= J * S * (s[(i+1) % L, j] + s[i, (j+1) % L])  # right + down only
    return energy



@njit(cache = True)
def Mag(s):
    mag = 0.0
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            mag += s[i, j]
    return mag


def equilibriate(s, L, J, max_sweeps,stable_blocks = 5, block_size = 100, tol = 1e-4):
    N = L*L
    prev_block_avg = None
    stable_count = 0

    count = 0
    while count < max_sweeps:
        E_block = 0
        for _ in range(block_size):
            MonterCarlo_move(s,L,J)
            E_block += Energy(s, L, J) / N
            
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


def main():
    for n, J in tqdm(enumerate(J_list), total=len(J_list), desc="J loop"):   
        s = init_lattice(L)
        E_j = M_j = 0.0
        E2_j = M2_j = 0.0
        M4_j = 0.0
        M_j_raw = 0.0
        equilibriate(s, L, J, eq_limit)
        
        for _ in range(mc_sweeps):
            MonterCarlo_move(s,L,J)
            E_sample = Energy(s, L, J)
            M_sample = Mag(s)
        
            E_j += E_sample
            M_j += abs(M_sample)
            M_j_raw += M_sample
            E2_j += E_sample*E_sample
            M2_j += M_sample*M_sample
            M4_j += M_sample**4
        M4_avg = M4_j/mc_sweeps
        M2_avg = M2_j/mc_sweeps
        T = T_list[n]
        E[n] = E_j/(mc_sweeps*L*L)
        M[n] = M_j/(mc_sweeps*L*L)
        U[n] = 1 - M4_avg / (3 * (M2_avg**2))
        CV[n] = (E2_j - E_j*E_j/mc_sweeps)/(mc_sweeps*L*L) # *T*T) 
        X[n] = (M2_j/mc_sweeps - (M_j_raw/mc_sweeps)**2)/(L*L) #*T) 

def to_list(x):
    return x.tolist() if isinstance(x, np.ndarray) else x

if __name__ == '__main__':

    ### Define algorithm parameters ###
    parser = argparse.ArgumentParser(description='2D Ising Model Monte Carlo Simulation')
    parser.add_argument("--plot", action="store_true",
                        help="Enable plotting")
    parser.add_argument("--io", action="store_true",
                        help="Enable saving JSON output")

    # Optional numeric arguments
    parser.add_argument("--L", type=int, default=32,
                        help="Lattice size")
    parser.add_argument("--sweeps", type=int, default=10000,
                        help="Number of MC sweeps")
    parser.add_argument("--eq", type=int, default=50000,
                        help="Equilibration sweeps")

    args = parser.parse_args()

    temp_points = 100           # number of temperature points
    L = args.L                  # lattice size
    eq_limit = args.eq          # Maximum iterations before equilibrum
    mc_sweeps  = args.sweeps    # Sweeps in Monte Carlo-sampling

    T_list = np.linspace(0.5, 5, temp_points)        #change to real Boltzman constant if necessary
    J_list = 1/(T_list)

    s = init_lattice(L)
    E, M = np.zeros(temp_points), np.zeros(temp_points)
    CV ,X = np.zeros(temp_points), np.zeros(temp_points)
    U = np.zeros(temp_points)
    
    print(f'Metropolis simulation with L = {L} started')
    print(30*'-')
    main()


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
               'eq_limit': int(eq_limit)}
    
    if  args.io:
        with open(f"./data/results_task1_L{L}.json", "w") as f:
            json.dump(results, f, indent=4)
        print("Results saved in .json")

    if args.plot:
        f = plt.figure(figsize=(16,10))

        f.suptitle(f"Metropolis-algorithm with L = {L}",
           fontsize=25)
        sp =  f.add_subplot(2, 2, 1 );
        plt.scatter(T_list, E, s=30, color='IndianRed')
        plt.xlabel("$k_B T/ J$", fontsize=20);
        plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');

        sp =  f.add_subplot(2, 2, 2);
        plt.scatter(T_list, M, s=30, color='RoyalBlue')
        plt.xlabel("$k_B T/ J$", fontsize=20);
        plt.ylabel("|Magnetization| ", fontsize=20);         plt.axis('tight');

        sp =  f.add_subplot(2, 2, 3 );
        plt.scatter(T_list, CV, s=30, color='IndianRed')
        plt.xlabel("$k_B T/ J$", fontsize=20);
        plt.ylabel("Specific Heat ", fontsize=20);         plt.axis('tight');

        sp =  f.add_subplot(2, 2, 4 );
        plt.scatter(T_list, X, s=30, color='RoyalBlue')
        plt.xlabel("$k_B T/ J$", fontsize=20);
        plt.ylabel("Susceptibility ", fontsize=20);         plt.axis('tight');

        plt.savefig(f'./figs/Task1_{L}x{L}')
        plt.show()

    print('Complete')