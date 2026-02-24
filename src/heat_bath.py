import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm.auto import tqdm
from numba import njit
import argparse
from sim import init_lattice, Energy, Mag, equilibriate, to_list




@njit(cache = True)
def p(s,i, j, L, J):
    ### Here beta is ignored to follow the scheme of the original logic where beta is baked in to J ###
    t = s[i - 1 if i>0 else L-1, j]
    b = s[i + 1 if i<L-1 else 0, j]
    l = s[i, j - 1 if j>0 else L-1]
    r = s[i, j + 1 if j<L-1 else 0]
    nn = t + b + l + r
    num = np.exp(2*J*nn)
    den = 1 + num
    return num/den

@njit(cache = True, fastmath = True)
def MonterCarlo_move(s, L, J):
    for _ in range(L*L):       
        i = np.random.randint(0,L)
        j = np.random.randint(0,L)
        p_plus = p(s,i,j,L, J)
        if np.random.random() < p_plus:
            s[i,j] = 1
        else:
            s[i,j] = -1
    return s


def main():
    for n, J in tqdm(enumerate(J_list), total=len(J_list), desc="J loop"):   
        s = init_lattice(L)
        E_j = M_j = 0
        E2_j = M2_j = 0
        M4_j = 0
        equilibriate(s, L, J, eq_limit)
        
        for _ in range(mc_sweeps):
            MonterCarlo_move(s,L,J)
            E_sample = Energy(s, L, J)
            M_sample = Mag(s)
        
            E_j += E_sample
            M_j += abs(M_sample)
            E2_j += E_sample*E_sample
            M2_j += M_sample*M_sample
            M4_j += M_sample**4
        M4_avg = M4_j/mc_sweeps
        M2_avg = M2_j/mc_sweeps
        T = T_list[n]
        E[n] = E_j/(mc_sweeps*L*L)
        M[n] = M_j/(mc_sweeps*L*L)
        U[n] = 1 - M4_avg / (3 * (M2_avg**2))
        CV[n] = (E2_j - E_j*E_j/mc_sweeps)/(mc_sweeps*L*L*T*T) 
        X[n] = (M2_j- M_j*M_j/mc_sweeps)/(mc_sweeps*L*L*T) 


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
    
    print(f'Heat Bath simulation with L = {L} started')
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
    if args.io:
        with open(f"./data/results_heat_bath_L{L}.json", "w") as f:
            json.dump(results, f, indent=4)

        print('Results saved in .json')


    if args.plot:
        f = plt.figure(figsize=(16,10))


        sp =  f.add_subplot(2, 2, 1 );
        plt.scatter(T_list, E, s=50, color='Red')
        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');

        sp =  f.add_subplot(2, 2, 2);
        plt.scatter(T_list, M, s=50, color='Blue')
        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("|Magnetization| ", fontsize=20);         plt.axis('tight');

        sp =  f.add_subplot(2, 2, 3 );
        plt.scatter(T_list, CV, s=50, color='Red')
        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Specific Heat ", fontsize=20);         plt.axis('tight');

        sp =  f.add_subplot(2, 2, 4 );
        plt.scatter(T_list, X, s=50, color='Blue')
        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Susceptibility ", fontsize=20);         plt.axis('tight');

        plt.savefig(f'./figs/Task1_{L}x{L}')
        plt.show()

    print('Complete')