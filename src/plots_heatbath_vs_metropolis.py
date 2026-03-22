import numpy as np
import json
import matplotlib.pyplot as plt
from analytical_sol import file_ret
import matplotlib.ticker as ticker
import argparse

def file_ret_alg(L, alg):
    if alg == 'heat_bath':
        filename = f'./data/results_heat_bath_L{L}.json'
    elif alg == 'metropolis':
        filename = f'./data/results_task1_L{L}_avg.json'
    elif alg == 'analytical':
        filename = f'./data/results_task1_analytical.json'
    else:
        raise SyntaxError(f'No algorithm named: {alg}')
    with open(filename, "r") as f:
        data = json.load(f)
    E = np.array(data["E"])
    M = np.array(data["M"])
    CV = np.array(data["CV"])
    if alg != 'analytical':
        X = np.array(data["X"])
        U = np.array(data["U"])
    else:
        X = np.zeros(len(E))
        U = np.zeros(len(E))

    T_list = np.array(data["T_list"])
    J_list = np.array(data["J_list"])

    return E, M, CV, X, U, J_list, T_list


if __name__ == '__main__':
    #List of B:s tested: [0.01, 0.05, 0.1, 0.5, 1]
    ### Specify L ###
    parser = argparse.ArgumentParser(description='Plot 2D Ising Model with HB vs Metropolis')
    parser.add_argument("--L", type=int, default=32,
                        help="Lattice size")
    args = parser.parse_args()
    
    L = args.L
    
    E_met, M_met, CV_met, X_met, _, J_list, T_list = file_ret_alg(L, 'metropolis')
    E_hb, M_hb, CV_hb, X_hb, _, _, _ = file_ret_alg(L, 'heat_bath')

    f = plt.figure(figsize=(16,10))

    f.suptitle(f"Comparison of Heat Bath- and Metropolis-algorithm with L = {L}",
           fontsize=24)

    sp =  f.add_subplot(2, 2, 1 );
    plt.scatter(T_list, E_met, s=10, color='IndianRed', label = 'Metropolis')
    plt.scatter(T_list, E_hb, s=10, color='RoyalBlue', label = 'Heat Bath')
    plt.legend(loc="best", framealpha=0.6)
    plt.xlabel("$k_B T/ J$", fontsize=20);
    plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');

    sp =  f.add_subplot(2, 2, 2);
    plt.scatter(T_list, M_met, s=10, color='IndianRed')
    plt.scatter(T_list, M_hb, s=10, color='RoyalBlue')
    plt.xlabel("$k_B T/ J$", fontsize=20);
    plt.ylabel("|Magnetization| ", fontsize=20);         plt.axis('tight');

    sp =  f.add_subplot(2, 2, 3 );
    plt.scatter(T_list, CV_met, s=10, color='IndianRed')
    plt.scatter(T_list, CV_hb, s=10, color='RoyalBlue')
    plt.xlabel("$k_B T/ J$", fontsize=20);
    plt.ylabel("Specific Heat ", fontsize=20);         plt.axis('tight');

    sp =  f.add_subplot(2, 2, 4 );
    plt.scatter(T_list, X_met, s=10, color='IndianRed')
    plt.scatter(T_list, X_hb, s=10, color='RoyalBlue')
    plt.xlabel("$k_B T/ J$", fontsize=20);
    plt.ylabel("Susceptibility ", fontsize=20);         plt.axis('tight');


    plt.savefig(f'./figs/task3_comp{L}')
    plt.show()   