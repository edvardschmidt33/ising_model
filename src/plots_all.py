import matplotlib.pyplot as plt
import numpy as np
import json



def file_ret(L):
    filename = f'./data/results_task1_L{L}.json'
    
    with open(filename, "r") as f:
        data = json.load(f)
    E = np.array(data["E"])
    M = np.array(data["M"])
    CV = np.array(data["CV"])
    X = np.array(data["X"])

    T_list = np.array(data["T_list"])
    J_list = np.array(data["J_list"])


    U = np.array(data["U"])
    return E, M, CV, U, X, J_list, T_list


if __name__ == '__main__':
    #List of B:s tested: [0.01, 0.05, 0.1, 0.5, 1]
    ### Specify L ###
    E_8, M_8, CV_8, U_8, X_8 ,J_list, T_list = file_ret(8)
    E_16, M_16, CV_16, U_16, X_16,  J_list, T_list = file_ret(16)
    E_32, M_32, CV_32, U32, X_32, J_list, T_list = file_ret(32)
    
    f = plt.figure(figsize=(16,10))
    f.suptitle(f"Metropolis Monte Carlo Simulation B = 0",
                fontsize=24)

    sp =  f.add_subplot(2, 2, 1 );
    plt.scatter(T_list, E_8, s=10, color='IndianRed', label = 'L = 8')
    plt.scatter(T_list, E_16, s=10, color='RoyalBlue', label = 'L = 16')
    plt.scatter(T_list, E_32, s=10, color='ForestGreen', label = 'L = 32')
    plt.legend(loc="best", framealpha=0.6)
    plt.xlabel("$k_B T/ J$", fontsize=20);
    plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');

    sp =  f.add_subplot(2, 2, 2);
    plt.scatter(T_list, M_8, s=10, color='IndianRed')
    plt.scatter(T_list, M_16, s=10, color='RoyalBlue')
    plt.scatter(T_list, M_32, s=10, color='ForestGreen')
    plt.xlabel("$k_B T/ J$", fontsize=20);
    plt.ylabel("|Magnetization| ", fontsize=20);         plt.axis('tight');

    sp =  f.add_subplot(2, 2, 3 );
    plt.scatter(T_list, CV_8, s=10, color='IndianRed')
    plt.scatter(T_list, CV_16, s=10, color='RoyalBlue')
    plt.scatter(T_list, CV_32, s=10, color='ForestGreen')
    plt.xlabel("$k_B T/ J$", fontsize=20);
    plt.ylabel("Specific Heat ", fontsize=20);         plt.axis('tight');

    sp =  f.add_subplot(2, 2, 4 );
    plt.scatter(T_list, X_8, s=10, color='IndianRed')
    plt.scatter(T_list, X_16, s=10, color='RoyalBlue')
    plt.scatter(T_list, X_32, s=10, color='ForestGreen')
    plt.xlabel("$k_B T/ J$", fontsize=20);
    plt.ylabel("Susceptibility ", fontsize=20);         plt.axis('tight');

    plt.savefig(f'./figs/task1_collective')
    plt.show()