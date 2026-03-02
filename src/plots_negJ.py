import matplotlib.pyplot as plt
import numpy as np
import json



def file_ret_J(L, J_sign):
    if J_sign == '-':
        filename = f'./data/results_task2_L{L}_negJ.json'
    else:
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
    return E, M, CV, X, U, J_list, T_list


if __name__ == '__main__':
    #List of B:s tested: [0.01, 0.05, 0.1, 0.5, 1]
    ### Specify L ###
    L = 32
    E_neg, M_neg, CV_neg, X_neg, _, J_list, T_list = file_ret_J(L, '-')
    E_pos, M_pos, CV_pos, X_pos, _, _, _ = file_ret_J(L, '+')

    f = plt.figure(figsize=(16,10))

    f.suptitle(f"Comparison of positive and negative J with L = {L}",
           fontsize=24)

    sp =  f.add_subplot(2, 2, 1 );
    plt.scatter(T_list, E_pos, s=10, color='IndianRed', label = 'Positive')
    plt.scatter(T_list, E_neg, s=10, color='RoyalBlue', label = 'Negative')
    plt.legend(loc="best", framealpha=0.6)
    plt.xlabel("$k_B T/ J$", fontsize=20);
    plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');

    sp =  f.add_subplot(2, 2, 2);
    plt.scatter(T_list, M_pos, s=10, color='IndianRed')
    plt.scatter(T_list, M_neg, s=10, color='RoyalBlue')
    plt.xlabel("$k_B T/ J$", fontsize=20);
    plt.ylabel("|Magnetization| ", fontsize=20);         plt.axis('tight');

    sp =  f.add_subplot(2, 2, 3 );
    plt.scatter(T_list, CV_pos, s=10, color='IndianRed')
    plt.scatter(T_list, CV_neg, s=10, color='RoyalBlue')
    plt.xlabel("$k_B T/ J$", fontsize=20);
    plt.ylabel("Specific Heat ", fontsize=20);         plt.axis('tight');

    sp =  f.add_subplot(2, 2, 4 );
    plt.scatter(T_list, X_pos, s=10, color='IndianRed')
    plt.scatter(T_list, X_neg, s=10, color='RoyalBlue')
    plt.xlabel("$k_B T/ J$", fontsize=20);
    plt.ylabel("Susceptibility ", fontsize=20);         plt.axis('tight');


    plt.savefig(f'./figs/task2_{L}_negJ')
    plt.show()   