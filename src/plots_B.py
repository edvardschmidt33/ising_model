import matplotlib.pyplot as plt
import numpy as np
import json



def file_ret_B(L, B):
    B_file = int(B*100)
    if B == 0:
        filename = f'./data/results_task1_L{L}.json'
    else:
        filename = f'./data/results_task2_L{L}_B{B_file}.json'
    
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
    E, M, CV, X, U, J_list, T_list = file_ret_B(L, 0)
    E001, M001, CV001, X001, U001, _, _ = file_ret_B(L, 0.01)
    E005, M005,CV005, X005,U005,_,_ = file_ret_B(L, 0.05)
    E01, M01, CV01, X01,U01,_,_ = file_ret_B(L, 0.1)
    E05, M05, CV05, X05,U05,_,_ = file_ret_B(L, 0.5)
    E1, M1, CV1, X1, U1,_,_ = file_ret_B(L, 1)
    
    f = plt.figure(figsize=(16,10))
    f.suptitle(f"Comparison of different B with L = {L} ",
                fontsize=24)

    sp =  f.add_subplot(2, 2, 1 );
    plt.scatter(T_list, E, s=10, color='IndianRed', label = 'B = 0')
    plt.scatter(T_list, E001, s=10, color='RoyalBlue', label = 'B = 0.01')
    plt.scatter(T_list, E005, s=10, color='ForestGreen', label = 'B = 0.05')
    plt.scatter(T_list, E01, s=10, color='Coral', label = 'B = 0.1')
    plt.scatter(T_list, E05, s=10, color='DarkMagenta', label = 'B = 0.5')
    plt.scatter(T_list, E1, s=10, color='Khaki', label = 'B = 1.0')
    plt.legend(loc="best", framealpha=0.6)
    plt.xlabel("$k_B T/ J$", fontsize=20);
    plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');

    sp =  f.add_subplot(2, 2, 2);
    plt.scatter(T_list, M, s=10, color='IndianRed')
    plt.scatter(T_list, M001, s=10, color='RoyalBlue')
    plt.scatter(T_list, M005, s=10, color='ForestGreen')
    plt.scatter(T_list, M01, s=10, color='Coral')
    plt.scatter(T_list, M05, s=10, color='DarkMagenta')
    plt.scatter(T_list, M1, s=10, color='Khaki')
    plt.xlabel("$k_B T/ J$", fontsize=20);
    plt.ylabel("|Magnetization| ", fontsize=20);         plt.axis('tight');

    sp =  f.add_subplot(2, 2, 3 );
    plt.scatter(T_list, CV, s=10, color='IndianRed')
    plt.scatter(T_list, CV001, s=10, color='RoyalBlue')
    plt.scatter(T_list, CV005, s=10, color='ForestGreen')
    plt.scatter(T_list, CV01, s=10, color='Coral')
    plt.scatter(T_list, CV05, s=10, color='DarkMagenta')
    plt.scatter(T_list, CV1, s=10, color='Khaki')
    plt.xlabel("$k_B T/ J$", fontsize=20);
    plt.ylabel("Specific Heat ", fontsize=20);         plt.axis('tight');

    sp =  f.add_subplot(2, 2, 4 );
    plt.scatter(T_list, X, s=10, color='IndianRed')
    plt.scatter(T_list, X001, s=10, color='RoyalBlue')
    plt.scatter(T_list, X005, s=10, color='ForestGreen')
    plt.scatter(T_list, X01, s=10, color='Coral')
    plt.scatter(T_list, X05, s=10, color='DarkMagenta')
    plt.scatter(T_list, X1, s=10, color='Khaki')
    plt.xlabel("$k_B T/ J$", fontsize=20);
    plt.ylabel("Susceptibility ", fontsize=20);         plt.axis('tight');

    plt.savefig(f'./figs/task2_L{L}')
    plt.show()