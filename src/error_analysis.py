import json
import matplotlib.pyplot as plt
from plots_heatbath_vs_metropolis import file_ret_alg
import numpy as np
import argparse


def residual(analytical, numerical):
    if len(analytical) == len(numerical):
        return numerical - analytical
    else:
        raise ValueError(f'{numerical} and {analytical} mus be same lenght')
    

def rmse(residuals: np.ndarray) -> float:
    residuals = np.asarray(residuals, dtype=float)
    return float(np.sqrt(np.mean(residuals**2)))

def mae(residuals: np.ndarray) -> float:
    residuals = np.asarray(residuals, dtype=float)
    return float(np.mean(np.abs(residuals)))



if __name__ == '__main__':
    
    ### Specify L ###
    parser = argparse.ArgumentParser(description='Error analysis')
    parser.add_argument("--L", type=int, default=32,
                        help="Lattice size")
    args = parser.parse_args()
    
    L = args.L
    
    E_met, M_met, CV_met, X_met, _, J_list, T_list = file_ret_alg(L, 'metropolis')
    E_hb, M_hb, CV_hb, X_hb, _, _, _ = file_ret_alg(L, 'heat_bath')
    E_an, M_an, CV_an, X_an, _, _, _ = file_ret_alg(L, 'analytical')


    res_E_met = residual(E_an, E_met)
    res_E_hb = residual(E_an, E_hb)

    res_M_met = residual(M_an, M_met)
    res_M_hb = residual(M_an, M_hb)

    res_CV_met = residual(CV_an, CV_met)
    res_CV_hb = residual(CV_an, CV_hb)



    rmse_E_met, rmse_E_hb = rmse(res_E_met), rmse(res_E_hb)
    rmse_M_met, rmse_M_hb = rmse(res_M_met), rmse(res_M_hb)
    rmse_CV_met, rmse_CV_hb = rmse(res_CV_met), rmse(res_CV_hb)

    print(f"L={L} RMSE (Energy):       Metropolis={rmse_E_met:.4e} | Heat bath={rmse_E_hb:.4e}")
    print(f"L={L} RMSE (Magnetization):Metropolis={rmse_M_met:.4e} | Heat bath={rmse_M_hb:.4e}")
    print(f"L={L} RMSE (Specific heat):Metropolis={rmse_CV_met:.4e} | Heat bath={rmse_CV_hb:.4e}")

    f = plt.figure(figsize=(8,12))

    f.suptitle(f"Residuals of Heat Bath- and Metropolis-algorithm with L = {L}",
           fontsize=18)

    sp = f.add_subplot(3, 1, 1)
    plt.scatter(T_list, res_E_met, s=10, color='IndianRed', label='Metropolis')
    plt.scatter(T_list, res_E_hb, s=10, color='RoyalBlue', label='Heat Bath')
    plt.axhline(0, color='black', linewidth=1, alpha=0.5)
    plt.title(f"RMSE: Met={rmse_E_met:.2e}, HB={rmse_E_hb:.2e}", fontsize=12)
    plt.legend(loc="best", framealpha=0.6)
    plt.ylabel("Energy residual", fontsize=20)
    plt.axis('tight')

    sp =  f.add_subplot(3, 1, 2);
    plt.scatter(T_list, res_M_met, s=10, color='IndianRed')
    plt.scatter(T_list, res_M_hb, s=10, color='RoyalBlue')
    plt.axhline(0, color='black', linewidth=1, alpha=0.5)
    plt.title(f"RMSE: Met={rmse_M_met:.2e}, HB={rmse_M_hb:.2e}", fontsize=12)
    plt.ylabel("|Magnetization| ", fontsize=20);         plt.axis('tight');

    sp =  f.add_subplot(3, 1, 3 );
    plt.scatter(T_list, res_CV_met, s=10, color='IndianRed')
    plt.scatter(T_list, res_CV_hb, s=10, color='RoyalBlue')
    plt.axhline(0, color='black', linewidth=1, alpha=0.5)
    plt.title(f"RMSE: Met={rmse_CV_met:.2e}, HB={rmse_CV_hb:.2e}", fontsize=12)
    plt.xlabel("$k_B T/ J$", fontsize=20);
    plt.ylabel("Specific Heat ", fontsize=20);         plt.axis('tight');


    plt.savefig(f'./figs/task3_residuals_L{L}')
    plt.show()   





