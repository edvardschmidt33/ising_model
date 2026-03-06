import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import json
import os
import argparse

def z(J):
    return np.exp(-2*J)

def kappa(J):
    return 2*np.sinh(2*J)/(np.cosh(2*J)**2)

def kappa_prim(J):
    return 2*(np.tanh(2*J)**2) - 1

def coth(x):
    return np.cosh(x)/np.sinh(x)

def K_1(kappa):
    kappa = float(kappa)
    if not (0<= kappa < 1.0):
        raise ValueError('Kappa must be between 0.0 and 1.0')
    f = lambda phi: 1.0 / ((1.0 - (kappa*kappa)*np.sin(phi)**2)**0.5)
    val, err = quad(f,0,np.pi/2, epsabs=1e-10, epsrel=1e-10, limit=200)
    return val

def E_1(kappa):
    kappa = float(kappa)
    if not (0<= kappa < 1.0):
        raise ValueError('Kappa must be between 0.0 and 1.0')
    f = lambda phi: (1 - (kappa*kappa)*np.sin(phi)**2)**0.5
    val, err = quad(f, 0, np.pi/2, epsabs=1e-10, epsrel=1e-10, limit=200)
    return val


def analytical_E(L, J, kappa, kappa_prim):
    res = -L*L *J *coth(2*J)*(1 + 2/np.pi * kappa_prim * K_1(kappa))
    return res


def analytical_M(J, z):
    Jc = 0.4406868 # 0.5 * np.log(1 + np.sqrt(2))
    if J <= Jc:
        return 0.0
    return (1+z**2)**0.25 * (1 - 6*z**2 + z**4)**0.125/(1-z**2)**0.5


def analytical_CV(L, J, kappa, kappa_prim):
    N = L*L
    return N * 2 / np.pi * (J*coth(2*J))**2 * (2*K_1(kappa) - 2* E_1(kappa) - (1-kappa_prim)*(np.pi/2 + kappa_prim*K_1(kappa)))

def analytical_E_per_spin(J):  # here J is K = 1/T
    k = kappa(J)
    kp = kappa_prim(J)
    return -J * coth(2*J) * (1 + (2/np.pi)*kp*K_1(k))

def analytical_CV_per_spin(J):
    k = kappa(J)
    kp = kappa_prim(J)
    K = K_1(k); E = E_1(k)
    pref = (2/np.pi) * (J*coth(2*J))**2
    return pref * (2*K - 2*E - (1-kp)*(np.pi/2 + kp*K))



def file_ret(L):
    filename = f'./data/results_task1_L{L}.json'
    
    with open(filename, "r") as f:
        data = json.load(f)
    E = np.array(data["E"])
    M = np.array(data["M"])
    CV = np.array(data["CV"])

    T_list = np.array(data["T_list"])
    J_list = np.array(data["J_list"])


    U = np.array(data["U"])
    return E, M, CV, U, J_list, T_list



def to_list(x):
    return x.tolist() if isinstance(x, np.ndarray) else x

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='2D Ising Model Analytical Comparison')
    parser.add_argument("--energy", action="store_true",
                        help="Include energy in plot")
    parser.add_argument("--io", action="store_true",
                        help="Enable saving JSON output")
    args = parser.parse_args()


    

    E_8, M_8, CV_8, U8 ,J_list, T_list = file_ret(8)
    E_16, M_16, CV_16, U16,  J_list, T_list = file_ret(16)
    E_32, M_32, CV_32, U32,J_list, T_list = file_ret(32)
    z_list = [z(J) for J in J_list]
    M_list = zip(z_list, J_list)
    E_analytical =  np.array([analytical_E_per_spin(J) for J in J_list])
    M_analytical = np.array([analytical_M(J, z) for z, J in M_list])
    CV_analytical = np.array([analytical_CV_per_spin(J) for J in J_list])


    results = {
               'E': to_list(E_analytical),
               'M': to_list(M_analytical),
               'CV': to_list(CV_analytical),
               'T_list': to_list(T_list),
               'J_list': to_list(J_list)
               }
    
    if args.io:    
        with open(f"./data/results_task1_analytical.json", "w") as f:
            json.dump(results, f, indent=4)
        print('Results saved in JSON file')


    import matplotlib.pyplot as plt

    ncols = 3 if args.energy else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.2))
    fig.suptitle("Comparison with different L and Analytical solution", fontsize=22)

    # If only 1 subplot, axes is not a list, but here ncols is always 2 or 3
    ax1 = axes[0]
    ax1.plot(T_list, M_analytical, color='Black', linestyle='--', alpha=0.7, label='Analytical')
    ax1.scatter(T_list, M_8, s=30, color='IndianRed', label='L = 8')
    ax1.scatter(T_list, M_16, s=30, color='RoyalBlue', label='L = 16')
    ax1.scatter(T_list, M_32, s=30, color='ForestGreen', label='L = 32')
    ax1.set_xlabel(r"$k_B T / J$", fontsize=16)
    ax1.set_ylabel("|Magnetization|", fontsize=16)
    ax1.legend(fontsize=12)
    ax1.axis('tight')

    ax2 = axes[1]
    ax2.plot(T_list, CV_analytical, color='Black', linestyle='--', alpha=0.7, label='Analytical')
    ax2.scatter(T_list, CV_8, s=30, color='IndianRed', label='L = 8')
    ax2.scatter(T_list, CV_16, s=30, color='RoyalBlue', label='L = 16')
    ax2.scatter(T_list, CV_32, s=30, color='ForestGreen', label='L = 32')
    ax2.set_xlabel(r"$k_B T / J$", fontsize=16)
    ax2.set_ylabel("Specific Heat", fontsize=16)
    ax2.axis('tight')

    if args.energy:
        ax3 = axes[2]
        ax3.plot(T_list, E_analytical, color='Black', linestyle='--', alpha=0.7, label='Analytical')
        ax3.scatter(T_list, E_8, s=30, color='IndianRed', label='L = 8')
        ax3.scatter(T_list, E_16, s=30, color='RoyalBlue', label='L = 16')
        ax3.scatter(T_list, E_32, s=30, color='ForestGreen', label='L = 32')
        ax3.set_xlabel(r"$k_B T / J$", fontsize=16)
        ax3.set_ylabel("Energy", fontsize=16)
        ax3.axis('tight')

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if args.energy:
        plt.savefig('./figs/Task1_analytical_energy')
    else:
        plt.savefig('./figs/Task1_analytical')

    plt.show()
    # plots = 2
    # if args.energy:
    #     plots = 3
    # length = 12.7/ 3

    # f = plt.figure(figsize=(8,length*plots))

    # sp =  f.add_subplot(plots, 1, 1);
    # f.suptitle(f"Comparison with different L and Analytical solution",
    #        fontsize=22)
    # plt.plot(T_list, M_analytical, color='Black', label = 'Analytical', linestyle = '--', alpha = 0.7)
    # plt.scatter(T_list, M_8, s=30, color='IndianRed', label= 'L = 8')
    # plt.scatter(T_list, M_16, s=30, color='RoyalBlue', label= 'L = 16')
    # plt.scatter(T_list, M_32, s=30, color='ForestGreen', label= 'L = 32')
    # plt.xlabel("$k_B T/ J$", fontsize=20);
    # plt.ylabel("|Magnetization| ", fontsize=20);         plt.axis('tight');
    # plt.legend()

    # sp =  f.add_subplot(plots, 1, 2 );
    # plt.plot(T_list, CV_analytical, color='Black', label= 'Analytical', linestyle = '--', alpha = 0.7)
    # plt.scatter(T_list, CV_8, s=30, color='IndianRed', label= 'L = 8')
    # plt.scatter(T_list, CV_16, s=30, color='RoyalBlue', label= 'L = 16')
    # plt.scatter(T_list, CV_32, s=30, color='ForestGreen', label= 'L = 32')
    # plt.xlabel("$k_B T/ J$", fontsize=20);
    # plt.ylabel("Specific Heat ", fontsize=20);         plt.axis('tight');

    # if args.energy:
    #     sp =  f.add_subplot(plots, 1, 3 );
    #     plt.plot(T_list, E_analytical, color='Black', label= 'Analytical', linestyle = '--', alpha = 0.7)
    #     plt.scatter(T_list, E_8, s=50, color='IndianRed', label= 'L = 8')
    #     plt.scatter(T_list, E_16, s=50, color='RoyalBlue', label= 'L = 16')
    #     plt.scatter(T_list, E_32, s=50, color='ForestGreen', label= 'L = 32')
    #     plt.xlabel("$k_B T/ J$", fontsize=20);
    #     plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');
    #     plt.savefig('./figs/Task1_analytical_energy')
    # if not args.energy:
    #     plt.savefig('./figs/Task1_analytical')
    
    # plt.show()