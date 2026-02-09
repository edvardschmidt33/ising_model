import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import json
import os


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
    Jc = 0.44068 # 0.5 * np.log(1 + np.sqrt(2))
    if J <= Jc:
        return 0.0
    return (1+z**2)**0.25 * (1 - 6*z**2 + z**4)**0.125/(1-z**2)**0.5


def analytical_CV(L, J, kappa, kappa_prim):
    N = L*L
    return N * 2 / np.pi * (J*coth(2*J))**2 * (2*K_1(kappa) - 2* E_1(kappa) - (1-kappa_prim)*(np.pi/2 + kappa_prim*K_1(kappa)))

def analytical_E_per_spin(J):  # here J is K = 1/T
    k = kappa(J)
    kp = kappa_prim(J)
    return -coth(2*J) * (1 + (2/np.pi)*kp*K_1(k))

def analytical_CV_per_spin(J):
    k = kappa(J)
    kp = kappa_prim(J)
    K = K_1(k); E = E_1(k)
    pref = (2/np.pi) * (J*coth(2*J))**2
    return pref * (2*K - 2*E - (1-kp)*(np.pi/2 + kp*K))



def file_ret(L):
    filename = f'data/results_task1_L{L}.json'
    
    with open(filename, "r") as f:
        data = json.load(f)
    E = np.array(data["E"])
    M = np.array(data["M"])
    CV = np.array(data["CV"])

    T_list = np.array(data["T_list"])
    J_list = np.array(data["J_list"])

    return E, M, CV, J_list, T_list
def to_list(x):
    return x.tolist() if isinstance(x, np.ndarray) else x

if __name__ == '__main__':
    

    E_8, M_8, CV_8, J_list, T_list = file_ret(8)
    E_16, M_16, CV_16, J_list, T_list = file_ret(16)
    E_32, M_32, CV_32, J_list, T_list = file_ret(32)
    z_list = [z(J) for J in J_list]
    M_list = zip(z_list, J_list)
    E_analytical = 0.5* np.array([analytical_E_per_spin(J) for J in J_list])
    M_analytical = np.array([analytical_M(J, z) for z, J in M_list])
    CV_analytical = 0.5* np.array([analytical_CV_per_spin(J) for J in J_list])


    results = {
               'E': to_list(E_analytical),
               'M': to_list(M_analytical),
               'CV': to_list(CV_analytical),
               'T_list': to_list(T_list),
               'J_list': to_list(J_list)
               }
    
    with open(f"data/results_task1_analytical.json", "w") as f:
        json.dump(results, f, indent=4)



    f = plt.figure(figsize=(7,7))



    sp =  f.add_subplot(2, 1, 1);
    plt.plot(T_list, M_analytical, color='IndianRed', label = 'Analytical')
    plt.scatter(T_list, M_8, s=50, color='RoyalBlue', label= 'L = 8')
    plt.scatter(T_list, M_16, s=50, color='ForestGreen', label= 'L = 16')
    plt.scatter(T_list, M_32, s=50, color='Orange', label= 'L = 32')
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("|Magnetization| ", fontsize=20);         plt.axis('tight');
    plt.legend()

    sp =  f.add_subplot(2, 1, 2 );
    plt.plot(T_list, CV_analytical, color='IndianRed', label= 'Analytical')
    plt.scatter(T_list, CV_8, s=50, color='RoyalBlue', label= 'L = 8')
    plt.scatter(T_list, CV_16, s=50, color='ForestGreen', label= 'L = 16')
    plt.scatter(T_list, CV_32, s=50, color='Orange', label= 'L = 32')
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Specific Heat ", fontsize=20);         plt.axis('tight');
    plt.show()

