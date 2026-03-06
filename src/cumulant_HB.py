import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def file_ret(L):
    filename = f'./data/results_heat_bath_L{L}.json'
    
    with open(filename, "r") as f:
        data = json.load(f)
    E = np.array(data["E"])
    M = np.array(data["M"])
    CV = np.array(data["CV"])

    T_list = np.array(data["T_list"])
    J_list = np.array(data["J_list"])


    U = np.array(data["U"])
    return E, M, CV, U, J_list, T_list





if __name__ == '__main__':
    _, _, _, U_8, J_list, T_list = file_ret(8)
    _,_,_,U_16,_,_ = file_ret(16)
    _,_,_,U_32,_,_ = file_ret(32)


    fig, ax = plt.subplots()

    ax.plot(T_list, U_8,  color='RoyalBlue',  label='L = 8')
    ax.plot(T_list, U_16, color='IndianRed',  label='L = 16')
    ax.plot(T_list, U_32, color='ForestGreen', label='L = 32')
    

    special = 2.26
    # Vertical line
    ax.axvline(special, color='red', linestyle='--', label = 'intersection')

    ticks = list(ax.get_xticks())
    ticks = sorted(set(ticks + [special]))
    ax.set_xticks(ticks)

    def custom_formatter(x, pos):
        if abs(x - special) < 1e-6:
            return f"{special:.3f}"          
        return f"{x:.0f}"            

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

    ax.set_xlabel('$k_B T/ J$')
    ax.set_ylabel('Cumulant (4th order)')
    ax.set_title('Cumulant versus temperature for L = 8, 16, 32 (HB algorithm)')
    ax.set_ylim(-0.2, 0.8)
    ax.set_xlim(0.5, 5)

    ax.legend()

    fig.savefig('./figs/HB_cumulant.png', dpi=300)
    plt.show()
        