import numpy as np
import json
import matplotlib.pyplot as plt
from conv import file_ret

if __name__ == '__main__':
    _, _, _, U_8, J_list, T_list = file_ret(8)
    _,_,_,U_16,_,_ = file_ret(16)
    _,_,_,U_32,_,_ = file_ret(32)

    plt.scatter(T_list, U_8, color = 'RoyalBlue',label = 'L = 8')
    plt.scatter(T_list, U_16, color = 'IndianRed' , label = 'L = 16')
    plt.scatter(T_list, U_32, color = 'forestGreen',label = 'L = 32')
    plt.legend()
    plt.show()