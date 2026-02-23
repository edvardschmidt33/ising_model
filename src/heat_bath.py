import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm.auto import tqdm
from numba import njit


def p(s,i, j, L, J):
    ### Here beta is ignored to follow the scheme of the original logic where beta is baked in to J ###

    t = s[i - 1 if i>0 else L-1, j]
    b = s[i + 1 if i<L-1 else 0, j]
    l = s[i, j - 1 if j>0 else L-1]
    r = s[i, j + 1 if j<L-1 else 0]
    num = np.exp(2*J*sum(t,b,r,l))
    den = 1 + num
    return num/den

