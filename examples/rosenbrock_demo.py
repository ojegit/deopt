from deopt import de
import numpy as np

def rosen(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

d = 3
lb, ub = np.full(d, -5), np.full(d, 10)

x_min, of_min, flag = de_v1(rosen, lb, ub, NP=10, CR=0.7, F=[0.5], gen_max=1000)

print("x_min:", x_min)
print("of_min:", of_min)