from deopt import de
import numpy as np

#objective function
def rosen(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

d = 3 #dimensionality
lb, ub = np.full(d, -5), np.full(d, 10) #bounds

#minimize
x_min, of_min, flag = de(rosen, lb, ub, NP=10, CR=0.7, F=[0.5], gen_max=1000)

#results
print("x_min (ground truth 1):", x_min)
print("of_min (ground truth 0):", of_min)