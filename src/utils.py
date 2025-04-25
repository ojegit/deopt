# ---------------------------
# Helper Functions
# ---------------------------

import numpy as np

def cauchy_mutate(x, prob, k, D, gamma1, gamma0):
    """
    Mutate vector x with Cauchy noise, controlled by mutation probability and gamma decay.
    Note: this might result in very large x_mut if gamma1 and gamma0 are not chosen carefully!
    (one could cap the output regardless)
    """
    I = np.random.rand(D) < prob
    if gamma1 > gamma0:
        gamma_ = gamma1 * (gamma0 / gamma1) ** (-k)
    else:
        gamma_ = gamma1
    x_mut = np.copy(x)
    x_mut[I] += gamma_ * np.tan((np.random.rand(np.sum(I)) - 0.5) * np.pi)
    return x_mut, I, gamma_

def choose(k, n, i):
    """
    Randomly choose k unique indices from range(n) excluding i.
    """
    if k > n - 1:
        raise ValueError("choose error: k is larger than n-1")
    if not (0 <= i < n):
        raise ValueError("choose error: i is not in [0, n-1]")
    r = np.delete(np.arange(n), i)
    np.random.shuffle(r)
    return r[:k]

def logistic(x,lb,ub):
    y = lb + (ub-lb) / (1 + np.exp(-x))
    return y

def tanh(x,lb,ub):
    y = 2 / (1 + np.exp(-2*x)) - 1
    y = (ub+lb)/2 + (ub-lb)*y/2
    return y

def inv_logistic(y,lb,ub):
    x = np.log((y-lb)/(ub-y))
    return x

def inv_tanh(y,lb,ub):
    x = (2*y - (lb + ub)) / (ub - lb)
    x = 0.5 * np.log( (1 + x) / (1 - x))
    return x

def pars_transf(x, lb, ub, N, trans_type):
    lb_ = np.array(lb).reshape(-1,1) #(n,) -> (n,1)
    ub_ = np.array(ub).reshape(-1,1)
    if x.ndim == 1:
        x_ = np.array(x).reshape(-1,1)
    else:
        x_ = np.array(x)
    
    #lb_, ub_ = lb.copy(), ub.copy()
    lb_ = lb_ @ np.ones((1,N))
    ub_ = ub_ @ np.ones((1,N))
    
    if trans_type == 1:
        return logistic(x_, lb_, ub_)  # Apply logistic transformation
    elif trans_type == 2:
        return tanh(x_, lb_, ub_)  # Apply tanh transformation

def inv_pars_transf(y, lb, ub, N, trans_type):
    lb_ = np.array(lb).reshape(-1,1) #(n,) -> (n,1)
    ub_ = np.array(ub).reshape(-1,1)
    if y.ndim == 1:
        y_ = np.array(y).reshape(-1,1)
    else:
        y_ = np.array(y)
    
    #lb_, ub_ = lb.copy(), ub.copy()
    lb_ = lb_ @ np.ones((1,N))
    ub_ = ub_ @ np.ones((1,N))
    
    if trans_type == 1:
        return inv_logistic(y_, lb_, ub_)  # Apply inverse logistic transformation
    elif trans_type == 2:
        return inv_tanh(y_, lb_, ub_)  # Apply inverse tanh transformation
