from .utils import (
    pars_transf, inv_pars_transf, cauchy_mutate, choose
)

import numpy as np

# ---------------------------
# Differential Evolution (DE)
# ---------------------------

def de(obj_fun_, 
       lb, 
       ub, 
       pars_init = None, 
       no_diff = 1,
       mutate_first_term = 2, 
       NP = 10, 
       CR = 0.8, 
       F = [0.1995,0.2005],
       cauchy_mutation_prob = -1, 
       gen_max = 400,
       F_gen_policy = 'generation', 
       cauchy_mutation_policy = 'generation',
       gamma = [0.01,0.001],
       xmin_update_policy = 'generation', 
       of_min_stuck_stop = 10, 
       cof_tol = 1e-3,
       cof_min_tol = -1,
       print_int = 1):
    
    """
    Differential Evolution (DE) for global minimization

    Parameters:
    -----------
    obj_fun_ : function
        The objective function to be minimized. This function should take
        a single input vector and return a scalar value.
    
    lb : array-like
        Lower bounds for the parameters, should match the dimensionality
        of the problem.
    
    ub : array-like
        Upper bounds for the parameters, should match the dimensionality
        of the problem.
    
    pars_init : array-like, optional
        Initial guess for the parameters. If not provided, random initialization
        within bounds `lb` and `ub` is used.
    
    no_diff : int, optional
        The number of difference vectors used in mutation. Default is 1.
    
    mutate_first_term : int, optional
        Controls how the first mutation term is generated:
        - 1: Random mutation
        - 2: Best individual mutation
        - 3: Mixed mutation
    
    NP : int, optional
        The population size (number of individuals in each generation).
        Default is 10.
    
    CR : float, optional
        Crossover probability. Determines the likelihood of an individual
        being modified by a trial vector. Should be between 0 and 1.
    
    F : list of floats, optional
        The scaling factor(s) used for mutation. Default is [0.1995, 0.2005].
    
    cauchy_mutation_prob : float, optional
        The probability of applying Cauchy mutation. If <= 0, Cauchy mutation
        is not used. Default is -1 (no Cauchy mutation).
    
    gen_max : int, optional
        The maximum number of generations for the optimization process.
    
    F_gen_policy : str, optional
        Determines how mutation scaling factors are generated:
        - 'generation' : scaling factor changes each generation
        - 'parameter' : scaling factor remains constant throughout.
    
    cauchy_mutation_policy : str, optional
        Determines when Cauchy mutation is applied:
        - 'generation' : apply Cauchy mutation every generation
        - 'parameter' : apply Cauchy mutation based on parameters.
    
    gamma : list of floats, optional
        Parameters controlling the decay rate for Cauchy mutation.
    
    xmin_update_policy : str, optional
        Determines when to update the best solution:
        - 'generation' : update each generation
        - 'parameter' : update based on parameter comparison.
    
    of_min_stuck_stop : int, optional
        Maximum number of generations with no improvement in the best
        objective function value before stopping.
    
    cof_tol : float, optional
        Convergence tolerance for stopping criteria based on change in objective
        function value.
    
    cof_min_tol : float, optional
        Minimum tolerance for change in objective function value before stopping.
    
    print_int : int, optional
        Controls the verbosity of the output. If greater than 0, progress
        information will be printed every `print_int` generations.

    Returns:
    --------
    xmin : array-like
        The best solution found by the algorithm.
    
    of_min : float
        The objective function value corresponding to the best solution.
    
    exit_flag : int
        A flag indicating why the optimization process stopped:
        - 1: Convergence reached based on objective function tolerance.
        - 2: Stagnation (no improvement for `of_min_stuck_stop` generations).
        - 3: Objective function change below `cof_min_tol`.
        - 4: Reached `gen_max` generations.
    """
    
    ###
    mutate_first_term_3_prob = 0.5 #probability of choosing between 1 and 2 when mutate_first_term = 3
    cof_hist_count = 100 #no of historical values for change in objective function value
    no_dec = 7 #no decimals shown by print progress
    transf_type = 1 #parameter transformation: 1=logistic, 2=tanh
    ###


    # check inputs
    if no_diff < 1:
        raise Exception('no_diff must be at least 1.')
    if cauchy_mutation_prob > 0:
        if cauchy_mutation_prob > 1:
            raise Exception('cauchy_mutation_prob must be [0,1].')
    if mutate_first_term not in [1,2,3]:
        raise Exception('mutate_first_term must be 1-3.')
    if F_gen_policy not in ['generation','parameter']:
        raise Exception('F_gen_policy must be either ''generation'' or ''parameter''.')
    if cauchy_mutation_policy not in ['generation','parameter']:
        raise Exception('cauchy_mutation_policy must be either ''generation'' or ''parameter''.')
    if xmin_update_policy not in ['generation','parameter']:
        raise Exception('xmin_update_policy must be either ''generation'' or ''parameter''.')
    if not 0 < CR <= 1:
        raise Exception('CR must be in [0,1].')
    if gen_max <= 0:
        raise Exception('gen_max must be positive.')
    if not len(gamma) == 2:
        raise Exception('gamma must have exactly 2 elements.')
    
    
    cb = lb is not None and ub is not None
    if pars_init is None:
        if not cb:
            raise ValueError("Must supply lb/ub when pars_init is None.")
        #endif
        D = len(lb)
        
        #pars_init = np.random.uniform(lb[:, None], ub[:, None], (D, NP))
        pars_init = np.zeros((D,NP))
        for i in range(NP):
            pars_init[:,i] = lb + (ub - lb) * np.random.rand(D)
        #endfor
    else:
        D, NP = pars_init.shape
    #endif

    if cb:
        pars_init = inv_pars_transf(pars_init, lb, ub, NP, transf_type)
        obj_fun = lambda x: obj_fun_(pars_transf(x, lb, ub, 1, transf_type))
    else:
        obj_fun = obj_fun_
    #endif
    
    gamma1, gamma0 = gamma
    x1 = pars_init.copy()
    x2 = np.copy(x1)
    trial = np.zeros(D)
    l = 0
    exit_flag = 0
    stuck = 0
    cof_min = np.inf
    of_min_stuck = 0
    no_mutations = 0
    gamma_ = -1
    cof_hist = np.zeros(cof_hist_count)

    cost = np.array([obj_fun(x1[:, i]) for i in range(NP)])
    of_min = np.min(cost)
    imin = np.argmin(cost)
    xmin = x1[:, imin].copy()
    score = of_min

    for count in range(1, gen_max + 1):

        if F_gen_policy == 'generation':
            Fj = F[0] if len(F) == 1 else F[0] + (F[1] - F[0]) * np.random.rand()
        #endif
        
        for i in range(NP):
            if mutate_first_term == 1:
                r = choose(2 * no_diff + 1, NP, i)
            elif mutate_first_term == 2:
                r = choose(2 * no_diff, NP, i)
            elif mutate_first_term == 3:
                w = np.random.rand()
                r = choose(2 * no_diff, NP, i) if w < mutate_first_term_3_prob else choose(2 * no_diff + 1, NP, i)
            #endif
            
            j = np.random.randint(D)
            for k in range(D):
                if np.random.rand() < CR or k == D - 1:
                    if mutate_first_term == 1:
                        xf = x1[j, r[2 * no_diff]]  # only if r has 2*no_diff+1 elements
                    elif mutate_first_term == 2:
                        xf = xmin[j]
                    elif mutate_first_term == 3:
                        xf = xmin[j] if w < mutate_first_term_3_prob else x1[j, r[0]]  # use r[0] safely

                    if F_gen_policy == 'parameter':
                        Fj = F[0] if len(F) == 1 else F[0] + (F[1] - F[0]) * np.random.rand()
                    #endif

                    dx = sum(Fj * (x1[j, r[2 * d]] - x1[j, r[2 * d + 1]]) for d in range(no_diff))
                    trial[j] = xf + dx

                    if cauchy_mutation_policy == 'parameter' and cauchy_mutation_prob > 0:
                        trial[j], I, gamma_ = cauchy_mutate(trial[j], cauchy_mutation_prob,
                                                            (count - 1) / (gen_max - 1), 1, gamma1, gamma0)
                        no_mutations += np.sum(I)
                    #endif
                else:
                    trial[j] = x1[j, i]
                #endif
                
                j = (j + 1) % D

                if xmin_update_policy == 'parameter':
                    score_prev = score
                    score = obj_fun(trial)
                    cof = abs(score - score_prev)

                    if not np.isnan(cof) and not np.isinf(cof):
                        if l < cof_hist_count:
                            cof_hist[l] = cof
                            l += 1
                        else:
                            cof_hist[:-1] = cof_hist[1:]
                            cof_hist[-1] = cof
                        #endif

                        if score <= cost[i]:
                            cost[i] = score
                            x2[j, i] = trial[j]
                            stuck = 0
                        else:
                            x2[j, i] = x1[j, i]
                            stuck += 1
                        #endif
                            
                        imin = np.argmin(cost)
                        xmin[j] = x2[j,imin]
                        
                    else:
                        stuck += 1
                    #endif
                        
                    x1[j,i] = x2[j,i]
                #endif
            #endfor
            
            if cauchy_mutation_policy == 'generation' and cauchy_mutation_prob > 0 and gamma1 > 0 and gamma0 > 0:
                trial, I, gamma_ = cauchy_mutate(trial, cauchy_mutation_prob,
                                                 (count - 1) / (gen_max - 1), D, gamma1, gamma0)
                no_mutations += np.sum(I)                
            #endif
    
            if xmin_update_policy == 'generation':
                score_prev = score
                score = obj_fun(trial)
                cof = abs(score - score_prev)

                if not np.isnan(cof) and not np.isinf(cof):
                    if l < cof_hist_count:
                        cof_hist[l] = cof
                        l += 1
                    else:
                        cof_hist[:-1] = cof_hist[1:]
                        cof_hist[-1] = cof
                    #endif

                    if score <= cost[i]:
                        cost[i] = score
                        x2[:, i] = trial
                        stuck = 0
                    else:
                        x2[:, i] = x1[:, i]
                        stuck += 1
                    #endif

                    imin = np.argmin(cost)
                    xmin = x2[:,imin].copy()

                else:
                    stuck += 1
                #endif
            #endif
        #endfor
        
        x1 = x2.copy()
        of_min_prev = of_min
        of_min, imin = np.min(cost), np.argmin(cost)
        if of_min < of_min_prev:
            cof_min = np.abs(of_min - of_min_prev)
            of_min_stuck = 0
        else:
            of_min_stuck += 1
        #endif
        
        xmin = x1[:,imin].copy()
        
        if print_int > 0 and count % print_int == 0:
            mutations_pct = 100 * no_mutations/(count*D*NP);
      
            print('GEN NO %d/%d | OF_MIN: %.*e, COF_MIN: %.*e (%d) | COF: %.*e (%d) | MUT: %.2f %% (gamma: %.2e)' % (
               count,gen_max,no_dec,of_min,no_dec,cof_min,of_min_stuck,no_dec,np.mean(cof_hist[0:l]),stuck,mutations_pct,gamma_))
        #endif
        
        if l >= cof_hist_count and np.mean(cof_hist) < cof_tol:
            exit_flag = 1
        elif of_min_stuck >= of_min_stuck_stop:
            exit_flag = 2
        elif abs(cof_min) <  cof_min_tol:
            exit_flag = 3
        elif count == gen_max:
            exit_flag = 4
        #endif
        
        if exit_flag > 0:
            if print_int > 0:
                print(f'exit_flag: {exit_flag}')
            break
        #endif
    #endfor

    if cb:
        xmin = pars_transf(xmin, lb, ub, 1, transf_type)
    #endif
        
    if print_int > 0:
        print('Done!')
    #enfif

    return xmin, of_min, exit_flag
#enddef