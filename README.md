# deopt

**deopt** is an experimental Python package for performing **Differential Evolution** (DE) global optimization. This is currently in the early stages of development.

## Version

**Current Version:** 0.0.0 (Alpha)

## Features

- Basic implementation of Differential Evolution (DE)
- Supports both logistic and tanh transformations for parameter mapping
- Includes Cauchy mutation for added diversity in mutation


```./src/de.py``` docstring:

```
Differential Evolution (DE) for global optimization

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

```

## Requirements
Python 3.12 or later. 
```
numpy>=2.1.3
```

## Installation (Local)

Clone the repository and install the package locally:

```bash
git clone https://github.com/ojegit/deopt.git
cd deopt
pip install -e .
```

## References

Storn, R., & Price, K. (1997). Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces. Journal of global optimization, 11, 341-359. <br/>
