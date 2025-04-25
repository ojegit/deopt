# deopt

**deopt** is an experimental Python package for performing **Differential Evolution** (DE) optimization. This is currently in the early stages of development.

## Version

**Current Version:** 0.0.0 (Alpha)

## Features

- Basic implementation of Differential Evolution (DE)
- Supports both logistic and tanh transformations for parameter mapping
- Includes Cauchy mutation for added diversity in mutation


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
