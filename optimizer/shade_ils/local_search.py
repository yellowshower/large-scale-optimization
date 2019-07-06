import numpy as np
from .structure import DEresult
from optimizer.shade_ils.mts import mtsls
from scipy.optimize import fmin_l_bfgs_b

def reset_ls(dim, lower, upper):
    SR = np.ones(dim) * (upper - lower) * 0.2
    return SR

def apply_ls(name, method,
             func, dim, lower, upper,
             current_best_solution, current_best_fitness, maxevals,
             SR_local_mts):

    bounds = [[lower, upper]] * dim

    if method == 'grad':
        sol, fit, info = fmin_l_bfgs_b(func, x0=current_best_solution, approx_grad=True, bounds=bounds, maxfun=maxevals, disp=False)
        funcalls = info['funcalls']
    elif method == 'mts':
        res, SR = mtsls(func, current_best_solution, current_best_fitness, lower, upper, maxevals, SR_local_mts)
        sol = res.solution
        fit = res.fitness
        funcalls = maxevals
    else:
        raise NotImplementedError(method)

    if fit <= current_best_fitness:
        return DEresult(solution=np.array(sol), fitness=fit, evaluations=funcalls)
    else:
        return DEresult(solution=current_best_solution, fitness=current_best_fitness, evaluations=funcalls)
