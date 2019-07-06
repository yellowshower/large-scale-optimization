import numpy as np
import cec2013lsgo.cec2013 as cec
from optimizer.shade_ils.structure import DEresult, PoolLast
from optimizer.shade_ils.local_search import reset_ls, apply_ls
from optimizer.shade_ils.global_search import apply_shade

def get_ratio_improvement(previous_fit, current_fit):
    if previous_fit == 0:
        improvement = 0
    else:
        improvement = (previous_fit-current_fit)/previous_fit

    return improvement

def shadeils(func, lower, upper, dim, max_evals = 3e6, pop_size = 100, H = 100, threshold=0.05):

    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([func(ind) for ind in population])
    bestId = np.argmin(fitness)
    totalevals = pop_size

    init_sol = np.ones(dim) * ((lower+upper)/2)
    init_fit = func(init_sol)
    totalevals += 1

    if init_fit < fitness[bestId]:
        population[bestId] = init_sol
        fitness[bestId] = init_fit

    current_best = DEresult(solution=population[bestId],
                            fitness=fitness[bestId],
                            evaluations=totalevals)

    # 记录全局最优解
    best_global_solution = current_best.solution
    best_global_fitness = current_best.fitness
    print('initial evals: {} best fitness: {}'.format(totalevals, best_global_fitness))

    methods = ['mts', 'grad']
    evals_gs = min(50 * dim, 25000)
    evals_de = min(50 * dim, 25000)
    evals_ls = min(10 * dim, 5000)

    pool_global = PoolLast(methods)
    pool_local = PoolLast(methods)

    SR_global_mts = reset_ls(dim, lower, upper)
    SR_local_mts = reset_ls(dim, lower, upper)
    num_worse = 0
    num_restart = 0

    # shade 的参数memf和memcr
    MemF = np.ones(H) * 0.5
    MemCR = np.ones(H) * 0.5

    while totalevals < max_evals:
        previous_fitness = current_best.fitness  # 区别previous_fit

        # global search : mts / grad
        previous_fit = current_best.fitness
        method_global = pool_global.get_new()
        current_best = apply_ls('Global', method_global, func, dim, lower, upper, current_best.solution, current_best.fitness, evals_gs, SR_global_mts)
        totalevals += current_best.evaluations
        improvement = get_ratio_improvement(previous_fit, current_best.fitness)
        pool_global.improvement(method_global, improvement, 2)
        print('Global : {} evals: {} best fitness: {}'.format(method_global, totalevals, current_best.fitness))

        # global search: shade
        current_best = apply_shade(func, dim, lower, upper, population, fitness, evals_de, MemF, MemCR, current_best)
        totalevals += current_best.evaluations
        print('Global : shade evals: {} best fitness: {}'.format(totalevals, current_best.fitness))

        # local search: mts / grad
        previous_fit = current_best.fitness
        method_local = pool_local.get_new()
        current_best = apply_ls('Local', method_local, func, dim, lower, upper, current_best.solution, current_best.fitness, evals_ls, SR_local_mts)
        totalevals += current_best.evaluations
        improvement = get_ratio_improvement(previous_fit, current_best.fitness)
        pool_local.improvement(method_local, improvement, 10, 0.25)
        print('Local : {} evals: {} best fitness: {}'.format(method_local, totalevals, current_best.fitness))

        current_best_solution = current_best.solution
        current_best_fitness = current_best.fitness
        current_best = DEresult(solution=current_best_solution, fitness=current_best_fitness, evaluations=totalevals)

        if current_best_fitness < best_global_fitness:
            best_global_fitness = current_best.fitness
            best_global_solution = current_best.solution

        # restart if no improvement
        if (previous_fitness == 0):
            ratio_improvement = 1
        else:
            ratio_improvement = (previous_fitness - current_best.fitness) / previous_fitness

        if ratio_improvement >= threshold:
            num_worse = 0
        else:
            num_worse += 1
            SR_local_mts = reset_ls(dim, lower, upper)
            SR_global_mts = reset_ls(dim, lower, upper)

        if num_worse >= 3:
            num_worse = 0
            # Increase a 1% of values
            posi = np.random.choice(pop_size)
            new_solution = np.random.uniform(0, 1, dim) * 0.1 * (upper - lower) + population[posi]
            new_solution = np.clip(new_solution, lower, upper)
            new_fitness = func(new_solution)
            totalevals += 1
            current_best = DEresult(solution=new_solution, fitness=new_fitness, evaluations=totalevals)

            # init DE
            population = np.random.uniform(low=lower, high=upper, size=(pop_size, dim))
            fitness = np.array([func(ind) for ind in population])
            MemF = np.ones(H) * 0.5
            MemCR = np.ones(H) * 0.5
            totalevals += pop_size

            # restart ls
            pool_local.reset()
            pool_global.reset()
            SR_local_mts = reset_ls(dim, lower, upper)
            SR_global_mts = reset_ls(dim, lower, upper)
            num_restart += 1
            print('evals: {} have restarted'.format(totalevals))

if __name__ =='__main__':
    bench = cec.Benchmark()
    i = 1
    fun = bench.get_function(i)
    info = bench.get_info(i)
    dim = info['dimension']
    upper = info['upper']
    lower = info['lower']
    shadeils(fun, lower, upper, dim)
