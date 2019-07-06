import numpy as np
from optimizer.shade_ils.structure import DEresult
from optimizer.shade_ils.shadeutils import random_indexes, shade_clip, limit_memory, update_params

def apply_shade(func, dim, lower, upper, population, fitness, maxevals, MemF, MemCR,
                current_best):

    # 确保shade 可以和ls 接上
    bestId = np.argmin(fitness)
    if current_best.fitness < fitness[bestId]:
        fitness[bestId] = current_best.fitness
        population[bestId] = current_best.solution

    totalevals = 0
    popsize = population.shape[0]
    H = popsize

    # Init memory with population
    memory = population.tolist()

    k = 0
    pmin = 2.0/popsize

    while totalevals < maxevals:
        SCR = []
        SF = []
        F = np.zeros(popsize)
        CR = np.zeros(popsize)
        u = np.zeros((popsize, dim))
        best_fitness = np.min(fitness)

        for (i, xi) in enumerate(population):
            # Getting F and CR for that solution
            index_H = np.random.randint(0, H)
            meanF = MemF[index_H]
            meanCR = MemCR[index_H]
            Fi = np.random.normal(meanF, 0.1)
            CRi = np.random.normal(meanCR, 0.1)
            p = np.random.rand()*(0.2-pmin)+pmin

            # Get two random values
            r1 = random_indexes(1, popsize, ignore=[i])
            # Get the second from the memory
            r2 = random_indexes(1, len(memory), ignore=[i, r1])
            xr1 = population[r1]
            xr2 = memory[r2]
            # Get one of the p best values
            maxbest = int(p*popsize)
            bests = np.argsort(fitness)[:maxbest]
            pbest = np.random.choice(bests)
            xbest = population[pbest]
            # Mutation
            v = xi + Fi*(xbest - xi) + Fi*(xr1-xr2)
            # Special clipping
            v = shade_clip(lower, upper, v, xi)
            # Crossover
            idxchange = np.random.rand(dim) < CRi
            u[i] = np.copy(xi)
            u[i, idxchange] = v[idxchange]
            F[i] = Fi
            CR[i] = CRi

        # Update population and SF, SCR
        weights = []

        for i, fit in enumerate(fitness):
            fit_u = func(u[i])

            if fit_u <= fit:
                # Add to memory
                if fit_u < fit:
                    memory.append(population[i])
                    SF.append(F[i])
                    SCR.append(CR[i])
                    weights.append(fit - fit_u)

                if (fit_u < best_fitness):
                    best_fitness = fit_u

                population[i] = u[i]
                fitness[i] = fit_u

        totalevals += popsize
        # Check the memory
        memory = limit_memory(memory, 2 * popsize)

        # Update MemCR and MemF
        if len(SCR) > 0 and len(SF) > 0:
            Fnew, CRnew = update_params(SF, SCR, weights)
            MemF[k] = Fnew
            MemCR[k] = CRnew
            k = (k + 1) % H

    bestId = np.argmin(fitness)
    return DEresult(solution=population[bestId], fitness=fitness[bestId], evaluations=totalevals)