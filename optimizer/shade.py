import numpy as np
import cec2013lsgo.cec2013 as cec

# 选出不重复的下标
def random_indexes(n, size, ignore=[]):
    indexes = [pos for pos in range(size) if pos not in ignore]

    assert len(indexes) >= n
    np.random.shuffle(indexes)

    if n == 1:
        return indexes[0]
    else:
        return indexes[:n]

# 处理超界解
def shade_clip(lower, upper, solution, original):
    clip_sol = np.clip(solution, lower, upper)

    if np.all(solution == clip_sol):
        return solution

    idx_lowest = (solution < lower)
    solution[idx_lowest] = (original[idx_lowest] + lower) / 2.0
    idx_upper = (solution > upper)
    solution[idx_upper] = (original[idx_upper] + upper) / 2.0
    return solution

# 更新参数
def update_params(SF, SCR, weights):
    weights = np.array(weights)
    w = weights/np.sum(weights)
    # 更新CR
    mean_CR = np.sum(w * SCR)
    mean_CR = np.clip(mean_CR, 0, 1)
    # 更新F
    mean_F = np.sum(w * SF* SF) / np.sum(w * SF)
    mean_F = np.clip(mean_F, 0, 1)
    return mean_CR, mean_F

# 限制A 的大小
def limit_memory(memory, memorySize):
    memory = np.array(memory)

    if len(memory) > memorySize:
        indexes = np.random.permutation(len(memory))[:memorySize]
        memory = memory[indexes]

    return memory.tolist()

def shade(func, upper, lower, dim, max_evals = 3e6, pop_size = 100, H = 100):
    # Initialize population P randomly
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([func(ind) for ind in population])
    current_evals = pop_size

    # Set all values in MCR, MF to 0.5
    mem_cr = np.ones(H) * 0.5
    mem_f = np.ones(H) * 0.5

    # Archive A = None
    A = []
    size_A = pop_size

    # Index counter k
    k =0
    pmin = float(2/pop_size)

    # while The termination criteria are not met do
    while current_evals < max_evals:
        SCR = []
        SF = []
        F = np.empty(pop_size)
        CR = np.empty(pop_size)
        u = np.empty((pop_size, dim))
        weights = []   # 用于更新CR 和 F
        # p = np.empty(pop_size)

        for i, ind in enumerate(population):
            # select from [1, H] randomly
            r = np.random.randint(0, H)
            # CR 取自正态分布
            CRi = np.random.normal(mem_cr[r], 0.1)
            CRi = np.clip(CRi, 0, 1)
            # F 取自柯西分布
            Fi = mem_f[r] * np.random.standard_cauchy() + 0.1
            while Fi<=0:
                Fi = mem_f[r] * np.random.standard_cauchy() + 0.1
            if Fi > 1:
                Fi = 1
            p = np.random.uniform(low=pmin, high=0.2, size=1)

            # 变异， current-to-pbest/1 策略
            r1 = random_indexes(1, pop_size, ignore=[i])
            # xr2 is selected from PuA
            r2 = random_indexes(1, pop_size+len(A), ignore=[i, r1])
            xr1 = population[r1]
            if len(A)!=0:
                PUA = np.concatenate((population,np.array(A)), axis=0)
                xr2 = PUA[r2]
            else:
                xr2 = population[r2]
            # Get one of the p best values
            maxbest = int(p * pop_size)
            bests = np.argsort(fitness)[:maxbest]
            pbest = np.random.choice(bests)
            xbest = population[pbest]
            # Mutation: current-to-pbest/1
            v = ind + Fi * (xbest - ind) + Fi * (xr1 - xr2)
            # Special clipping
            v = shade_clip(lower, upper, v, ind)
            # 交叉
            jrands = np.random.randint(low=0, high=dim, size=dim)
            js = np.arange(0, dim)
            idx1 = np.where(jrands == js)[0]
            idx2 = np.where(np.random.rand(dim) <= CRi)[0]
            idxchange = np.concatenate((idx1, idx2), axis=0)
            idxchange = np.unique(idxchange)
            u[i] = ind[:]
            u[i, idxchange] = v[idxchange]
            CR[i] = CRi
            F[i] = Fi

        for i, fit in enumerate(fitness):
            fit_u = func(u[i])
            if fit_u <= fit:
                if fit_u<fit:
                    A.append(population[i])
                    SCR.append(CR[i])
                    SF.append(F[i])
                    weights.append(fit - fit_u)
                population[i] = u[i]
                fitness[i] = fit_u
        current_evals += pop_size

        # Whenever the size of A , randomly selected individuals are deleted so that A<=P
        A = limit_memory(A, size_A)

        if len(SCR)!= 0 and len(SF)!=0:
            # update MCR MF based on SCR, SF
            mean_CR, mean_F = update_params(SF, SCR, weights)
            mem_cr[k] = mean_CR
            mem_f[k] = mean_F
            k = k+1
            if k>=H:
                k = 0

        # 对population排序
        bestIndex = np.argsort(fitness)
        population = population[bestIndex]
        fitness = fitness[bestIndex]
        print('current evals: {}; current fitness: {}'.format(current_evals, fitness[0]))


if __name__ == '__main__':
    bench = cec.Benchmark()
    for i in range(2, 16):
        print(i)
        fun = bench.get_function(i)
        info = bench.get_info(i)
        dim = info['dimension']
        upper = info['upper']
        lower = info['lower']
        shade(fun, upper, lower, dim)