import numpy as np

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