import collections
import numpy as np

# 存储最好个体和最好适应度的结构
DEresult = collections.namedtuple('DEresults', ['solution', 'fitness', 'evaluations'])
# 记录MTS 算法global search 和 local search部分
SR_global_MTS = []
SR_local_MTS = []

# 记录 不同local search方法的improvement
class PoolLast:
    def __init__(self, options):
        """
        Constructor
        :param options:to store (initially the probability is equals)
        :return:
        """
        size = len(options)
        assert size > 0

        self.options = np.copy(options)  # 方法
        self.improvements = []
        self.count_calls = 0
        self.first = np.random.permutation(self.options).tolist()

        self.new = None
        self.improvements = dict(zip(options, [0] * size))  # 字典类型

    def reset(self):
        self.first = np.random.permutation(self.options).tolist()
        self.new = None
        options = self.options
        size = len(options)
        self.improvements = dict(zip(options, [0] * size))

    def has_no_improvement(self):
        return np.all([value == 0 for value in self.improvements.values()])

    def get_new(self):
        """
        Get one of the options, following the probabilities
        :return: one of the stored object
        """
        # First time it returns all
        if self.first:
            return self.first.pop()

        if self.new is None:
            self.new = self.update_prob()

        return self.new

    def is_empty(self):
        counts = self.improvements.values()
        return np.all(np.array(list(counts)) == 0)

    def improvement(self, obj, account, freq_update, minimum=0.15):
        """
        Received how much improvement this object has obtained (higher is better), it only update
        the method improvements

        :param object:
        :param account: improvement obtained (higher is better), must be >= 0
        :param freq_update: Frequency of improvements used to update the ranking
        :return: None
        """
        if account < 0:
            return

        if obj not in self.improvements:
            raise Exception("Error, object not found in PoolProb")

        previous = self.improvements[obj]
        self.improvements[obj] = account
        self.count_calls += 1

        if self.first:
            return

        if not self.new:
            self.new = self.update_prob()
        elif account == 0 or account < previous:
            self.new = self.update_prob()

    def update_prob(self):
        """
        update the probabilities considering improvements value, following the equation
        prob[i] = Improvements[i]/TotalImprovements

        :return: None
        """

        if np.all([value == 0 for value in self.improvements.values()]):
            # import ipdb; ipdb.set_trace()
            tmps = np.random.permutation(self.options).tolist()
            new_method = tmps[0]
            # print("new_method: {}".format(new_method))
            return new_method

        # Complete the ranking
        indexes = sorted(self.improvements.items(),key=lambda x:x[1])
        best = indexes[-1][0]
        # indexes = np.argsort(self.improvements.values())
        # posbest = indexes[-1][0]
        # best = list(self.improvements.keys())[posbest]
        return best