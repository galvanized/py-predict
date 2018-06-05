'''
    An implementation of the German Tank Problem to demonstrate comparing
    and evaluating different models.
'''

import random
import math
import statistics
from main import SingleOutputModel

def generate_serials(N, q):
    '''
    Generate q (quantity) numbers ranging from 1 to N, inclusive,
    without replacement.

    '''
    return random.sample(range(1,N), q)

class MaxMethod(SingleOutputModel):
    def __init__(self):
        self.name = 'MaxMethod'

    def eval(self, x):
        return(max(x))

class PartitionMethod(SingleOutputModel):
    def __init__(self):
        self.name = 'PartitonMethod'

    def eval(self, x):
        return(max(x)*5/4)


class MeanMethod(SingleOutputModel):
    def __init__(self):
        self.name = 'MeanMethod'

    def eval(self, x):
        return(statistics.mean(x)*2 - 1)

class MedianMethod(SingleOutputModel):
    def __init__(self):
        self.name = 'MeanMethod'

    def eval(self, x):
        return(statistics.median(x)*2)


ss = []
xs = []
ys = []
q = 10

m = PartitionMethod()

for i in range(1000):
    n = random.randrange(50,1000)
    ss.append(generate_serials(n, 10))
    xs.append(n)

ys = m.run_multi(ss)
m.simple_scatter_plot(xs, ys)
