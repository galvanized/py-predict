from math import *
from statistics import *
from scipy.stats import norm

def is_iter(v):
    # returns a boolean of whether v is iterable
    try:
        iterator = iter(v)
    except TypeError:
        return False
    else:
        return True

def exponential_factor(initial, final, delta):
    if initial == final or delta == 0:
        return 1
    return log(final/initial) / delta

'''
    VECTOR FUNCTIONS

    All of the below functions return a vector.


'''

def normalize(v, to_zero=False):
    if to_zero:
        m = min(v)
        v = [i-m for i in v]
    return v_multiply(v, 1/max(v))

def v_multiply(v, m):
    if is_iter(m):
        return [i*n for i,n in zip(v,m)]
    else:
        return [i*m for i in v]

def v_diff(v1, v2):
    return [i2 - i1 for i1,i2 in zip(v1, v2)]

def vs_diff(v, vs):
    # like v_diff but supporting both vector and scalar
    # note that the subtraction order is reversed
    if is_iter(vs):
        return v_diff(vs, v)
    else:
        return [val - vs for val in v]


def v_adiff(v1, v2):
    # absolute difference
    return [abs(i2 - i1) for i1,i2 in zip(v1, v2)]

def sma(v, n, f=mean):
    # simple moving average
    out = []
    if n > len(v):
        n = len(v)
    for i in range(1,len(v)+1):
        vals = v[max(0, i-n):i]
        print(vals)
        print(mean(vals), f(vals))
        out.append(f(vals))
    return out

def cma(v, f=mean):
    # cumulative moving average
    out = []
    for r in range(0,len(v)):
        out.append(f(v[:r+1]))
    return out

def wma(v, n):
    # weighted moving average
    if n > len(v):
        n = len(v)

    out = []

    divisor = sum([i*1/(n) for i in range(1,n+1)])

    for i in range(1,len(v)+1):
        num_q = min(i, n)
        if i < n:
            # recompute
            d = sum([q*1/(i) for q in range(1,i+1)])
        else:
            d = divisor
        vals = v[max(0, i-n):i+1]
        weights = []
        for q in range(1,num_q+1):
            weights.append(q * 1/num_q)

        weighted = [val * w for val,w in zip(vals, weights)]

        out.append(sum(weighted)/d)

    return out



def ema(v, a):
    if len(v) <= 1:
        return v
    out = [v[0]]
    for val in v[1:]:
        prev = out[-1]
        new = val*a + prev*(1-a)
        out.append(new)
    return out



def deltas(v):
    # difference between values, aka discrete derivative
    # WARNING: returns a length one less than the input
    out = []
    for i in range(len(v)-1):
        out.append(v[i+1]-v[i])
    return out


def summa(v, c=0):
    # cumulative sum, aka discrete integral
    out = []
    if c != None:
        out.append(c)
    for val in v:
        out.append(out[-1]+val)
    return out


'''
    SCALAR FUNCTIONS

    The below functions each return a scalar value

'''


def rms(v, w=None):
    # root mean square
    # todo: branch if weights not needed
    if w == None:
        w = [1]*len(v)
    cumulative_w = 0
    cumulative_v = 0
    for iv, iw in zip(v,w):
        cumulative_w += iw
        cumulative_v += iv**2 * iw
    return(sqrt(cumulative_v/cumulative_w))

def rmse(v, vs=0, w=None):
    # root mean square error
    # provide target as vs (can be vector or scalar)
    # w is weights for v
    return rms(vs_diff(v, vs), w)

def mae(v, vs=0, f=mean):
    # mean absolute error
    return f([abs(a) for a in vs_diff(v, vs)])

def exp_rate(v):
    return [exponential_factor(val, v[-1], len(v)-i-1) for i, val in enumerate(v)]

def twee(v, d, stdev=None, loc=None):
    # time-weighted exponential extrapolation

    # v: input vector
    # d: displacement of prediction from last input vector

    if loc == None:
        loc = len(v) - d
    if stdev == None:
        stdev = 2*d

    weights = [norm.pdf(x,loc=loc,scale=stdev) for x in range(len(v)-1)] + [0]
    efs = exp_rate(v)

    expected = sum([w*e for w, e in zip(weights, efs)])/sum(weights)

    return v[-1] * exp(expected * d)
