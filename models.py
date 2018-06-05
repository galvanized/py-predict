from main import *
from analysis import *
from stocks import Database
import random
from math import *


def sago(errf, params, param_mags=None, pows=20, iters=None, reps=10):
    '''
    Simulated Annealing General Optimizer

    errf = Error Function, takes params as argument and returns error
    params = Parameters, a list of numbers passed as argument to errf.
                Returned at end of optimization.
    param_mags = Parameter magnitudes, a tuple of numbers. Best if an overestimate is provided.
    pows = Powers of simulated annealing, defaults to 20
    iters = Attempts per power, defaults to 2**len(params)
    reps = Repetitions of simulated annealing starting at param_mags, defaults to 10
    '''
    scales = [2**x for x in range(0,-pows,-1)]

    if not param_mags:
        param_mags = [1]*len(params)

    if not iters:
        iters = 2**len(params)

    best_params = params
    best_err = errf(params)

    for rep in range(reps):
        for s in scales:
            for i in range(iters):
                for p in range(len(params)):
                    new_params = best_params[:]
                    variability = 2*param_mags[p]*s
                    new_params[p] += random.uniform(-variability, variability)
                    new_err = errf(new_params)
                    if new_err < best_err:
                        best_err = new_err
                        best_params = new_params

    return best_params

def single_stock_optimimal_twee(symbol, length, train_f_start=0.2, train_f_end=0.8):
    # returns optimal twee parameters
    db = Database('stockdata.sqlite')
    stock_data = db.close_and_dates(symbol)[0]
    if len(stock_data) < length:
        print("Not enough data for twee", len(stock_data))
        return [0,0]

    start_data_index = int(floor(len(stock_data)*train_f_start))
    end_data_index = int(floor(len(stock_data)*train_f_end - length))
    step_size = int((end_data_index - start_data_index)/10)

    opt_params = []

    for i in range(start_data_index, end_data_index-30, step_size):
        print(i)

        def errf(params):
            dat = stock_data[:i+30]
            t = twee(dat, length, stdev=params[0], loc=params[1])
            dt = stock_data[i+30+length] - t
            return dt**2

        opt = sago(errf, [10,20], [1000,1000], pows=10, iters=10, reps=3)
        print(opt)
        opt_params.append(opt)

    print(opt_params)

    stdevs = [x[0] for x in opt_params]
    displacements = [x[1] for x in opt_params]

    return [median(stdevs), median(displacements)]
