from neural import sets_from_npz
import numpy as np
from tpot import TPOTRegressor

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline

from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV


def npz_to_single_regression(dataset_path, sr_path):
    d = sets_from_npz(dataset_path)

    sets = ['train','test','validation']
    out_dict = {}

    for s in sets:
        print(s)
        xs = d[s+'_x']
        ys = d[s+'_y']

        code_xs = xs
        code_ys = np.array([y[-1] for y in ys])

        out_dict[s+'_x'] = np.array(code_xs)
        out_dict[s+'_y'] = np.array(code_ys)

    np.savez_compressed(sr_path, train_y=out_dict['train_y'], train_x=out_dict['train_x'],
                        test_y=out_dict['test_y'],test_x=out_dict['test_x'],
                        validation_y=out_dict['validation_y'],validation_x=out_dict['validation_x'])


def generate_sr_npz():
    dataset_path = 'dataset.npz'
    sr_path = 'singleregression.npz'
    npz_to_single_regression(dataset_path,sr_path)

def tpot_optimize():
    d = np.load('singleregression.npz')

    train_x, train_y = d['train_x'], d['train_y']
    test_x, test_y = d['test_x'], d['test_y']

    #print(train_x.shape, train_y.shape)

    tpot = TPOTRegressor(generations=10, population_size=80, verbosity=2, memory='auto')
    tpot.fit(train_x, train_y)
    print()
    print('-'*40)
    print(tpot.score(test_x, test_y))
    tpot.export('tpot_pure.py')

def predict_on_xs_v1(xs):

    d = np.load('singleregression.npz')

    train_x, train_y = d['train_x'], d['train_y']
    test_x, test_y = d['test_x'], d['test_y']

    exported_pipeline = make_pipeline(
        FastICA(tol=0.8),
        KNeighborsRegressor(n_neighbors=65, p=1, weights="distance")
    )

    exported_pipeline.fit(train_x, train_y)
    results = exported_pipeline.predict(xs)
    return results

def predict_on_xs_v2(xs):

    d = np.load('singleregression.npz')

    train_x, train_y = d['train_x'], d['train_y']
    test_x, test_y = d['test_x'], d['test_y']

    # Score on the training set was:-0.006897601822671224
    exported_pipeline = make_pipeline(
        PCA(iterated_power=2, svd_solver="randomized"),
        FeatureAgglomeration(affinity="euclidean", linkage="ward"),
        ElasticNetCV(l1_ratio=1.0, tol=0.01)
    )

    exported_pipeline.fit(train_x, train_y)
    results = exported_pipeline.predict(xs)
    return results

def s(n):
    # round. short for 'shorten'
    return round(n,3)

def compare_arrays(a, b, print_arrays = False, print_diff = False):
    diffs = abs(b - a)
    print('median:  {}   mean: {}   stdev: {}\nr-square: {}'.format(
        s(np.median(diffs)), s(np.mean(diffs)), s(np.std(diffs)),
        s(np.corrcoef(a, b)[0][1]**2)
    ))

if __name__ == '__main__':
    #generate_sr_npz()
    #tpot_optimize()


    d = np.load('singleregression.npz')

    v_x, v_y = d['validation_x'], d['validation_y']

    p_y = predict_on_xs_v2(v_x)

    print('model')
    compare_arrays(v_y, p_y)

    print('random')
    r_y = np.copy(v_y)
    np.random.shuffle(r_y)
    compare_arrays(v_y, r_y)
