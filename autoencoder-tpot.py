from neural import ModelDorkyDandelion, sets_from_npz
import numpy as np
from tpot import TPOTRegressor

from tpot_only import compare_random

import pandas as pd
from sklearn.feature_selection import SelectFwe, SelectPercentile, f_regression
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

def code_to_npz(model, dataset_path, code_path):
    d = sets_from_npz(dataset_path)

    sets = ['train','test','validation']
    out_dict = {}

    for s in sets:
        print(s)
        xs = d[s+'_x']
        ys = d[s+'_y']

        code_xs = model.get_code_activations(xs)
        code_ys = np.array([y[-1] for y in ys])

        out_dict[s+'_x'] = np.array(code_xs)
        out_dict[s+'_y'] = np.array(code_ys)

        #print(code_xs.shape, code_ys.shape)


    np.savez_compressed(code_path, train_y=out_dict['train_y'], train_x=out_dict['train_x'],
                        test_y=out_dict['test_y'],test_x=out_dict['test_x'],
                        validation_y=out_dict['validation_y'],validation_x=out_dict['validation_x'])



def generate_code_npz():
    m = ModelDorkyDandelion()
    dataset_path = 'dataset.npz'
    code_path = 'code.npz'
    code_to_npz(m, dataset_path, code_path)

def tpot_optimize():
    d = np.load('code.npz')

    train_x, train_y = d['train_x'], d['train_y']
    test_x, test_y = d['test_x'], d['test_y']

    #print(train_x.shape, train_y.shape)

    tpot = TPOTRegressor(generations=10, population_size=100, verbosity=2)
    tpot.fit(train_x, train_y)
    print()
    print('-'*40)
    print(tpot.score(test_x, test_y))
    tpot.export('tpot_code_dorky.py')

def predict_on_xs(xs):

    d = np.load('code.npz')

    train_x, train_y = d['train_x'], d['train_y']

    # Score on the training set was:-0.006897601822671224
    exported_pipeline = make_pipeline(
        PCA(iterated_power=2, svd_solver="randomized"),
        FeatureAgglomeration(affinity="euclidean", linkage="ward"),
        ElasticNetCV(l1_ratio=1.0, tol=0.01)
    )

    exported_pipeline.fit(train_x, train_y)
    results = exported_pipeline.predict(xs)
    return results

if __name__ == '__main__':
    #generate_code_npz()
    #tpot_optimize()

    d = np.load('code.npz')

    v_x, v_y = d['validation_x'], d['validation_y']

    p_y = predict_on_xs(v_x)
    print(p_y)

    compare_random(v_y, p_y)
