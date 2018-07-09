from neural import sets_from_npz
import numpy as np
from tpot import TPOTRegressor


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

    tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2)
    tpot.fit(train_x, train_y)
    print()
    print('-'*40)
    print(tpot.score(test_x, test_y))
    tpot.export('tpot_pure.py')

if __name__ == '__main__':
    generate_sr_npz()
    tpot_optimize()
