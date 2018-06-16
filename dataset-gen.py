#np.save(file, arr)
#np.savez_compressed(file, a=array1, etc)
#np.load['a'] = array1

'''
dataset-gen.py

Routines for .npz creation for later training.

'''

import random
import numpy as np
from stocks import Database

class NotEnoughDataError(Exception):
    pass

def overlap_detection(la, lb):
    for i in la:
        if i in lb:
            return True
    return False

def generate_npz(create_path = 'dataset.npz', database_path = 'stockdata.sqlite',
                 in_len = 100, f_len = 10, symbols = None,
                 train_samples = 10000, test_samples = 1000, validation_samples = 100):

    db = Database('stockdata.sqlite')
    db.init_db()

    if not symbols:
        symbols = self.list_symbols()

    lengths = []

    for symbol in symbols:
        db.c.execute('SELECT count(*) FROM days WHERE symbol=?', (symbol,))
        lengths.append(max(db.c.fetchone()[0]-in_len-f_len,0))
        print(symbol,lengths[-1])

    lengths_sum = sum(lengths)
    needed_samples = train_samples + test_samples + validation_samples

    if  needed_samples > lengths_sum:
        raise NotEnoughDataError('Only {} points available, need at least {}'.format(
            lengths_sum, needed_samples
        ))

    indexes = list(range(lengths_sum))

    random.shuffle(indexes)

    # list of 'edge' indexes: min and max for slices
    edge_i = [0, train_samples]
    edge_i.append(edge_i[-1] + test_samples)
    edge_i.append(edge_i[-1] + validation_samples)

    train_indexes = indexes[edge_i[0]:edge_i[1]]
    test_indexes = indexes[edge_i[1]:edge_i[2]]
    verification_indexes = indexes[edge_i[2]:edge_i[3]]

    # todo: switch these to a test
    print(len(train_indexes), len(test_indexes), len(verification_indexes))

    overlap = overlap_detection(train_indexes,test_indexes) +\
            overlap_detection(verification_indexes,test_indexes) +\
            overlap_detection(verification_indexes,train_indexes)

    print('OVERLAP DETECTED!' if overlap else 'No overlap detected.')


if __name__ == '__main__':
    generate_npz(symbols = ['GOOGL'])
