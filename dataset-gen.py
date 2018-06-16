#np.save(file, arr)
#np.savez_compressed(file, a=array1, etc)
#np.load['a'] = array1

'''
dataset-gen.py

Routines for .npz creation for later training.


todo: memmap support
'''

import random
import math
import numpy as np
from stocks import Database

class NotEnoughDataError(Exception):
    pass

def overlap_detection(la, lb):
    for i in la:
        if i in lb:
            return True
    return False

def get_stock_index(symbols, lengths, index):
    '''
    Returns the symbol and corresponding index of the symbol.
    '''
    i_cumsum = 0
    for i, l in enumerate(lengths):
        if l + i_cumsum >= index:
            return symbols[i], index - i_cumsum
        else:
            i_cumsum += l
    raise NotEnoughDataError("Group index ({}) exceeds sum of indexes ({})".format(
        index, sum(lengths)
    ))

def sample_point():
    possible_vectors = ['open/close','new-norm close past','old-norm close past',
               'high/low', 'max-norm volume', 'close mults',
               'new-norm close future', 'new-norm close future and past']
    matching_vectors = [open_close, last_normed, first_normed,
                        high_low, max_norm_vol, close_mults,
                        future_close, total_close]

    possible_scalars = ['future close','mean','stdev']
    matching_scalars = [future_close,mean,stdev]

    selected_vectors = vectors

    output_vectors = []

    for v in selected_vectors:
        output_vectors.append(matching_vectors[possible_vectors.index(v)])


def generate_npz(create_path = 'dataset.npz', database_path = 'stockdata.sqlite',
                 in_len = 100, f_len = 10, symbols = None, vectors = None,
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

    # sample at indexes
    train_pairs = []
    ct = 0
    numlen = int(math.ceil(math.log10(train_samples))) + 1
    for i in train_indexes:
        ct += 1
        target_sym, target_index = get_stock_index(symbols,lengths, i)
        print('selected {}/{}'.format(
            str(ct).zfill(numlen),str(train_samples).zfill(numlen))
              ,target_sym, target_index)
        train_pairs.append(
            db.sample_point(in_len = in_len, f_len = f_len,
                            symbol=target_sym, index=target_index)
            )

    return train_pairs




if __name__ == '__main__':
    print(
    generate_npz(symbols = ['^DJI','GOOG','MMM'], train_samples = 10)
    )
