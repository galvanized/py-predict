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
                 train_samples = 1, test_samples = 1, validation_samples = 1):

    db = Database(database_path)
    db.init_db()


    if not symbols:
        symbols = db.list_symbols()

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

    print(lengths_sum, 'points available')

    indexes = list(range(lengths_sum))

    random.shuffle(indexes)

    validation_pairs = []
    test_pairs = []
    train_pairs = []

    ct = 0 # used for progress indication
    numlen = int(math.ceil(math.log10(needed_samples))) + 1

    current_index = 0

    for i in range(needed_samples + 1):
        ct += 1

        pt = None

        while type(pt) == type(None):

            target_sym, target_index = get_stock_index(symbols,lengths, indexes[current_index])
            print('selected {}/{}'.format(
                str(ct).zfill(numlen),str(needed_samples).zfill(numlen))
                  ,target_sym, target_index)

            try:
                pt = db.sample_point(in_len = in_len, f_len = f_len,
                                symbol=target_sym, index=target_index)
            except Exception as e:
                print("SAMPLE FAILURE!", e)

            current_index += 1


        validation_f = len(validation_pairs) / validation_samples
        test_f = len(test_pairs) / test_samples
        train_f = len(train_pairs) / train_samples

        fs = [validation_f, test_f, train_f]
        fs_min = min(fs)
        i_min = fs.index(fs_min)

        if i_min == 0:
            validation_pairs.append(pt)
        elif i_min == 1:
            test_pairs.append(pt)
        elif i_min == 2:
            train_pairs.append(pt)

        if min(fs) >= 1:
            # enough samples! we're done here.
            print("All samples collected!")
            break

        print('val  :',len(validation_pairs),
              'test :',len(test_pairs),
              'train:',len(train_pairs))





    np.savez_compressed(create_path, train=train_pairs, test=test_pairs, validation=validation_pairs)

def generate_recent(create_path = 'recent.npz', database_path = 'stockdata.sqlite',
                 in_len = 100, f_len = 10, symbols = None, vectors = None):

    db = Database(database_path)
    db.init_db()


    if not symbols:
        symbols = db.list_symbols()

    lengths = []

    lengths_sum = sum(lengths)

    indexes = list(range(lengths_sum))

    random.shuffle(indexes)

    recent_pairs = []
    recent_symbols = []

    needed_samples = len(symbols)

    ct = 0 # used for progress indication
    numlen = int(math.ceil(math.log10(needed_samples))) + 1

    current_index = 0

    for s in symbols:
        try:
            sym_data = db.read_values(s)
            closes = sym_data['close']

            in_vals = closes[-in_len:]
            if len(in_vals) < in_len:
                print('Not enough data for {}, skipping'.format(s))
                continue

            ptmin = min(in_vals)
            ptmax = max(in_vals)
            if ptmin == 0:
                print('Zero in {}, skipping'.format(s))
                continue
            if ptmax/ptmin > 100:
                print('Max/min ratio exceeds 100 in {}, skipping'.format(s))
                continue

            denom = in_vals[-1]
            last_normed = [x/denom for x in in_vals] + [0]*f_len

            pt = [last_normed,[1]*f_len]




            recent_pairs.append(pt)
            recent_symbols.append(s)

            print("Got {}.".format(s))

        except Exception as e:
            print("SAMPLE FAILURE ON {}!".format(s), e)


    np.savez_compressed(create_path, recent=recent_pairs, syms=recent_symbols)


if __name__ == '__main__':
    preset = 'medium'

    if preset == 'small':
        generate_npz(symbols = None, create_path='dataset1k-300in-20out.npz',
                     train_samples = 1000, test_samples = 1000, validation_samples = 1000,
                     in_len = 300, f_len = 20)

    elif preset == 'medium':
        generate_npz(symbols = None, create_path='10k-300in-20out.npz',
                     train_samples = 10000, test_samples = 5000, validation_samples = 2000,
                     in_len = 300, f_len = 20)

    elif preset == 'medium-trusted':
        generate_npz(symbols = ['^GSPC','^DJI','^IXIC','GOOG','AAPL','ETSY','SQ'],
                     create_path='dataset-trusted-10k-300in-20out.npz',
                     train_samples = 10000, test_samples = 5000, validation_samples = 2000,
                     in_len = 300, f_len = 20)

    elif preset == 'trusted-recent':
        generate_recent(symbols = ['^GSPC','^DJI','^IXIC','GOOG','AAPL','ETSY','SQ'],
                     create_path='dataset-recent-300in-20out.npz',
                     in_len = 300, f_len = 20)

    elif preset == 'big':
        generate_npz(symbols = None, create_path='dataset100k-300in-20out.npz',
                     train_samples = 100000, test_samples = 10000, validation_samples = 1000,
                     in_len = 300, f_len = 20)
