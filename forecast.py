'''

    Forecast

    A one-click future-predicting csv-exporting script that uses pretrained models.
'''

import stocks
from models import nuclearninja
import dataset_gen

sym_file = 'watch.txt'
syms = []
with open(sym_file, 'r') as infile:
    syms = infile.read().split('\n')
syms = syms[:-1] # omit last 'stock', which is empty due to trailing newline

db = stocks.Database('stockdata.sqlite')
db.init_db()
db.yahoo_to_db(sym_file)
dataset_gen.generate_recent(symbols = syms,
             create_path='dataset-recent-300in-20out.npz',
             in_len = 300, f_len = 20)
nuclearninja.model(*nuclearninja.data('dataset-trusted-10k-300in-20out.npz'),
                   load_existing='models/'+nuclearninja.version_name()+'.best.hd5',
                   skip_train=True,
                   load_recent='dataset-recent-300in-20out.npz',)
