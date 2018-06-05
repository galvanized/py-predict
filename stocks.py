import requests
import re
import time
import random
import os
import numpy as np
import sqlite3
import hashlib
from math import *
from statistics import *
from analysis import *

def hash_iterable(iterable):
    # todo: check if input is actually an iterable
    # please note that items with the same repr may collide
    s = b'hashme'
    for i in iterable:
            s = hashlib.sha3_256(s + hashlib.sha3_256(str(i).encode('utf-8')).hexdigest().encode('utf-8')).hexdigest().encode('utf-8')
    return s

def fdn(x):
    # float and de-null
    try:
        return float(x)
    except:
        return 0.0

class Database():
    def __init__(self, db_path=None):
        self.db_path = 'db.sqlite' if not db_path else db_path
        self.conn = sqlite3.connect(self.db_path)
        self.c = self.conn.cursor()

    def __del__(self):
        self.conn.commit()
        self.conn.close()
        del self

    def init_db(self):
        try:
            self.c.execute('''CREATE TABLE stocks
                (symbol text primary key, type text, quantity real)''')
            self.c.execute('''CREATE TABLE days
                (hash text primary key, symbol text, date text, open real, close real, high real, low real,
                 adjclose real, volume real, interpolated integer,
                 UNIQUE(hash))''')
            self.conn.commit()
            print("Created database.")
        except sqlite3.OperationalError:
            print("Database exists.")

    def update_stocks_table(self, symbol, dtype='stock'):
        self.c.execute('SELECT count(*) FROM days WHERE symbol=?', (symbol,))
        length = self.c.fetchone()[0]
        self.c.execute('REPLACE INTO stocks VALUES (?,?,?)', (symbol, dtype, length))
        self.conn.commit()



    def update_db_from_csv(self, data, symbol):
        l = data.splitlines()
        for r in l[1:]: #skip legend row
            self.add_db_row(r, symbol)
        self.conn.commit()
        self.update_stocks_table(symbol)

    def add_db_row(self, row, symbol):
        #Date,Open,High,Low,Close,Adj Close,Volume
        lines = row.split(',')
        d, o, h, l, c, ac, v = lines
        o = fdn(o)
        h = fdn(h)
        l = fdn(l)
        c = fdn(c)
        ac = fdn(ac)
        v = fdn(v)

        ha = hash_iterable((symbol, d))

        self.c.execute('INSERT OR IGNORE INTO days VALUES (?,?,?,?,?,?,?,?,?,?)', (ha, symbol, d, o, h, l, c, ac, v, 0))



    def yahoo_to_db(self, stocklist='stocks.txt'):

        s = requests.session()

        url = 'https://query1.finance.yahoo.com/v7/finance/download/{}?period1=-9999999999&period2=9999999999&interval=1d&events=history&crumb={}'

        p = re.compile('"CrumbStore":\{"crumb":"(.*?)"}')
        crumb = None
        while not crumb or '\\' in crumb:
            if crumb:
                s = requests.session()
            print("Getting crumb.")
            r1 = s.get('https://finance.yahoo.com/quote/AAPL/history')
            crumb = p.findall(r1.text)[0]
            print("Got crumb", crumb)
            cookies = s.cookies.get_dict()

        print('Cookies: {}\tCrumb: {}'.format(cookies,crumb))

        stocks = []
        with open(stocklist, 'r') as infile:
            stocks = infile.read().split('\n')

        stocks = stocks[:-1] # omit last 'stock', which is empty due to trailing newline

        i = 0
        for stock in stocks:
            i += 1
            time.sleep(random.triangular(0,10,0.3)) # courtsey delay
            r = s.get(url.format(stock, crumb), cookies=cookies)
            if r.status_code != 200:
                print("Got code {} on stock {}".format(r.status_code, stock))
            else:
                self.update_db_from_csv(r.text, stock)
                print('Got ' + stock.ljust(8) + '({} of {})'.format(i, len(stocks)))

    def close_and_dates(self, symbol):
        # return two lists: close values and corresponding dates
        closes = []
        dates = []
        for row in self.c.execute('SELECT close, date FROM days WHERE symbol=? ORDER BY date', (symbol,)):
            closes.append(row[0])
            dates.append(row[1])
        return closes, dates

    def n_day_changes(self, sym, n, input_length):
        close, date = self.close_and_dates(sym)
        if len(close) < input_length or n >= input_length:
            return None
        close = close[:input_length]
        change = []
        for i in range(input_length-n):
            if close[i+n] and close[i]: # resist breaking on zeroes
                change.append(close[i+n]/close[i])
        return change


    def list_symbols(self):
        self.c.execute('SELECT distinct symbol FROM days')
        return [x[0] for x in self.c.fetchall()]

    def analysis_csv(self, outfile='analysis.csv', a=None):
        syms = db.list_symbols()
        f = open(outfile,'w')
        f.write('SYMBOL,LAST,CLOSE,TREND,MOVING TREND,10 DAY TWEE, 100 DAY TWEE\n')
        for sym in syms:
            print('Analyzing', sym)
            try:
                dat, dates = db.close_and_dates(sym)

                line = sym + ',' #SYMBOL
                line += dates[-1] + ',' #LAST
                line += str(round(dat[-1],2)) + ',' #CLOSE


                moving = ema(dat, 0.01)
                dm = multipliers(moving)
                mdm = ema(dm, 0.01)

                # delta moving. percent per day
                # TREND
                line += str(round((dm[-1]-1)*100,2)) + ','

                # moving delta moving. percent per day
                # MOVING TREND
                line += str(round((mdm[-1]-1)*100,2)) + ','

                try:
                    line += str(round((twee(dat, 10)/dat[-1]-1)*100,2)) + ',' # 10 DAY TWEE
                except:
                    line += 'N/A' + ','

                try:
                    line += str(round((twee(dat, 100)/dat[-1]-1)*100,2)) + '\n'# 100 DAY TWEE
                except:
                    line += 'N/A' + '\n'

                f.write(line)
            except:
                print("FATAL ERROR! Stock excluded.")

        f.close()

    def simple_sample_generator(self, in_len, f_len, symbols=None, norm=True):
        '''
        in_len : input length, days
        f_len  : forecast length, days
        '''
        # todo: cache in memory
        # todo: read only necessary indexes
        queue = []
        if not symbols:
            symbols = self.list_symbols()
        lengths = []
        for symbol in symbols:
            self.c.execute('SELECT count(*) FROM days WHERE symbol=?', (symbol,))
            lengths.append(max(self.c.fetchone()[0]-in_len-f_len,0))
            print(symbol,lengths[-1])

        length_sum = sum(lengths)
        last_failure = False

        while 1:
            if not queue:
                # randomly select from stocks weighted by length
                target_value = random.random()*length_sum
                remaining = target_value
                target_i = 0
                for i, v in enumerate(lengths):
                    if v >= remaining:
                        x = i
                        while not len(symbols[x]):
                            # avoid selecting an empty stock
                            print("Queue skip! Avoid empty.")
                            x = (x + 1)%len(symbols)
                        target_i = x
                        break
                    else:
                        remaining -= v

                if last_failure:
                    print("Generating value from {}".format(symbols[target_i]))

                self.c.execute('SELECT close FROM days where symbol=? ORDER BY date',
                               (symbols[target_i],))
                closes = [x[0] for x in self.c.fetchall()]
                if lengths[target_i]-f_len-in_len < 1:
                    # avoid breaking on stocks that are too short
                    print("Queue skip! Too short.")
                    continue
                for i in range(10): # 10 samples per stock selected
                    samp_i = random.randrange(0,lengths[target_i]-f_len-in_len)
                    in_vals = closes[samp_i:samp_i+in_len]
                    f_val = closes[samp_i+in_len+f_len]
                    if norm:
                        if 0 in in_vals:
                            continue
                        denom = in_vals[-1]
                        in_vals = [x/denom for x in in_vals]
                        f_val = f_val/denom
                    queue.append((in_vals, f_val))
                    #print("Appending {} to queue.".format((in_vals, f_val)))

            try:
                yield queue.pop()
                last_failure = False
            except IndexError:
                if last_failure:
                    print('CRITICAL GENERATOR FAILURE')
                    raise GeneratorFailure('Failure in sample generation.')
                last_failure = True

    def read_values(self, symbol):
        #hash, symbol, date, open, close, high, low, adjclose, volume, interpolated
        self.c.execute('SELECT * FROM days where symbol=? ORDER BY date',
                       (symbol,))
        fetch = np.array(self.c.fetchall())
        fetch = np.transpose(fetch)
        keys = ['hash','symbol','date','open','close','high','low',
                'adjclose','volume','interpolated']
        nonnumerical = ['hash','symbol','date']
        out_dict = {}
        for i, key in enumerate(keys):
            out_dict[key] = fetch[i]
            if not key in nonnumerical:
                out_dict[key] = [float(x) for x in out_dict[key]]
        return out_dict


    def advanced_sample_generator(self, in_len, f_len, symbols=None,
                                  vectors=None, scalars=None, samps_per_stock=10):
        if vectors == None:
            vectors = ['open/close','new-norm close','old-norm close',
                       'high/low', 'max-norm volume', 'close mults']

        if scalars == None:
            scalars = ['mean','stdev','10 percentiles']

        queue = []
        if not symbols:
            symbols = self.list_symbols()
        lengths = []
        for symbol in symbols:
            self.c.execute('SELECT count(*) FROM days WHERE symbol=?', (symbol,))
            lengths.append(max(self.c.fetchone()[0]-in_len-f_len,0))
            print(symbol,lengths[-1])

        length_sum = sum(lengths)
        last_failures = 0

        while 1:
            if not queue:
                # randomly select from stocks weighted by length
                target_value = random.random()*length_sum
                remaining = target_value
                target_i = 0
                for i, v in enumerate(lengths):
                    if v >= remaining:
                        x = i
                        while not len(symbols[x]):
                            # avoid selecting an empty stock
                            print("Queue skip! Avoid empty.")
                            x = (x + 1)%len(symbols)
                        target_i = x
                        break
                    else:
                        remaining -= v

                if last_failures:
                    print("Generating value from {}".format(symbols[target_i]))

                sym_data = self.read_values(symbols[target_i])

                closes = sym_data['close']

                if lengths[target_i]-f_len-in_len < 1:
                    # avoid breaking on stocks that are too short
                    print("Queue skip! Too short.")
                    continue

                for i in range(samps_per_stock):
                    #multiple samples are taken from the same stock to improve
                    #lookup efficiency

                    # select the sample position
                    samp_i = random.randrange(0,lengths[target_i]-f_len-in_len)

                    in_vals = closes[samp_i:samp_i+in_len]
                    f_val = closes[samp_i+in_len+f_len]

                    vol_dat = sym_data['volume'][samp_i:samp_i+in_len]

                    if 0 in in_vals:
                        print("Zero found in closes for stock {}! Omitting.".format(symbols[target_i]))
                        break
                    if max(vol_dat) == 0:
                        print("Zero maximum volume for stock {}! Omitting.".format(symbols[target_i]))
                        break
                    denom = in_vals[-1]
                    last_normed = [x/denom for x in in_vals]
                    first_normed = [x/in_vals[0] for x in in_vals]

                    close_mults = [1] + multipliers(in_vals)

                    high_low = [x[0]/x[1] for x in zip(
                        sym_data['high'][samp_i:samp_i+in_len],
                        sym_data['low'][samp_i:samp_i+in_len])]

                    open_close = [x[0]/x[1] for x in zip(
                        sym_data['open'][samp_i:samp_i+in_len],
                        sym_data['close'][samp_i:samp_i+in_len])]

                    max_norm_vol = [x/max(vol_dat) for x in vol_dat]

                    f_val = f_val/denom

                    possible_vectors = ['open/close','new-norm close','old-norm close',
                               'high/low', 'max-norm volume', 'close mults']

                    matching_vectors = [open_close, last_normed, first_normed,
                                        high_low, max_norm_vol, close_mults]

                    selected_vectors = vectors

                    output_vectors = []

                    for v in selected_vectors:
                        output_vectors.append(matching_vectors[possible_vectors.index(v)])


                    queue.append((
                        np.array(output_vectors),
                        f_val))
                    #print("Appending {} to queue.".format((in_vals, f_val)))

            try:
                yield queue.pop()
                last_failures = 0
            except IndexError:
                if last_failures > 10:
                    # more than 10 attempts have been made and every one was a failure
                    # something is probably wrong. raise exception
                    print('CRITICAL GENERATOR FAILURE')
                    raise GeneratorFailure('Failure in sample generation.')
                last_failures += 1



class GeneratorFailure(Exception):
    pass


if __name__ == '__main__':
    db = Database('stockdata.sqlite')
    db.init_db()
    db.yahoo_to_db('stocks.txt')
    #g = db.advanced_sample_generator(30,100, symbols=['GOOG'])
    #print(next(g))
