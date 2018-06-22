from main import *
from stocks import *
import numpy as np
import keras
from keras import models
from keras.models import Sequential
from keras.layers import *
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model

import os.path



class ModelSushiSpree(SingleOutputModel):
    '''
    SushiSpree

    Test model using split mlp and cnn archetecture.
    '''

    def __init__(self, path='SushiSpree.hd5', ignore_existing=False,
                 train_stocks = ['^DJI','^GSPC']):

        self.name = 'SushiSpree'
        self.path = path
        self.input_length = 400
        self.forecast_length = 20
        self.input_layers = 6
        self.x_shape = (self.input_layers ,self.input_length)
        self.train_stocks = train_stocks

        if ignore_existing or not os.path.isfile(path): # if file doesn't exist
            dropout_level = 0.25

            i0 = Input(shape=self.x_shape, name='input_layer')
            f0 = Flatten()(i0)
            l0 = Reshape(self.x_shape[::-1])(i0)

            fff = Dense(128, activation='relu', name='feedforward_flattened')(f0)

            ffs = Dense(128, activation='relu', name='feedforward_structured')(i0)




            ca0 = Conv1D(32, 8, activation='relu')(l0)
            ca1 = Conv1D(32, 8, activation='relu')(ca0)
            ca2 = Conv1D(32, 8, activation='relu')(ca1)
            ca3 = MaxPooling1D()(ca2)

            ct = concatenate([Dropout(dropout_level)(Flatten()(x)) for x in \
                [ca3, ffs] ]+ [fff])

            mlp0 = Dense(128, activation='relu')(ct)
            mlp1 = Dense(128, activation='relu')(Dropout(dropout_level)(mlp0))
            mlp2 = Dense(32, activation='relu')(Dropout(dropout_level)(mlp1))
            mlp3 = Dense(8, activation='linear')(mlp2)

            o0 = Dense(1, activation='linear')(mlp3)

            self.model = Model(inputs=i0, outputs=o0)
            self.model.compile(loss='mean_squared_logarithmic_error',
                          optimizer='adadelta', #adam #RMSprop
                          metrics=['mse','mean_absolute_percentage_error'])

        else:
            self.import_model_from_file(path)

    def import_model_from_file(self, path):
        self.model = models.load_model(path)

    def train(self, train_stocks=None, batches = 10, epochs_per_batch=64, items_per_batch=128):
        if not train_stocks:
            train_stocks = self.train_stocks

        db = Database('stockdata.sqlite')
        giter = db.advanced_sample_generator(self.input_length, self.forecast_length, train_stocks)

        tb_callback = keras.callbacks.TensorBoard(log_dir='/tmp/sp-tb', write_graph=False)
        nan_callback = keras.callbacks.TerminateOnNaN()

        for b in range(batches):
            xs = []
            ys = []
            for i in range(items_per_batch):
                nx, ny = next(giter)
                xs.append(nx)
                ys.append([ny])

            xs = np.array(xs)
            ys = np.array(ys)

            print(xs.shape, ys.shape)

            self.model.fit(xs, ys, epochs=epochs_per_batch,
                           callbacks = [tb_callback, nan_callback])

            self.model.save(self.path)

    def evaluate(self, steps=100):
        db = Database('stockdata.sqlite')
        giter = db.advanced_sample_generator(self.input_length, self.forecast_length, self.train_stocks)
        print(self.model.evaluate_generator(giter, use_multiprocessing=True))


    def eval(self, x):
        return self.model.predict(x)

class ModelTwistyTrambone(SingleOutputModel):
    '''
    Twisty Trambone

    More layers! More convolutions! More connectivity! More dropout!

    '''

    def __init__(self, path='TwistyTrambone.hd5', ignore_existing=False,
                 train_stocks = ['^DJI','^GSPC']):

        self.name = 'TwistyTrambone'
        self.path = path
        self.input_length = 400
        self.forecast_length = 20
        self.input_layers = 6
        self.x_shape = (self.input_layers ,self.input_length)
        self.train_stocks = train_stocks

        if ignore_existing or not os.path.isfile(path): # if file doesn't exist
            dropout_level = 0.3

            i0 = Input(shape=self.x_shape, name='input_layer')
            f0 = Flatten()(i0)
            l0 = Reshape(self.x_shape[::-1])(i0)

            fff = Dense(128, activation='relu', name='feedforward_flattened')(f0)

            ffs0 = Dense(128, activation='relu', name='feedforward_structured0')(i0)
            ffs1 = Dense(128, activation='relu', name='feedforward_structured1')(ffs0)

            ca0 = Conv1D(32, 8, activation='relu')(l0)
            ca1 = Conv1D(32, 8, activation='relu')(ca0)
            ca2 = Conv1D(32, 8, activation='relu')(ca1)
            ca3 = MaxPooling1D()(ca2)

            cb0 = Conv1D(16, 32, activation='relu')(l0)
            cb1 = Conv1D(16, 32, activation='relu')(cb0)
            cb2 = MaxPooling1D()(cb1)

            cc0 = Conv1D(1, 7, activation='relu')(l0)

            ct = concatenate([Dropout(dropout_level)(Flatten()(x)) for x in \
                [ca0, ca1, ca2, ca3,
                 cb0, cb1, cb2,
                 cc0,
                 ffs1] ]+ [fff])

            mlp0 = Dense(128, activation='relu')(ct)
            mlp1 = Dense(128, activation='relu')(Dropout(dropout_level)(mlp0))
            mlp2 = Dense(32, activation='relu')(Dropout(dropout_level)(mlp1))
            mlp3 = Dense(8, activation='linear')(mlp2)

            o0 = Dense(1, activation='linear')(mlp3)

            self.model = Model(inputs=i0, outputs=o0)
            self.model.compile(loss='mean_squared_logarithmic_error',
                          optimizer='adadelta', #adam #RMSprop
                          metrics=['mse','mean_absolute_percentage_error'])

        else:
            self.import_model_from_file(path)

    def import_model_from_file(self, path):
        self.model = models.load_model(path)

    def train(self, train_stocks=None, batches = 10, epochs_per_batch=64, items_per_batch=128):
        #if not train_stocks:
        #    train_stocks = self.train_stocks

        db = Database('stockdata.sqlite')
        giter = db.advanced_sample_generator(self.input_length, self.forecast_length, train_stocks)

        tb_callback = keras.callbacks.TensorBoard(log_dir='/tmp/sp-tb', write_graph=False)
        nan_callback = keras.callbacks.TerminateOnNaN()

        for b in range(batches):
            xs = []
            ys = []
            for i in range(items_per_batch):
                nx, ny = next(giter)
                xs.append(nx)
                ys.append([ny])

            xs = np.array(xs)
            ys = np.array(ys)

            print(xs.shape, ys.shape)

            self.model.fit(xs, ys, epochs=epochs_per_batch,
                           callbacks = [tb_callback, nan_callback])

            self.model.save(self.path)

    def plot(self, steps=1000):
        db = Database('stockdata.sqlite')
        #['MSFT','BGNE','SHOP','RF','MMM','EC','GMED','SQ','ETSY','BABA','VKTX','CCMP','ADBE','CBOE','ALGN','BLKB']
        predict_stocks = ['INGN','WB','BIDU','AMTD','BA']
        giter = db.advanced_sample_generator(self.input_length, self.forecast_length, predict_stocks)

        xs = []
        ys = []
        for i in range(steps):
            nx, ny = next(giter)
            xs.append(nx)
            ys.append([ny])

        xs = np.array(xs)
        ys = np.array(ys)

        ps = self.model.predict(xs)

        self.graph_one_one_error(ys.flatten(), ps.flatten(), 'TwistyTrambone.png')




    def eval(self, x):
        return self.model.predict(x)

class ModelUnbiasedUnicorn(SingleOutputModel):
    '''
    Unbiased Unicorn

    Recurrent archetecture.

    '''

    def __init__(self, path=None, ignore_existing=False,
                 train_stocks = ['^DJI','^GSPC']):

        self.name = 'UnbiasedUnicorn'
        self.path = path if path else self.name+'.hd5'
        self.input_length = 600
        self.forecast_length = 100
        self.input_layers = 6
        self.x_shape = (self.input_layers ,self.input_length)
        self.train_stocks = train_stocks

        if ignore_existing or not os.path.isfile(self.path): # if file doesn't exist
            dropout_level = 0.5

            i0 = Input(shape=self.x_shape, name='input_layer')
            f0 = Flatten()(i0)
            l0 = Reshape(self.x_shape[::-1])(i0)

            g = GRU(1024)(i0)


            mlp0 = Dense(16, activation='relu')(g)
            #mlp1 = Dense(256, activation='relu')(Dropout(dropout_level)(mlp0))
            #mlp2 = Dense(32, activation='relu')(Dropout(dropout_level)(mlp1))
            #mlp3 = Dense(8, activation='linear')(mlp2)

            o0 = Dense(1, activation='linear')(mlp0)

            self.model = Model(inputs=i0, outputs=o0)
            self.model.compile(loss='mean_squared_logarithmic_error',
                          optimizer='adadelta', #adam #RMSprop
                          metrics=['mse','mean_absolute_percentage_error'])

        else:
            self.import_model_from_file(self.path)

    def import_model_from_file(self, path):
        self.model = models.load_model(path)

    def train(self, train_stocks=None, batches = 10, epochs_per_batch=64, items_per_batch=128):
        #if not train_stocks:
        #    train_stocks = self.train_stocks

        db = Database('stockdata.sqlite')
        giter = db.advanced_sample_generator(self.input_length, self.forecast_length, train_stocks)

        tb_callback = keras.callbacks.TensorBoard(log_dir='/tmp/sp-tb', write_graph=False)
        nan_callback = keras.callbacks.TerminateOnNaN()

        for b in range(batches):
            print("Batch {} of {}".format(b+1, batches))
            xs = []
            ys = []
            for i in range(items_per_batch):
                nx, ny = next(giter)
                xs.append(nx)
                ys.append([ny])

            xs = np.array(xs)
            ys = np.array(ys)

            print(xs.shape, ys.shape)

            self.model.fit(xs, ys, epochs=epochs_per_batch,
                           callbacks = [tb_callback, nan_callback])

            self.model.save(self.path)

    def plot(self, stocks=None, steps=1000):
        db = Database('stockdata.sqlite')
        #['MSFT','BGNE','SHOP','RF','MMM','EC','GMED','SQ','ETSY','BABA','VKTX','CCMP','ADBE','CBOE','ALGN','BLKB']
        #predict_stocks = ['INGN','WB','BIDU','AMTD','BA']
        giter = db.advanced_sample_generator(self.input_length, self.forecast_length, stocks)

        xs = []
        ys = []
        for i in range(steps):
            nx, ny = next(giter)
            xs.append(nx)
            ys.append([ny])

        xs = np.array(xs)
        ys = np.array(ys)

        ps = self.model.predict(xs)

        self.graph_one_one_error(ys.flatten(), ps.flatten(), self.name + '.png')




    def eval(self, x):
        return self.model.predict(x)

class ModelVelvetViper(SingleOutputModel):
    '''
    VelvetViper

    Test model using pure MLP.
    '''

    def __init__(self, path='VelvetViper.hd5', ignore_existing=False,
                 train_stocks = ['^DJI','^GSPC']):

        self.name = 'VelvetViper'
        self.path = path
        self.input_length = 400
        self.forecast_length = 20
        self.input_layers = 6
        self.x_shape = (self.input_layers ,self.input_length)
        self.train_stocks = train_stocks

        if ignore_existing or not os.path.isfile(path): # if file doesn't exist
            dropout_level = 0.25

            i0 = Input(shape=self.x_shape, name='input_layer')
            f0 = Flatten()(i0)


            mlp0 = Dense(1024, activation='relu')(f0)
            mlp1 = Dense(256, activation='relu')(Dropout(dropout_level)(mlp0))
            mlp2 = Dense(16, activation='relu')(Dropout(dropout_level)(mlp1))
            mlp3 = Dense(256, activation='relu')(Dropout(dropout_level)(mlp2))
            mlp4 = Dense(256, activation='relu')(Dropout(dropout_level)(mlp3))
            mlp5 = Dense(8, activation='linear')(mlp4)

            o0 = Dense(1, activation='linear')(mlp5)

            self.model = Model(inputs=i0, outputs=o0)
            self.model.compile(loss='mean_squared_logarithmic_error',
                          optimizer='adadelta', #adam #RMSprop
                          metrics=['mse','mean_absolute_percentage_error'])

        else:
            self.import_model_from_file(path)

    def import_model_from_file(self, path):
        self.model = models.load_model(path)

    def train(self, train_stocks=None, batches = 10, epochs_per_batch=64, items_per_batch=128):
        if not train_stocks:
            train_stocks = self.train_stocks

        db = Database('stockdata.sqlite')
        giter = db.advanced_sample_generator(self.input_length, self.forecast_length, train_stocks)

        tb_callback = keras.callbacks.TensorBoard(log_dir='/tmp/sp-tb', write_graph=False)
        nan_callback = keras.callbacks.TerminateOnNaN()

        for b in range(batches):
            xs = []
            ys = []
            for i in range(items_per_batch):
                nx, ny = next(giter)
                xs.append(nx)
                ys.append([ny])

            xs = np.array(xs)
            ys = np.array(ys)

            print(xs.shape, ys.shape)

            self.model.fit(xs, ys, epochs=epochs_per_batch,
                           callbacks = [tb_callback, nan_callback])

            self.model.save(self.path)

    def evaluate(self, steps=100):
        db = Database('stockdata.sqlite')
        giter = db.advanced_sample_generator(self.input_length, self.forecast_length, self.train_stocks)
        print(self.model.evaluate_generator(giter, use_multiprocessing=True))


    def eval(self, x):
        return self.model.predict(x)

    def plot(self, stocks=None, steps=1000):
        db = Database('stockdata.sqlite')
        #['MSFT','BGNE','SHOP','RF','MMM','EC','GMED','SQ','ETSY','BABA','VKTX','CCMP','ADBE','CBOE','ALGN','BLKB']
        #predict_stocks = ['INGN','WB','BIDU','AMTD','BA']
        giter = db.advanced_sample_generator(self.input_length, self.forecast_length, stocks)

        xs = []
        ys = []
        for i in range(steps):
            nx, ny = next(giter)
            xs.append(nx)
            ys.append([ny])

        xs = np.array(xs)
        ys = np.array(ys)

        ps = self.model.predict(xs)

        self.graph_one_one_error(ys.flatten(), ps.flatten(), self.name + '.png')

class ModelZenZeppelin(SingleOutputModel):
    '''
    ZenZeppelin

    Test model .
    '''

    def __init__(self, path='ZenZeppelin.hd5', ignore_existing=False,
                 train_stocks = ['^DJI','^GSPC']):

        self.name = 'ZenZeppelin'
        self.path = path
        self.input_length = 400
        self.forecast_length = 20
        self.input_layers = 6
        self.x_shape = (self.input_layers ,self.input_length)
        self.train_stocks = train_stocks

        self.act = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)

        if ignore_existing or not os.path.isfile(path): # if file doesn't exist

            i0 = Input(shape=self.x_shape, name='input_layer')
            l0 = Reshape(self.x_shape[::-1])(i0)
            f0 = Flatten()(i0)

            #conv -> prelu -> norm

            block0 = MaxPooling1D()(Conv1D(32, 4, strides=2, activation='relu')(l0))
            block1 = MaxPooling1D()(Conv1D(32, 4, strides=2, activation='relu')(block0))
            block2 = MaxPooling1D()(Conv1D(32, 4, strides=2, activation='relu')(block1)) # code layer
            #block3 = MaxPooling1D()(Conv1D(32, 1, dilation_rate=4, activation=PReLU())(block2))
            #block4 = MaxPooling1D()(Conv1D(32, 1, dilation_rate=4, activation=PReLU())(block3))
            blockout = Flatten()(block2)

            mlp0 = Dense(1024, activation='relu')(blockout)
            mlp1 = Dense(256, activation='relu')(mlp0)
            mlp2 = Dense(64, activation='relu')(mlp1)
            mlp3 = Dense(16, activation='linear')(mlp2)

            o0 = Dense(1, activation='linear')(blockout)

            self.model = Model(inputs=i0, outputs=o0)
            self.model.compile(loss='mean_squared_logarithmic_error',
                          optimizer='adadelta', #adam #RMSprop
                          metrics=['mse','mean_absolute_percentage_error'])

        else:
            self.import_model_from_file(path)

    def import_model_from_file(self, path):
        self.model = models.load_model(path)

    def train(self, train_stocks=None, batches = 10, epochs_per_batch=64, items_per_batch=128):
        if not train_stocks:
            train_stocks = self.train_stocks

        db = Database('stockdata.sqlite')
        giter = db.advanced_sample_generator(self.input_length, self.forecast_length, train_stocks)

        tb_callback = keras.callbacks.TensorBoard(log_dir='/tmp/sp-tb', write_graph=False)
        nan_callback = keras.callbacks.TerminateOnNaN()

        for b in range(batches):
            xs = []
            ys = []
            for i in range(items_per_batch):
                nx, ny = next(giter)
                xs.append(nx)
                ys.append([ny])

            xs = np.array(xs)
            ys = np.array(ys)

            print(xs.shape, ys.shape)

            self.model.fit(xs, ys, epochs=epochs_per_batch,
                           callbacks = [tb_callback, nan_callback])

            self.model.save(self.path)

    def evaluate(self, steps=100):
        db = Database('stockdata.sqlite')
        giter = db.advanced_sample_generator(self.input_length, self.forecast_length, self.train_stocks)
        print(self.model.evaluate_generator(giter, use_multiprocessing=True))


    def eval(self, x):
        return self.model.predict(x)

    def plot(self, stocks=None, steps=1000):
        db = Database('stockdata.sqlite')
        #['MSFT','BGNE','SHOP','RF','MMM','EC','GMED','SQ','ETSY','BABA','VKTX','CCMP','ADBE','CBOE','ALGN','BLKB']
        #predict_stocks = ['INGN','WB','BIDU','AMTD','BA']
        giter = db.advanced_sample_generator(self.input_length, self.forecast_length, stocks)

        xs = []
        ys = []
        for i in range(steps):
            nx, ny = next(giter)
            xs.append(nx)
            ys.append([ny])

        xs = np.array(xs)
        ys = np.array(ys)

        ps = self.model.predict(xs)

        self.graph_one_one_error(ys.flatten(), ps.flatten(), self.name + '.png')

class ModelAntsyAntonym(SingleOutputModel):
    '''
    AntsyAntonym

    Close-only MLP autoencoder.
    '''

    def __init__(self, path='AntsyAntonym.hd5', ignore_existing=False,
                 train_stocks = ['^DJI','^GSPC']):

        self.name = 'AntsyAntonym'
        self.path = path
        self.input_length = 400
        self.forecast_length = 1
        self.input_layers = 1
        self.x_shape = (self.input_layers ,self.input_length)
        self.train_stocks = train_stocks

        if ignore_existing or not os.path.isfile(path): # if file doesn't exist

            i0 = Input(shape=self.x_shape, name='input_layer')
            f0 = Flatten()(i0)

            mlp0 = Dense(1024, activation='relu')(f0)
            mlp1 = Dense(256, activation='relu')(mlp0)
            mlp2 = Dense(64, activation='relu')(mlp1)
            mlp3 = Dense(4, activation='relu')(mlp2)
            mlp4 = Dense(64, activation='relu')(mlp3)
            mlp5 = Dense(256, activation='relu')(mlp4)
            mlp6 = Dense(1024, activation='relu')(mlp5)

            o0 = Dense(self.input_length, activation='linear')(mlp6)
            o1 = Reshape(target_shape=self.x_shape)(o0)

            self.model = Model(inputs=i0, outputs=o1)
            self.model.compile(loss='mean_squared_error',
                          optimizer='adadelta', #adam #RMSprop
                          metrics=['mse','mean_absolute_percentage_error'])

        else:
            self.import_model_from_file(path)

    def import_model_from_file(self, path):
        self.model = models.load_model(path)

    def train(self, train_stocks=None, batches = 10, epochs_per_batch=64, items_per_batch=128):
        if not train_stocks:
            train_stocks = self.train_stocks

        db = Database('stockdata.sqlite')
        giter = db.advanced_sample_generator(self.input_length, self.forecast_length, train_stocks, vectors=['new-norm close'])

        tb_callback = keras.callbacks.TensorBoard(log_dir='/tmp/sp-tb', write_graph=False)
        nan_callback = keras.callbacks.TerminateOnNaN()

        for b in range(batches):
            xs = []
            for i in range(items_per_batch):
                nx, ny = next(giter)
                xs.append(nx)

            xs = np.array(xs)

            self.model.fit(xs, xs, epochs=epochs_per_batch,
                           callbacks = [tb_callback, nan_callback])

            self.model.save(self.path)

    def evaluate(self, steps=100):
        db = Database('stockdata.sqlite')
        giter = db.advanced_sample_generator(self.input_length, self.forecast_length, self.train_stocks, vectors=['new-norm close'])
        print(self.model.evaluate_generator(giter, use_multiprocessing=True))


    def eval(self, x):
        return self.model.predict(x)

class ModelBoozyBovine(SingleOutputModel):
    '''
    BoozyBovine

    Mixed model using both unsupervised and supervised metrics.
    '''

    def __init__(self, path='BoozyBovine.hd5', ignore_existing=False,
                 train_stocks = ['^DJI','^GSPC']):

        self.name = 'BoozyBovine'
        self.path = path
        self.input_length = 400
        self.forecast_length = 20
        self.input_layers = 1
        self.x_shape = (self.input_layers ,self.input_length)
        self.train_stocks = train_stocks

        if ignore_existing or not os.path.isfile(path): # if file doesn't exist

            i0 = Input(shape=self.x_shape, name='input_layer')
            f0 = Flatten()(i0)

            mlp0 = Dense(1024, activation='relu')(f0)
            mlp1 = Dense(256, activation='relu')(mlp0)
            mlp2 = Dense(64, activation='relu')(mlp1)
            mlp3 = Dense(4, activation='relu')(mlp2)
            mlp4 = Dense(64, activation='relu')(mlp3)
            mlp5 = Dense(256, activation='relu')(mlp4)
            mlp6 = Dense(1024, activation='relu')(mlp5)

            o0 = Dense(self.input_length, activation='linear')(mlp6)
            o1 = Reshape(target_shape=self.x_shape, name='autoencoder_output')(o0)

            p0 = Dense(64, activation='relu')(mlp5)
            p1 = Dense(16, activation='relu')(p0)
            p2 = Dense(4, activation='relu')(p1)
            p3 = Dense(1, activation='linear', name='predict_output')(p2)

            self.model = Model(inputs=i0, outputs=[o1,p3])
            self.model.compile(loss=['mean_squared_logarithmic_error','mean_squared_logarithmic_error'],
                          optimizer='adam', #adam #RMSprop #adadelta
                          metrics=['mse','mean_absolute_percentage_error'])


        else:
            self.import_model_from_file(path)

    def import_model_from_file(self, path):
        self.model = models.load_model(path)

    def train(self, train_stocks=None, batches = 10, epochs_per_batch=64, items_per_batch=128):
        if not train_stocks:
            train_stocks = self.train_stocks

        db = Database('stockdata.sqlite')
        giter = db.advanced_sample_generator(self.input_length, self.forecast_length, train_stocks, vectors=['new-norm close'])

        tb_callback = keras.callbacks.TensorBoard(log_dir='/tmp/sp-tb', write_graph=False)
        nan_callback = keras.callbacks.TerminateOnNaN()

        for b in range(batches):
            xs = []
            ys = []
            for i in range(items_per_batch):
                nx, ny = next(giter)
                xs.append(nx)
                ys.append([ny])

            xs = np.array(xs)
            ys = np.array(ys)

            self.model.fit(xs, [xs, ys], epochs=epochs_per_batch,
                           callbacks = [tb_callback, nan_callback])

            self.model.save(self.path)

    def eval(self, x):
        return self.model.predict(x)

    def plot(self, stocks=None, steps=1000):
        db = Database('stockdata.sqlite')
        #['MSFT','BGNE','SHOP','RF','MMM','EC','GMED','SQ','ETSY','BABA','VKTX','CCMP','ADBE','CBOE','ALGN','BLKB']
        #predict_stocks = ['INGN','WB','BIDU','AMTD','BA']
        giter = db.advanced_sample_generator(self.input_length, self.forecast_length, train_stocks, vectors=['new-norm close'])

        xs = []
        ys = []
        for i in range(steps):
            nx, ny = next(giter)
            xs.append(nx)
            ys.append([ny])

        xs = np.array(xs)
        ys = np.array(ys)

        ps = self.model.predict(xs)

        self.graph_one_one_error(ys.flatten(), ps[1].flatten(), self.name + '.png')

class ModelCavalierCactus(SingleOutputModel):
    '''
    CavalierCactus

    Mixed model using both unsupervised and supervised metrics.
    '''

    def __init__(self, path='CavalierCactus.hd5', ignore_existing=False,
                 train_stocks = ['^DJI','^GSPC']):

        self.name = 'CavalierCactus'
        self.path = path
        self.input_length = 400
        self.forecast_length = 20
        self.input_layers = 1
        self.x_shape = (self.input_layers ,self.input_length)
        self.train_stocks = None

        if ignore_existing or not os.path.isfile(path): # if file doesn't exist

            i0 = Input(shape=self.x_shape, name='input_layer')
            l0 = Reshape(self.x_shape[::-1])(i0)
            f0 = Flatten()(i0)

            block0 = MaxPooling1D()(Conv1D(32, 5, strides=1, activation='relu', padding='same', name='c0')(l0))
            block1 = MaxPooling1D()(Conv1D(32, 5, strides=1, activation='relu', padding='same', name='c1')(block0))
            block2 = MaxPooling1D()(Conv1D(32, 5, strides=1, activation='relu', padding='same', name='c2')(block1))

            bridge0 = concatenate([f0, Flatten()(block0)])
            mlp0 = Dense(1024, activation='relu')(bridge0)

            bridge1 = concatenate([mlp0, Flatten()(block1)])
            mlp1 = Dense(256, activation='relu')(bridge1)

            bridge2 = concatenate([mlp1, Flatten()(block2)])
            mlp2 = Dense(64, activation='relu')(bridge2)

            codec = block2
            codem = mlp2

            '''
            block3 = MaxPooling1D()(Conv1D(32, 5, dilation_rate=3, activation='relu', padding='causal', name='c3')(codec))
            block4 = MaxPooling1D()(Conv1D(32, 5, dilation_rate=3, activation='relu', padding='causal', name='c4')(block3))
            block5 = MaxPooling1D()(Conv1D(32, 5, dilation_rate=3, activation='relu', padding='causal', name='c5')(block4))
            print(self.x_shape)
            print([x.shape for x in [block0,block1,block2,block3,block4,block5]])
            '''

            us0 = Dense(256, activation='relu')(Flatten()(codec))
            us1 = Dense(2048, activation='relu')(us0)
            us2 = Dense(np.prod(self.x_shape), activation='relu')(us1)
            print(self.x_shape, np.prod(self.x_shape), us2.shape)

            o0 = Reshape(target_shape=self.x_shape, name='autoencoder_output')(us2)

            p0 = concatenate([codem, Flatten()(codec)])
            p1 = Dense(256, activation='relu')(p0)
            p2 = Dense(256, activation='relu')(p1)
            p3 = Dense(1, activation='linear')(p2)

            self.model = Model(inputs=i0, outputs=[o0,p3])
            self.model.compile(loss=['mean_squared_logarithmic_error','mean_squared_logarithmic_error'],
                          optimizer='adam', #adam #RMSprop #adadelta
                          metrics=['mse','mean_absolute_percentage_error'])


        else:
            self.import_model_from_file(path)

    def import_model_from_file(self, path):
        self.model = models.load_model(path)

    def train(self, train_stocks=None, batches = 10, epochs_per_batch=64, items_per_batch=128):


        db = Database('stockdata.sqlite')

        if not train_stocks:
            train_stocks = db.list_symbols()

        giter = db.advanced_sample_generator(self.input_length, self.forecast_length, train_stocks, vectors=['new-norm close'])

        tb_callback = keras.callbacks.TensorBoard(log_dir='/tmp/sp-tb', write_graph=False)
        nan_callback = keras.callbacks.TerminateOnNaN()

        for b in range(batches):
            xs = []
            ys = []
            for i in range(items_per_batch):
                nx, ny = next(giter)
                xs.append(nx)
                ys.append([ny])

            xs = np.array(xs)
            ys = np.array(ys)

            self.model.fit(xs, [xs, ys], epochs=epochs_per_batch,
                           callbacks = [tb_callback, nan_callback])

            self.model.save(self.path)

    def eval(self, x):
        return self.model.predict(x)

    def plot(self, stocks=None, steps=1000):
        db = Database('stockdata.sqlite')
        #['MSFT','BGNE','SHOP','RF','MMM','EC','GMED','SQ','ETSY','BABA','VKTX','CCMP','ADBE','CBOE','ALGN','BLKB']
        #predict_stocks = ['INGN','WB','BIDU','AMTD','BA']
        giter = db.advanced_sample_generator(self.input_length, self.forecast_length, stocks, vectors=['new-norm close'])

        xs = []
        ys = []
        for i in range(steps):
            nx, ny = next(giter)
            xs.append(nx)
            ys.append([ny])

        xs = np.array(xs)
        ys = np.array(ys)

        ps = self.model.predict(xs)

        self.graph_one_one_error(ys.flatten(), ps[1].flatten(), self.name + '.png')


class ModelDorkyDandelion(SingleOutputModel):
    '''
    DorkyDandelion

    An autoencoder that also attempts to encode the future.
    '''

    def __init__(self, path='models/DorkyDandelion.hd5', ignore_existing=False,
                 train_stocks = ['^DJI','^GSPC']):

        self.name = 'DorkyDandelion'
        self.save_path = path
        self.best_path = path + '.best'
        self.input_length = 100
        self.forecast_length = 10
        self.input_layers = 1
        self.output_layers = 1
        self.x_shape = (self.input_length + self.forecast_length,)
        self.y_shape = (self.input_length + self.forecast_length,)

        if ignore_existing or not os.path.isfile(self.save_path) \
            and not os.path.isfile(self.best_path): # if file doesn't exist

            i0 = Input(shape=self.x_shape, name='input_layer')
            bn = BatchNormalization()(i0)
            c0 = Concatenate()([i0, bn])
            f0 = c0#Flatten()(c0)

            ds0 = Dense(1000, activation='relu')(f0)
            ds1 = Dense(1000, activation='relu')(ds0)
            ds2 = Dense(1000, activation='relu')(ds1)
            ds3 = Dense(100, activation='relu')(ds2)
            ds4 = Dense(100, activation='relu')(ds3)
            ds5 = Dense(32, activation='relu')(ds4)

            codelayer = ds5

            us0 = Dense(100, activation='relu')(codelayer)
            us1 = Dense(100, activation='relu')(us0)
            us2 = Dense(1000, activation='relu')(us1)
            us3 = Dense(1000, activation='relu')(us2)
            us4 = Dense(1000, activation='relu')(us3)

            out0 = Dense(np.prod(self.y_shape), activation='relu')(us4)

            o0 = Reshape(target_shape=self.y_shape, name='autoencoder_output')(out0)

            self.model = Model(inputs=i0, outputs=o0)
            self.model.compile(loss='mean_absolute_percentage_error',
                          optimizer='adam', #adam #RMSprop #adadelta
                          metrics=['mse','mean_squared_logarithmic_error'])

        else:
            self.import_model_from_file(
                self.best_path if os.path.isfile(self.best_path)
                else self.save_path
                )

    def get_code_activations(self, xs):
        layers = self.model.layers
        w = [l.get_weights() for l in layers]

        i0 = Input(shape=self.x_shape, name='input_layer')
        bn = BatchNormalization()(i0)
        c0 = Concatenate()([i0, bn])
        f0 = c0#Flatten()(c0)

        n = 3
        ds0 = Dense(1000, activation='relu', weights=w[n])(f0)
        ds1 = Dense(1000, activation='relu', weights=w[n+1])(ds0)
        ds2 = Dense(1000, activation='relu', weights=w[n+2])(ds1)
        ds3 = Dense(100, activation='relu', weights = w[n+3])(ds2)
        ds4 = Dense(100, activation='relu', weights = w[n+4])(ds3)
        ds5 = Dense(32, activation='relu', weights=w[n+5])(ds4)

        codelayer = ds5

        code_model = Model(inputs=i0, outputs=codelayer)

        return code_model.predict(xs)

    def import_model_from_file(self, path):
        self.model = models.load_model(path)

    def train(self, epochs=10, loadfrom='dataset.npz'):
        d = sets_from_npz(loadfrom)

        xs = d['train_x']
        ys = d['train_y']

        tb_callback = keras.callbacks.TensorBoard(log_dir='/tmp/sp-tb', write_graph=False)
        nan_callback = keras.callbacks.TerminateOnNaN()
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            self.best_path, monitor='loss', save_best_only=True)


        self.model.fit(xs, ys, epochs=epochs,
                    callbacks = [tb_callback, nan_callback, checkpoint_callback],
                    validation_data = (d['test_x'],d['test_y']))

        self.model.save(self.save_path)

    def eval(self, x):
        return self.model.predict(x)

def sets_from_npz(path):
    d = np.load(path)

    sets = ['train','test','validation']

    out_dict = {}

    for s in sets:
        pairs = d[s]

        xs = []
        ys = []

        for p in pairs:
            xnew, ynew = p
            xs.append(xnew[0])
            ys.append(ynew[0])

        out_dict[s+'_x'] = np.array(xs)
        out_dict[s+'_y'] = np.array(ys)

    return out_dict




if __name__ == '__main__':
    m = ModelDorkyDandelion()

    m.train(epochs=10000, loadfrom='dataset.npz')

    '''
    # Code layer activation extraction demonstration

    d = sets_from_npz('dataset.npz')
    xs = d['train_x']
    ys = d['train_y']

    print(m.get_code_activations(xs[:10]))
    '''
