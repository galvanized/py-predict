import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
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

from hyperopt import Trials, STATUS_OK, tpe, rand
from hyperas import optim
from hyperas.distributions import choice, uniform

import os.path


def data():
    d = np.load('../dataset.npz')

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

    d = out_dict

    x_train = d['train_x']
    y_train = d['train_y']
    x_test = d['test_x']
    y_test = d['test_y']

    return x_train, y_train, x_test, y_test


def model(x_train, y_train, x_test, y_test):
    name = 'ElasticElk'
    save_path = 'models/ElasticElk.hd5'
    best_path = 'models/ElasticElk.hd5' + '.best'
    input_length = 100
    forecast_length = 10
    input_layers = 1
    output_layers = 1
    x_shape = (input_length + forecast_length,)
    y_shape = (input_length + forecast_length,)

    i0 = Input(shape=x_shape, name='input_layer')
    bn = BatchNormalization()(i0)
    c0 = Concatenate()([i0, bn])
    f0 = c0#Flatten()(c0)

    ds0 = Dense(1000, activation='relu')(f0)
    ds1 = Dense(1000, activation='relu')(ds0)
    ds2 = Dense(1000, activation='relu')(ds1)
    ds3 = Dense({{choice([512,256,128])}}, activation='relu')(ds2)
    ds4 = Dense({{choice([512,256,128])}}, activation='relu')(ds3)
    ds5 = Dense({{choice([512,256,128,64,32,16])}}, activation='relu')(ds4)

    codelayer = ds5

    us0 = Dense({{choice([512,256,128])}}, activation='relu')(codelayer)
    us1 = Dense({{choice([512,256,128])}}, activation='relu')(us0)
    us2 = Dense(1000, activation='relu')(us1)
    us3 = Dense(1000, activation='relu')(us2)
    us4 = Dense(1000, activation='relu')(us3)

    out0 = Dense(np.prod(y_shape), activation='relu')(us4)

    o0 = Reshape(target_shape=y_shape, name='autoencoder_output')(out0)

    model = Model(inputs=i0, outputs=o0)
    model.compile(loss='mean_absolute_percentage_error',
                  optimizer='adam', #adam #RMSprop #adadelta
                  metrics=['mse','mean_squared_logarithmic_error'])

    tb_callback = keras.callbacks.TensorBoard(log_dir='/tmp/sp-tb', write_graph=False)
    nan_callback = keras.callbacks.TerminateOnNaN()
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
    best_path, monitor='loss', save_best_only=True)

    print('FITTING')

    model.fit(x_train, y_train, epochs=1,
        callbacks = [tb_callback, nan_callback],
        validation_data = (x_test, y_test))

    print('EVALUATING')

    score, mse, msle = model.evaluate(x_train, y_train)

    print('Test score:', score)
    print('mse {},  msle {}'.format(mse, msle))
    return {'loss': msle, 'status': STATUS_OK, 'model': model}


def optimize():
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    print(best_run)


if __name__ == '__main__':
    optimize()
