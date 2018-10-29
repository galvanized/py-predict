'''
NuclearNinja

    low dropout version of mysticalmagnesium

    Test losses: [0.5653241596221924, 0.08713585588335991, 0.05836932975053787, 0.05097915539145469, 0.026168839924037457, 0.04537621736526489, 0.03318183604627847, 0.04628213608264923, 0.03274798749387264]
    Best losses: [0.5998387345671654, 0.09147641441598535, 0.05840622755885124, 0.04820761889219284, 0.023823161318898202, 0.04590229606628418, 0.030691411778330804, 0.038036927610635755, 0.02918039954826236]
    Best validation: [0.18878392878543962, 0.016901307048268623, 0.056339849227598375, 0.045558608800440756, 0.019933234509471414, 0.04427513605231172, 0.024956559025488056, 0.03566064302261535, 0.034224714076222774]


'''
def version_name():
    return 'NuclearNinja'


import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from main import *
from stocks import *

import numpy as np
from math import *

import keras
from keras import models
from keras.models import Sequential, load_model
from keras.layers import *
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model

from hyperopt import Trials, STATUS_OK, tpe, rand
from hyperas import optim
from hyperas.distributions import choice, uniform

import os.path

#https://datascience.stackexchange.com/questions/25029/custom-loss-function-with-additional-parameter-in-keras
def penalized_loss(noise):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) - K.abs(y_true - noise), axis=-1)
    return loss

def lin_sq_avg(a,b,lin,quad,pred_err=None,pred_lin=1,pred_quad=0):
    '''
    Linear, Square, Average utility function for error

    Args:
        a: first value (order unimportant)
        b: second value (order unimportant)
        lin: linear coefficient
        quad: quadratic coefficient
        pred_err: predicted magnitude of linear error

    Returns:
        Linear-Quadratic hybrid loss value
    '''
    lin_e = K.abs(a-b)
    quad_e = K.square(lin_e)
    sum_e = K.mean(lin_e*lin + quad_e*quad)

    if pred_err is not None:
        pred_err_e = K.abs(K.mean(lin_e) - pred_err)
        pred_sq_e = K.square(pred_err_e)
        sum_e += K.mean(pred_err_e*pred_lin + pred_sq_e*pred_quad)


    return sum_e

def predicted_error_loss(full_pred_err, forecast_pred_err, endpoint_pred_err):
    def loss(y_true, y_pred):
        # normal loss
        err_norm = lin_sq_avg(y_pred, y_true, 0, 4, full_pred_err, 0, 0.5)

        # forecast loss
        err_forecast = lin_sq_avg(y_pred[-20:], y_true[-20:], 0, 1,
                                  forecast_pred_err, 0, .25)

        # endpoint loss
        err_endpoint = lin_sq_avg(y_pred[-1], y_true[-1], 0.5, 5,
                                  endpoint_pred_err, 0, 3)

        return err_norm + err_forecast + err_endpoint

    return loss


def get_pred_err(pred_err):
    # not a loss! use as a metric for diagnostic purposes only
    def value(y_true, y_pred):

        pred_err_mean = K.mean(pred_err)

        return pred_err_mean
    return value

def linear_err(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def get_pred_err_loss(pred_err,pos):
    def pred_err_loss(y_true, y_pred):
        if pos == 1:
            # normal loss
            return lin_sq_avg(y_pred, y_true, 0, 0, pred_err, 1, 0)
        elif pos == 2:
            # forecast loss
            return lin_sq_avg(y_pred[-20:], y_true[-20:], 0, 0,
                                      pred_err, 1, 0)
        else:
            # endpoint loss
            return lin_sq_avg(y_pred[-1], y_true[-1], 0, 0,
                                      pred_err, 1, 0)
    return pred_err_loss


def data():
    d = np.load('../dataset1k-300in-20out.npz')

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
    x_validation = d['validation_x']
    y_validation = d['validation_y']

    return x_train, y_train, x_test, y_test, x_validation, y_validation


def model(x_train, y_train, x_test, y_test, x_val=None, y_val=None):
    name = version_name()
    save_path = name+'.hd5'
    best_path = name + '.best.hd5'
    input_length = 300
    forecast_length = 20
    input_layers = 1
    output_layers = 1
    x_shape = (input_length + forecast_length,)
    y_shape = (input_length + forecast_length,)

    i0 = Input(shape=x_shape, name='input_layer')
    f0type = 2

    if f0type==0:
        f0 = i0
    elif f0type==1:
        f0 = BatchNormalization()(i0)
    else:
        f0 = Concatenate()([i0, BatchNormalization()(i0)])

    ci = Reshape((x_shape+(1,)))(i0)

    # --- begin down ---

    do0 = Dropout(0.1)(f0)

    d0s = 512
    ds0 = Dense(d0s, activation='relu')(do0)

    do1 = Dropout(0.1)(ds0)

    d1s = 512
    ds1 = Dense(d1s, activation='relu')(do1)

    kernel_size = 5
    filters = 32
    do2 = Dropout(0)(ci)

    c0 = Conv1D(filters, kernel_size)(do2)

    # --- begin code ---

    do3 = Dropout(0.1)(ds1)

    codesize = 64

    densecode = Dense(codesize, activation='relu')(do0)

    pool_size = 5
    convcode = MaxPooling1D(pool_size)(c0)

    codelayer = Concatenate()([densecode, Flatten()(convcode)])





    # --- begin up ----

    do4 = Dropout(0.1)(codelayer)

    u0s = 256
    us0 = Dense(u0s, activation='relu')(do4)

    do5 = Dropout(0.1)(us0)

    u1s = 256
    us1 = Dense(u1s, activation='relu')(do5)

    do6 = Dropout(0.1)(us1)

    last_layer = do6

    # --- begin output ---

    o0act = 'linear'

    out0 = Dense(np.prod(y_shape), activation=o0act)(last_layer)

    oerract = 'linear'

    o1 = Dense(1, activation=oerract)(last_layer)
    o2 = Dense(1, activation=oerract)(last_layer)
    o3 = Dense(1, activation=oerract)(last_layer)

    o0 = Reshape(target_shape=y_shape, name='autoencoder_output')(out0)

    model = Model(inputs=i0, outputs=[o0])
    model.compile(loss=predicted_error_loss(o1,o2,o3),
                  optimizer='adadelta',
                  metrics=['mse',linear_err,
                           get_pred_err(o1),
                           get_pred_err_loss(o1, 1),
                           get_pred_err(o2),
                           get_pred_err_loss(o2, 2),
                           get_pred_err(o3),
                           get_pred_err_loss(o1, 3)])

    tb_callback = keras.callbacks.TensorBoard(log_dir='/tmp/sp-tb', write_graph=False)
    nan_callback = keras.callbacks.TerminateOnNaN()
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
    best_path, monitor='loss', save_best_only=True)

    print('FITTING')

    model.fit(x_train, y_train, epochs=2000,
        callbacks = [tb_callback, nan_callback, checkpoint_callback],
        validation_data = (x_test, y_test))

    model.save(save_path)

    print('EVALUATING')

    score = model.evaluate(x_test, y_test)
    loss = score[0]




    print('Test losses:', score)
    model.load_weights(best_path)

    print('Best losses:', model.evaluate(x_test, y_test))

    if x_val is not None and y_val is not None:
        print('Best validation:', model.evaluate(x_val, y_val))

    return {'loss': loss, 'status': STATUS_OK}


if __name__ == '__main__':
    model(*data())
