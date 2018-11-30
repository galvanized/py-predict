'''
NuclearNinja

    Normal 10k

    Test losses: [0.5653241596221924, 0.08713585588335991, 0.05836932975053787, 0.05097915539145469, 0.026168839924037457, 0.04537621736526489, 0.03318183604627847, 0.04628213608264923, 0.03274798749387264]
    Best losses: [0.5998387345671654, 0.09147641441598535, 0.05840622755885124, 0.04820761889219284, 0.023823161318898202, 0.04590229606628418, 0.030691411778330804, 0.038036927610635755, 0.02918039954826236]
    Best validation: [0.18878392878543962, 0.016901307048268623, 0.056339849227598375, 0.045558608800440756, 0.019933234509471414, 0.04427513605231172, 0.024956559025488056, 0.03566064302261535, 0.034224714076222774]

    "Trusted" 10k

    Test losses: [0.029671217530965806, 0.0020782391257584097, 0.023168500620126726, 0.02038313897848129, 0.009017764800786972, 0.022850150445103645, 0.008942136883735657, 0.019843418461084367, 0.014010740262269974]
    Best losses: [0.0312716003537178, 0.0021814295906573532, 0.023752483874559402, 0.02285811047554016, 0.008168225035071372, 0.025452969813346864, 0.008581066036224366, 0.020046039432287215, 0.014289962908625603]
    Best validation: [0.024447782768466247, 0.0017242083291773094, 0.02246514113172181, 0.02320069247956338, 0.007918391291094923, 0.025365078455020583, 0.008514379758549772, 0.020097630435432393, 0.013360454686458918]


'''
def version_name():
    return 'NuclearNinjaV3'


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

baseline = 1 # loss when certainty = 0

#https://datascience.stackexchange.com/questions/25029/custom-loss-function-with-additional-parameter-in-keras
def penalized_loss(noise):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) - K.abs(y_true - noise), axis=-1)
    return loss

def lin_sq_avg(a,b,c,lin,quad,pred_err=None,pred_lin=1,pred_quad=0):
    '''
    Linear, Square, Average utility function for error

    Args:
        a: first value (order unimportant)
        b: second value (order unimportant)
        c: certainty, [0,1]
        lin: linear coefficient
        quad: quadratic coefficient
        pred_err: predicted magnitude of linear error

    Returns:
        Linear-Quadratic hybrid loss value
    '''
    lin_e = baseline*(1-c)+abs(a-b)*c #baseline*(1-c)+(abs(a-b)-1/abs(a-b))*c
    quad_e = K.square(lin_e)
    sum_e = K.mean(lin_e*lin + quad_e*quad)

    if pred_err is not None:
        pred_err_e = K.abs(K.mean(lin_e) - pred_err)
        pred_sq_e = K.square(pred_err_e)
        sum_e += K.mean(pred_err_e*pred_lin) + K.sqrt(K.mean(pred_sq_e*pred_quad))


    return sum_e

def predicted_error_loss(full_pred_err, forecast_pred_err, endpoint_pred_err,
                         certainties):
    def loss(y_true, y_pred):
        c = certainties
        # normal loss
        err_norm = lin_sq_avg(y_pred, y_true, c, 0, 4, full_pred_err, 0, 0.5)

        # forecast loss
        err_forecast = lin_sq_avg(y_pred[-20:], y_true[-20:], c[-20:], 0, 5,
                                  forecast_pred_err, 0, .25)

        # endpoint loss
        err_endpoint = lin_sq_avg(y_pred[-1], y_true[-1], c[-1], 0.5, 10,
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


def data(dataset_path='../dataset-trusted-10k-300in-20out.npz'):
    d = np.load(dataset_path)

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

def recent_data(loadpath='../dataset-trusted-recent-300in-20out.npz'):
    d = np.load(loadpath)

    pairs = d['recent']
    xs = []

    o = {}

    for p in pairs:
        print(p[0])
        xnew, ynew = p
        xs.append(xnew)

    xs = np.array(xs)
    print(xs.shape)

    return xs, d['syms']


def model(x_train, y_train=None, x_test=None, y_test=None, x_val=None, y_val=None,
          load_existing=None, skip_train=False, load_recent=None, predict=False):
    name = version_name()
    save_path = name+'.hd5'
    best_path = name + '.best.hd5'
    input_length = 300
    forecast_length = 20
    input_layers = 1
    output_layers = 1
    x_shape = (input_length + forecast_length,)
    y_shape = (input_length + forecast_length,)

    if y_train is not None:

        yfill_train = np.array([[0]]*len(y_train))
        yfill_test = np.array([[0]]*len(y_test))
        if y_val is not None:
            yfill_val = np.array([[0]]*len(y_val))

        print(x_train.shape, y_train.shape, yfill_train.shape)

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

    do0 = Dropout(0.5)(f0)

    d0s = 512
    ds0 = Dense(d0s, activation='relu')(do0)

    do1 = Dropout(0.5)(ds0)

    d1s = 512
    ds1 = Dense(d1s, activation='relu')(do1)

    kernel_size = 5
    filters = 32
    do2 = Dropout(0)(ci)

    c0 = Conv1D(filters, kernel_size)(do2)
    c1 = Conv1D(filters//2, 3)(c0)

    # --- begin code ---

    do3 = Dropout(0.25)(ds1)

    codesize = 64

    densecode = Dense(codesize, activation='relu')(do0)

    pool_size = 5
    convcode0 = MaxPooling1D(pool_size)(c0)
    convcode1 = MaxPooling1D(pool_size)(c1)

    codelayer = Concatenate()([densecode, Flatten()(convcode0), Flatten()(convcode1)])


    # --- begin up ----

    do4 = Dropout(0.25)(codelayer)

    u0s = 256
    us0 = Dense(u0s, activation='relu')(do4)

    do5 = Dropout(0.25)(us0)

    u1s = 256
    us1 = Dense(u1s, activation='relu')(do5)

    do6 = Dropout(0.25)(us1)

    last_layer = do6

    # --- begin output ---

    o0act = 'linear'

    out0 = Dense(np.prod(y_shape), activation=o0act)(last_layer)
    cert = Dense(np.prod(y_shape), activation='tanh')(last_layer)

    oerract = 'linear'

    o1 = Dense(1, activation=oerract, name='o1')(last_layer)
    o2 = Dense(1, activation=oerract, name='o2')(last_layer)
    o3 = Dense(1, activation=oerract, name='o3')(last_layer)

    o0 = Reshape(target_shape=y_shape, name='autoencoder_output1')(out0)
    certout = Reshape(target_shape=y_shape, name='autoencoder_cert')(cert)

    model = Model(inputs=i0, outputs=[o0,o1,o2,o3,certout])

    model.compile(loss=predicted_error_loss(o1,o2,o3,certout),
                  optimizer='adadelta',
                  metrics=['mse',linear_err])

    tb_callback = keras.callbacks.TensorBoard(log_dir='/tmp/sp-tb', write_graph=False)
    nan_callback = keras.callbacks.TerminateOnNaN()
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
    best_path, monitor='loss', save_best_only=True)

    if load_existing:
        print('Loading model from {} .'.format(load_existing))
        model.load_weights(load_existing)

    if not skip_train:

        print('FITTING')

        model.fit([x_train], [y_train,yfill_train,yfill_train,yfill_train,yfill_train], epochs=1000,
            callbacks = [tb_callback, nan_callback, checkpoint_callback],
            validation_data = (x_test, [y_test,yfill_test,yfill_test,yfill_test,yfill_test]))

        model.save(save_path)

    if not load_recent and y_train is not None:

        print('EVALUATING')

        score = model.evaluate(x_test, [y_test,yfill_test,yfill_test,yfill_test,yfill_test])
        loss = score[0]

        print('Test losses:', score)
        model.load_weights(best_path)

        print('Best losses:', model.evaluate(x_test, [y_test,yfill_test,yfill_test,yfill_test,yfill_test]))

        if x_val is not None and y_val is not None:
            print('Best validation:', model.evaluate(x_val, [y_val,yfill_val,yfill_val,yfill_val,yfill_val]))


    if load_recent:
        print('FORECASTING FROM RECENT')
        recent_xs, recent_syms = recent_data(load_recent)

        print("Recent xs shape:")
        print(recent_xs.shape)

        p0, p1, p2, p3 = model.predict(recent_xs)

        for i in range(len(p0)):
            print(recent_syms[i], p0[i][-20:], p1[i], p2[i], p3[i])

    if predict:
        print('Predicting...')
        p0, p1, p2, p3, c = model.predict(x_train)
        return [p0, p1, p2, p3, c]


if __name__ == '__main__':
    mode = 'train'
    if mode is 'train':
        model(*data('../10k-300in-20out.npz'))
    if mode is 'evaluate':
        model(*data('../10k-300in-20out.npz'),load_existing=version_name()+'.best.hd5',
              skip_train=True)
    elif mode is 'continue':
        model(*data('../10k-300in-20out.npz'),load_existing=version_name()+'.best.hd5')
    elif mode is 'recent':
        model(*data('../10k-300in-20out.npz'),load_existing=version_name()+'.best.hd5',
              skip_train=True,load_recent='../dataset-recent-300in-20out.npz')
