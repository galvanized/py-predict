'''

FeistyFerret



'''
def version_name():
    return 'FeistyFerret'
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

#https://datascience.stackexchange.com/questions/25029/custom-loss-function-with-additional-parameter-in-keras
def penalized_loss(noise):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) - K.abs(y_true - noise), axis=-1)
    return loss

def predicted_error_loss(pred_err):
    def loss(y_true, y_pred):
        # normal loss
        full_mse = K.square(y_pred - y_true)
        mse_pow = 1.5
        # linear loss
        full_lin = K.abs(y_pred - y_true)
        lin_pow = 0.5
        # predicted error loss
        pred_err_loss = K.abs(full_lin - pred_err)
        pred_err_pow = 1

        mse_mean = K.mean(full_mse, axis=-1)
        lin_mean = K.mean(full_lin, axis=-1)
        pred_err_mean = K.mean(pred_err_loss)

        return mse_mean**mse_pow + lin_mean**lin_pow + pred_err_mean**pred_err_mean
    return loss



def pred_err_only(pred_err):
    def loss(y_true, y_pred):
        # linear loss
        full_lin = K.abs(y_pred - y_true)
        # predicted error loss
        pred_err_loss = K.abs(full_lin - pred_err)

        pred_err_mean = K.mean(pred_err_loss)

        return pred_err_mean
    return loss

def get_pred_err(pred_err):
    # not a loss! use as a metric for diagnostic purposes only
    def loss(y_true, y_pred):

        pred_err_mean = K.mean(pred_err)

        return pred_err_mean
    return loss

def linear_err(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


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
    name = version_name()
    save_path = 'models/'+name+'.hd5'
    best_path = save_path + '.best'
    input_length = 100
    forecast_length = 10
    input_layers = 1
    output_layers = 1
    x_shape = (input_length + forecast_length,)
    y_shape = (input_length + forecast_length,)

    i0 = Input(shape=x_shape, name='input_layer')
    bn = BatchNormalization()(i0)
    c0 = Concatenate()([i0, bn])
    f0 = c0

    ds0 = Dense(1000, activation='relu')(f0)
    ds1 = Dense(1000, activation='relu')(ds0)
    ds2 = Dense(1000, activation='relu')(ds1)
    ds3 = Dense(512, activation='relu')(ds2) #512
    ds4 = Dense(512, activation='relu')(ds3) #512

    do0 = Dropout(0.5)(ds4) #0.517

    codelayer = Dense({{choice([512,256,128,64,32,16])}}, activation='relu')(do0) #64

    us0 = Dense(128, activation='relu')(codelayer) #128
    us1 = Dense(128, activation='relu')(us0) #128
    us2 = Dense(1000, activation='relu')(us1)
    us3 = Dense(1000, activation='relu')(us2)
    us4 = Dense(1000, activation='relu')(us3)

    out0 = Dense(np.prod(y_shape), activation='relu')(us4)

    e0 = Dense(64, activation='relu')(us2)
    e1 = Dense(4, activation='relu')(e0)
    o1 = Dense(1, activation='linear')(e1)

    o0 = Reshape(target_shape=y_shape, name='autoencoder_output')(out0)

    model = Model(inputs=i0, outputs=[o0])
    model.compile(loss=predicted_error_loss(o1),
                  optimizer='adam',
                  metrics=['mse',linear_err,
                           get_pred_err(o1),
                           pred_err_only(o1)])

    tb_callback = keras.callbacks.TensorBoard(log_dir='/tmp/sp-tb', write_graph=False)
    nan_callback = keras.callbacks.TerminateOnNaN()
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
    best_path, monitor='loss', save_best_only=True)

    print('FITTING')

    model.fit(x_train, y_train, epochs=3,
        callbacks = [tb_callback, nan_callback],
        validation_data = (x_test, y_test))

    print('EVALUATING')

    score = model.evaluate(x_test, y_test)
    loss = score[0]



    print('Test losses:', score)
    return {'loss': loss, 'status': STATUS_OK, 'model': model}


def optimize():
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          functions=[predicted_error_loss,
                                                     version_name,
                                                     pred_err_only,
                                                     get_pred_err,
                                                     linear_err],
                                          algo=tpe.suggest,
                                          max_evals=30,
                                          trials=Trials())
    print(best_run)


if __name__ == '__main__':
    optimize()
