import keras
from keras.models import Sequential
from keras.layers import *
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model

import numpy as np

from stocks import Database
from analysis import multipliers



class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class MaxNorm(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


input_length = 400
model_length = input_length #- 1 # subtract by 1 if using diffs

dropout_level = 0.8

i0 = inputs = Input(shape=(model_length,))
l0 = Reshape((model_length,1))(i0)
#l1 = Flatten()(i0)

ff0 = Dense(128, activation='relu')(i0)
ff1 = Dense(128, activation='relu')(i0)

ca0 = Conv1D(32, 8, activation='relu')(l0)
ca1 = Conv1D(32, 8, activation='relu')(ca0)
ca2 = Conv1D(32, 8, activation='relu')(ca1)
ca3 = MaxPooling1D()(ca2)

cb0 = Conv1D(32, 16, activation='relu')(l0)
cb1 = Conv1D(32, 16, activation='relu')(cb0)
cb2 = MaxPooling1D()(cb1)

cc0 = Conv1D(32, 32, activation='relu')(l0)
cc1 = MaxPooling1D()(cc0)

ct = concatenate([Dropout(dropout_level)(Flatten()(x)) for x in \
    [ca3, cb2, cc1, ca0, ca1, ca2, cb0, cb1, cc0] ]+ [ff0])

mlp0 = Dense(128, activation='relu')(ct)
mlp1 = Dense(128, activation='relu')(Dropout(dropout_level)(mlp0))
mlp2 = Dense(32, activation='relu')(Dropout(dropout_level)(mlp1))

mlp3 = concatenate([mlp2, ff1])

mlp4 = Dense(8, activation='linear')(mlp3)

'''
a0 = Dense(256, activation='relu')(l1)
'''
'''
c0 = Conv1D(32, 5, activation='relu')(l0)
c1 = Conv1D(32, 3, activation='relu')(c0)
c2 = MaxPooling1D()(c1)
c3 = Flatten()(c2)
'''
'''
r1 = LSTM(128, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)(l0)
'''
'''
b0 = concatenate([a0, c3, r1])
b1 = Dense(64, activation='relu')(b0)
b1 = Dense(64, activation='relu')(b1)
b2 = Dense(8, activation='linear')(b1)'''

'''
lin0 = Dense(128, activation='linear')(i0)
lin1 = Dense(32, activation='linear')(lin0)
lin2 = Dense(8, activation='linear')(lin1)
'''

o0 = Dense(1, activation='linear')(mlp4)

print("Conv + MLP")
model = Model(inputs=i0, outputs=o0)
model.compile(loss='mean_squared_logarithmic_error',
              optimizer='adadelta', #adam #RMSprop
              metrics=['mse','mean_absolute_percentage_error'])

'''
Typical loss and metrics:

Linear MLP: [0.015278755454346538, 9.866830706596375] r = 0.08
Pure LSTM: [0.0367, 12.4759] r = 0.06
Pure 1d Conv: [0.03380762576125562, 9.723406195640564] r = 0.11
'''


db = Database('stockdata.sqlite')
#data_samples = 2**13
#print("Generating dataset. {} samples.".format(data_samples))
sample_stocks = None#['^DJI','^GSPC']#None #['GOOG','GM','KRO','^DJI','EC','ANY','AMD','MMM','^GSPC']
giter = db.sample_generator(input_length, 45, sample_stocks)
#graw = [next(giter) for x in range(data_samples)]
#g = []
#for i in graw:
    #g.append([multipliers(i[0]),i[1]])

tb_callback = keras.callbacks.TensorBoard(log_dir='/tmp/sp-tb', write_graph=False)
nan_callback = keras.callbacks.TerminateOnNaN()

batches = 64
items_per_batch = 256
iterations_per_batch = 128


for i in range(batches):
    print('Generating batch',i+1,'of',batches)
    x_batch = []
    y_batch = []
    for b in range(items_per_batch):
        x, y = next(giter)
        #x, y = g[b]
        x_batch.append(x)
        y_batch.append([y])
    x_batch = np.array(x_batch)#np.array([multipliers(x) for x in x_batch])
    y_batch = np.array(y_batch)

    print(model.evaluate(np.array(x_batch),np.array(y_batch)))
    print('Training.')
    model.fit(np.array(x_batch), np.array(y_batch),
              epochs=iterations_per_batch, callbacks=[tb_callback])

    model.save('everything.h5')
    #for t in range(iterations_per_batch):
    #    model.train_on_batch()


print("Generating evaluation dataset.")
x_batch = []
y_batch = []
for b in range(1024):
    if not b%(1024//32):
        print(b, 'of 1024')
    x, y = next(giter)
    #x, y = g[b]
    x_batch.append(x)
    y_batch.append([y])
x_batch = x_batch#[multipliers(x) for x in x_batch]

forecasts = model.predict(np.array(x_batch), batch_size=128)

csv_out = 'predicted,real\n'
for i in range(len(x_batch)):
    csv_out += ('{},{}\n'.format(forecasts[i][0],y_batch[i][0]))
    print('predicted',forecasts[i][0],', actually',y_batch[i][0])
with open('resultsneural.csv','w') as of:
    of.write(csv_out)

print('correlation', np.corrcoef((
    [forecasts[i][0] for i in range(len(forecasts))],
    [y_batch[i][0] for i in range(len(forecasts))]
    ))[0][1]
    )
