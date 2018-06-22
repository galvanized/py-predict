import keras.backend as K
from keras import models
import numpy as np

# https://github.com/philipperemy/keras-visualize-activations/blob/master/read_activations.py

def get_activations(model, model_inputs):
    return get_activations_philipperemy(model, model_inputs)

def get_activations_philipperemy(model, model_inputs, print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    print(funcs)

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations

def weights_from_model(model):
    layers = model.layers
    weights = []
    for l in layers:
        try:
            w = l.get_weights()[0]
            #print(w)
            weights.append(w)
        except:
            print('Cannot get weights for', l)
            weights.append([])

    return weights

if __name__ == '__main__':
    m = models.load_model('models/DorkyDandelion.hd5')
    '''
    train_pairs = np.load('dataset100k.npz')['train']

    xs = []
    ys = []

    for p in train_pairs:
        xs, ys = p
    '''

    print ([len(w) for w in weights_from_model(m)])
    #print(get_activations(m, xs[0]))
