import ScratchLSTM
import ScratchEmbedding
import ScratchDense
import ScratchRNN
import numpy as np
function_dict ={
    "relu" : lambda x: np.maximum(0, x),
    "softmax" : lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True),
    "tanh" : lambda x: np.tanh(x)
}

class Model:
    def __init__(self, layers=[], name="Model"):
        self.layers = layers
        self.name = name

    def predict(self, x):
        print(f"{self.name} Forward propagation started with input shape: {x.shape}")
        for i,layer in enumerate(self.layers):
            x = layer.forward(x)
        return x

def convertModel(model, model_name="Model"):
    activated = False
    LSTM_layer_flags = {}
    RNN_layer_flags = {}
    for i, layer in enumerate(model.layers):
        if layer.name.lower().startswith("lstm"):
            activated = True
            i += 1
            while(model.layers[i].name.lower().startswith("lstm")):
                LSTM_layer_flags[i-1] = True
                i += 1
            if activated:
                LSTM_layer_flags[i-1] = False
                activated = False
        if layer.name.lower().startswith("simple_rnn"):
            activated = True
            i += 1
            while(model.layers[i].name.lower().startswith("simple_rnn")):
                RNN_layer_flags[i-1] = True
                i += 1
            if activated:
                RNN_layer_flags[i-1] = False
                activated = False 
    layers = []
    for j, layer in enumerate(model.layers):
        if layer.name.lower().startswith("lstm"):
            print(f"{j} Lstm layer found")
            layers.append(ScratchLSTM.ScratchLSTM(
                hidden_size=layer.units,
                lstm_weights=layer.get_weights(),
                manytomany=LSTM_layer_flags.get(j)
            ))
        if layer.name.lower().startswith("dense"):
            print(f"{j} Dense layer found")
            layers.append(ScratchDense.DenseScratch(
                dense_weights=layer.get_weights(),
                activation=function_dict.get(layer.activation.__name__.lower())
            ))
        if layer.name.lower().startswith("embedding"):
            print(f"{j} Embedding layer found")
            layers.append(ScratchEmbedding.ScratchEmbedding(
                embedding_weight=layer.get_weights()[0]
            ))
        if layer.name.lower().startswith("simple_rnn"):
            print(f"{j} RNN layer found")
            layers.append(ScratchRNN.ScratchRNN(
                weight=layer.get_weights(),
                activation=function_dict.get(layer.activation.__name__.lower()),
                manytomany=RNN_layer_flags.get(j)
            ))
    return Model(layers=layers, name=model_name)