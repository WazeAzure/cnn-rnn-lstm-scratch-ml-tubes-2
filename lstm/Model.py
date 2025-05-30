import ScratchLSTM
import ScratchEmbedding
import ScratchDense
import ScratchRNN
import ScratchBiLSTM
import ScratchBiRNN
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

        if layer.name.lower().startswith("bidirectional"):
            activated = True
            if layer.forward_layer.name.lower().startswith("forward_lstm"):
                i += 1
                while(model.layers[i].name.lower().startswith("bidirectional")):
                    LSTM_layer_flags[i-1] = True
                    i += 1
                if activated:
                    LSTM_layer_flags[i-1] = False
                    activated = False 
            if layer.forward_layer.name.lower().startswith("forward_simple_rnn"):
                i += 1
                while(model.layers[i].name.lower().startswith("bidirectional")):
                    RNN_layer_flags[i-1] = True
                    i += 1
                if activated:
                    RNN_layer_flags[i-1] = False
                    activated = False 
    print(f"LSTM Layer flags: {LSTM_layer_flags}")
    print(f"RNN Layer flags: {LSTM_layer_flags}")
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
        if layer.name.lower().startswith("bidirectional"):
            if layer.forward_layer.name.lower().startswith("forward_lstm"):
                print(f"{j} BiLSTM layer found")
                layers.append(ScratchBiLSTM.ScratchBiLSTM(
                    hidden_size=layer.forward_layer.units,
                    lstm_weights_fwd=layer.forward_layer.get_weights(),
                    lstm_weights_bwd=layer.backward_layer.get_weights(),
                    manytomany=LSTM_layer_flags.get(j)
                ))
            if layer.forward_layer.name.lower().startswith("forward_simple_rnn"):
                print(f"{j} BiRNN layer found")
                layers.append(ScratchBiRNN.ScratchBiRNN(
                    fwd_rnn_weights=layer.forward_layer.get_weights(),
                    bwd_rnn_weights=layer.backward_layer.get_weights(),
                    manytomany=RNN_layer_flags.get(j),
                    activation=function_dict.get(layer.forward_layer.activation.__name__.lower())
                ))
    return Model(layers=layers, name=model_name)