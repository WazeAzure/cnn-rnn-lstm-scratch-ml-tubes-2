import ScratchLSTM
import ScratchEmbedding
import ScratchDense
import numpy as np
function_dict ={
    "relu" : lambda x: np.maximum(0, x),
    "softmax" : lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
}

class Model:
    def __init__(self, layers=[]):
        self.layers = layers
        
    def predict(self, x):
        for i,layer in enumerate(self.layers):
            print(i, x.shape)
            x = layer.forward(x)
        return x

def convertModel(model):
    activated = False
    LSTM_layer_flags = {}
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
    print(LSTM_layer_flags)
    layers = []
    for j, layer in enumerate(model.layers):
        if layer.name.lower().startswith("lstm"):
            print(f"{j} Lstm layer found")
            print(layer.units)
            print(LSTM_layer_flags.get(j))
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
            print(f"Embedding weight shape: {layer.get_weights()[0].shape}")
    return Model(layers=layers)