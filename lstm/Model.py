class Model:
    def __init__(self, layers=[]):
        self.layers = layers
        
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x