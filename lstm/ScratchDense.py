import numpy as np

class DenseScratch:
    def __init__(self, dense_weights: np.ndarray, activation):
        print(len(dense_weights))
        self.dense_weights, self.dense_biases = dense_weights
        self.activation = activation
        print(f"DenseScratch initialized with weights shape: {len(self.dense_weights)}")

    def forward(self, x: np.ndarray):
        return self.activation(x @ self.dense_weights + self.dense_biases)
    
    def getShape(self):
        return self.dense_weights.shape