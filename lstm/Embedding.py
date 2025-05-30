import numpy as np

class ScratchEmbedding:
    def __init__(self, embedding_weight: np.ndarray, batch_size=1):
        self.embedding_weight = embedding_weight.copy()

    def forward(self, x):
        return self.embedding_weight[x]