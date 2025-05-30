import numpy as np

class ScratchRNN:
    def __init__ (self, weight, activation=None, manytomany=False):
        self.Wx, self.Wh, self.b = weight
        self.b = self.b.reshape(1, -1)
        self.hidden_size = self.Wh.shape[0]
        self.activation = activation if activation is not None else np.tanh
        self.manytomany = manytomany

    def forward(self, x):
        if len(x.shape) == 2:
            x = x[np.newaxis, :, :]

        sequence_length = x.shape[1]
        outputs = []

        self.h = np.zeros((x.shape[0], self.hidden_size))

        for t in range(sequence_length):
            x_t = x[:, t, :]
            self.h = self.activation(np.matmul(x_t, self.Wx) + np.matmul(self.h, self.Wh) + self.b)
            outputs.append(self.h)

        if self.manytomany:
            return np.stack(outputs, axis=1)
        else:
            return self.h