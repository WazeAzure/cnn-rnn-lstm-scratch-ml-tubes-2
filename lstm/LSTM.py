import numpy as np

class ScratchLSTM:
    def __init__(self,
                hidden_size,
                lstm_weights: np.ndarray,
                batch_size=1,
                manytomany=False
               ):
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.manytomany = manytomany

        self.h = np.zeros((batch_size, hidden_size))
        self.c = np.zeros((batch_size, hidden_size))

        W = lstm_weights[0]
        U = lstm_weights[1]
        b = lstm_weights[2]

        self.W_i, self.W_f, self.W_c, self.W_o = np.split(W, 4, axis=1)
        self.U_i, self.U_f, self.U_c, self.U_o = np.split(U, 4, axis=1)
        self.b_i, self.b_f, self.b_c, self.b_o = np.split(b, 4)

        self.result = np.ndarray([])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x: np.ndarray):
        if len(x.shape) == 2:
            x = x[np.newaxis, :, :]

        sequence_length = x.shape[1]
        outputs = []

        for t in range(sequence_length):
            x_t = x[:, t, :]

            # Compute Gates
            i_t = self.sigmoid(np.dot(x_t, self.W_i) + np.dot(self.h, self.U_i) + self.b_i)
            f_t = self.sigmoid(np.dot(x_t, self.W_f) + np.dot(self.h, self.U_f) + self.b_f)
            g_t = np.tanh(np.dot(x_t, self.W_c) + np.dot(self.h, self.U_c) + self.b_c)
            o_t = self.sigmoid(np.dot(x_t, self.W_o) + np.dot(self.h, self.U_o) + self.b_o)

            # Update cell state and hidden state
            self.c = f_t * self.c + i_t * g_t
            self.h = o_t * np.tanh(self.c)

            outputs.append(self.h)

        if self.manytomany:
            return np.stack(outputs, axis=1)  # Many to many output
        else:
            return self.h  # Single output