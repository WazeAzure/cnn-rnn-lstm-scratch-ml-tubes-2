import ScratchRNN
import numpy as np

class ScratchBiRNN:
    def __init__(self, fwd_rnn_weights, bwd_rnn_weights, batch_size=1, manytomany=False, activation=None):
        self.fwd_rnn = ScratchRNN.ScratchRNN(fwd_rnn_weights, manytomany=manytomany, activation=activation)
        self.bwd_rnn = ScratchRNN.ScratchRNN(bwd_rnn_weights, manytomany=manytomany, activation=activation)
        self.manytomany = manytomany
    
    def forward(self, x: np.ndarray):
        if len(x.shape) == 2:
            x = x[np.newaxis, :, :]

        # Forward RNN
        output_fwd = self.fwd_rnn.forward(x)

        # Backward RNN
        x_reversed = x[:, ::-1, :]
        output_bwd = self.bwd_rnn.forward(x_reversed)

        if self.manytomany:
            output_bwd = output_bwd[:, ::-1, :]
            return np.concatenate([output_fwd, output_bwd], axis=2)
        else:
            return np.concatenate([output_fwd, output_bwd], axis=1)