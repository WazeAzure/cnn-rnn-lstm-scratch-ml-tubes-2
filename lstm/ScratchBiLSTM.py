import ScratchLSTM
import numpy as np

class ScratchBiLSTM:
    def __init__(self, hidden_size, lstm_weights_fwd, lstm_weights_bwd, batch_size=1, manytomany=False):
        self.fwd_lstm = ScratchLSTM.ScratchLSTM(hidden_size, lstm_weights_fwd, batch_size, manytomany)
        self.bwd_lstm = ScratchLSTM.ScratchLSTM(hidden_size, lstm_weights_bwd, batch_size, manytomany)
        self.manytomany = manytomany

    def forward(self, x: np.ndarray):
        if len(x.shape) == 2:
            x = x[np.newaxis, :, :]

        # Forward LSTM
        output_fwd = self.fwd_lstm.forward(x)

        # Backward LSTM
        x_reversed = x[:, ::-1, :]
        output_bwd = self.bwd_lstm.forward(x_reversed)

        if self.manytomany:
            # Reverse backward output to align with forward
            output_bwd = output_bwd[:, ::-1, :]
            return np.concatenate([output_fwd, output_bwd], axis=2)
        else:
            return np.concatenate([output_fwd, output_bwd], axis=1)