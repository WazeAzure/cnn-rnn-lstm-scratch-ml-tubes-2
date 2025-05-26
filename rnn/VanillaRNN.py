from ActivationFunction import ActivationFunctions

import matplotlib.pyplot as plt
import numpy as np

class VanillaRNN:
    def __init__(self, input_size, hidden_size, output_size, activation='tanh'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output
        
        # Initialize biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        
        # Set activation function
        self.activation = activation
        if activation == 'tanh':
            self.activate = ActivationFunctions.tanh
        elif activation == 'sigmoid':
            self.activate = ActivationFunctions.sigmoid
        elif activation == 'relu':
            self.activate = ActivationFunctions.relu
        else:
            raise ValueError("Unsupported activation function")
    
    def forward_step(self, x, h_prev):
        """
        Single forward step of RNN
        x: input at current time step (input_size, 1)
        h_prev: hidden state from previous time step (hidden_size, 1)
        """
        # Compute hidden state
        h_raw = np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh
        h = self.activate(h_raw)
        
        # Compute output
        y_raw = np.dot(self.Why, h) + self.by
        y = ActivationFunctions.softmax(y_raw)
        
        # Store intermediate values for backprop
        cache = {
            'x': x,
            'h_prev': h_prev,
            'h_raw': h_raw,
            'h': h,
            'y_raw': y_raw,
            'y': y
        }
        
        return h, y, cache
    
    def forward_propagation(self, inputs, h0=None):
        """
        Forward propagation through entire sequence
        inputs: list of input vectors or (seq_length, input_size) array
        h0: initial hidden state
        """
        if isinstance(inputs, np.ndarray):
            if inputs.ndim == 2:
                inputs = [inputs[i:i+1].T for i in range(inputs.shape[0])]
            else:
                inputs = [inputs[:, i:i+1] for i in range(inputs.shape[1])]
        
        seq_length = len(inputs)
        
        # Initialize hidden state
        if h0 is None:
            h = np.zeros((self.hidden_size, 1))
        else:
            h = h0.copy()
        
        # Store all states and outputs
        hidden_states = []
        outputs = []
        caches = []
        
        # Forward pass through sequence
        for t in range(seq_length):
            h, y, cache = self.forward_step(inputs[t], h)
            hidden_states.append(h.copy())
            outputs.append(y.copy())
            caches.append(cache)
        
        return hidden_states, outputs, caches
    
    def predict(self, inputs, h0=None):
        """Make predictions for a sequence"""
        hidden_states, outputs, _ = self.forward_propagation(inputs, h0)
        return outputs, hidden_states



