from ActivationFunction import ActivationFunctions
from VanillaRNN import VanillaRNN
from VanillaGRU import GRU
from VanillaLSTM import LSTM

import numpy as np
import matplotlib.pyplot as plt

def generate_sine_wave_data(seq_length=50, num_sequences=100):
    """Generate sine wave data for testing"""
    X = []
    y = []
    
    for _ in range(num_sequences):
        # Random phase and frequency
        phase = np.random.uniform(0, 2*np.pi)
        freq = np.random.uniform(0.1, 2.0)
        
        # Generate sequence
        t = np.linspace(0, 4*np.pi, seq_length + 1)
        sequence = np.sin(freq * t + phase)
        
        X.append(sequence[:-1])  # Input sequence
        y.append(sequence[1:])   # Target sequence (shifted by 1)
    
    return np.array(X), np.array(y)

def test_rnn_models():
    """Test all RNN models with synthetic data"""
    print("Testing RNN Forward Propagation Models")
    print("=" * 50)
    
    # Generate test data
    seq_length = 10
    input_size = 1
    hidden_size = 20
    output_size = 1
    
    # Create sample input sequence
    X = np.sin(np.linspace(0, 2*np.pi, seq_length)).reshape(-1, 1)
    inputs = [X[i:i+1].T for i in range(seq_length)]
    
    print(f"Input sequence shape: {X.shape}")
    print(f"Sequence length: {seq_length}")
    print(f"Input size: {input_size}, Hidden size: {hidden_size}, Output size: {output_size}")
    print()
    
    # Test Vanilla RNN
    print("1. Testing Vanilla RNN:")
    vanilla_rnn = VanillaRNN(input_size, hidden_size, output_size, activation='tanh')
    h_states, outputs, caches = vanilla_rnn.forward_propagation(inputs)
    
    print(f"   Hidden states shape: {len(h_states)} x {h_states[0].shape}")
    print(f"   Outputs shape: {len(outputs)} x {outputs[0].shape}")
    print(f"   Sample output: {outputs[0].flatten()[:3]}...")
    print()
    
    # Test LSTM
    print("2. Testing LSTM:")
    lstm = LSTM(input_size, hidden_size, output_size)
    h_states, c_states, outputs, caches = lstm.forward_propagation(inputs)
    
    print(f"   Hidden states shape: {len(h_states)} x {h_states[0].shape}")
    print(f"   Cell states shape: {len(c_states)} x {c_states[0].shape}")
    print(f"   Outputs shape: {len(outputs)} x {outputs[0].shape}")
    print(f"   Sample output: {outputs[0].flatten()[:3]}...")
    print()
    
    # Test GRU
    print("3. Testing GRU:")
    gru = GRU(input_size, hidden_size, output_size)
    h_states, outputs, caches = gru.forward_propagation(inputs)
    
    print(f"   Hidden states shape: {len(h_states)} x {h_states[0].shape}")
    print(f"   Outputs shape: {len(outputs)} x {outputs[0].shape}")
    print(f"   Sample output: {outputs[0].flatten()[:3]}...")
    print()
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Plot input sequence
    plt.subplot(1, 4, 1)
    plt.plot(X.flatten())
    plt.title('Input Sequence')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    
    # Plot RNN outputs
    rnn_out = [out.flatten()[0] for out in outputs]
    plt.subplot(1, 4, 2)
    plt.plot(rnn_out)
    plt.title('Vanilla RNN Output')
    plt.xlabel('Time Step')
    plt.ylabel('Output')
    
    # Plot LSTM outputs
    lstm_out = [out.flatten()[0] for out in outputs]
    plt.subplot(1, 4, 3)
    plt.plot(lstm_out)
    plt.title('LSTM Output')
    plt.xlabel('Time Step')
    plt.ylabel('Output')
    
    # Plot GRU outputs
    gru_out = [out.flatten()[0] for out in outputs]
    plt.subplot(1, 4, 4)
    plt.plot(gru_out)
    plt.title('GRU Output')
    plt.xlabel('Time Step')
    plt.ylabel('Output')
    
    plt.tight_layout()
    plt.show()
    
    return vanilla_rnn, lstm, gru

if __name__ == "__main__":
    # Run tests
    vanilla_rnn, lstm, gru = test_rnn_models()
    
    # Example of using the models for prediction
    print("\nExample Usage:")
    print("-" * 30)
    
    # Create a simple sequence
    test_sequence = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
    test_inputs = [test_sequence[i:i+1].T for i in range(len(test_sequence))]
    
    print("Input sequence:", [x.flatten()[0] for x in test_inputs])
    
    # Get predictions from each model
    vanilla_pred, _ = vanilla_rnn.predict(test_inputs)
    lstm_h, lstm_c, lstm_pred, _ = lstm.forward_propagation(test_inputs)
    gru_pred, _ = gru.forward_propagation(test_inputs)
    
    print("Vanilla RNN predictions:", [p.flatten()[0] for p in vanilla_pred[:3]])
    print("LSTM predictions:", [p.flatten()[0] for p in lstm_pred[:3]])
    print("GRU predictions:", [p.flatten()[0] for p in gru_pred[:3]])