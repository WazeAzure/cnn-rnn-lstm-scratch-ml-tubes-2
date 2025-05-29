import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_NAME = "rnn_model_final.keras"

class VanillaSimpleRNN:
    def __init__(self):
        self.vocab_size = None
        self.sequence_length = None
        self.embedding_dim = None
        self.hidden_dim = None
        self.output_dim = None
        
        # Text vectorization layer
        self.vectorizer = None
        
        # Weight matrices
        self.embedding_weights = None
        
        # SimpleRNN weights (for each direction and layer)
        self.rnn1_forward_weights = {}
        self.rnn1_backward_weights = {}
        self.rnn2_forward_weights = {}
        self.rnn2_backward_weights = {}
        
        # Dense layer weights
        self.dense_weights = None
        self.dense_bias = None
        
    def load_weights_from_model(self, model_path_or_model):
        """Load weights from a trained Keras model"""
        if isinstance(model_path_or_model, str):
            model = load_model(model_path_or_model, compile=False)
        else:
            model = model_path_or_model
            
        layers = model.layers
        
        # Extract text vectorization layer
        text_vec_layer = None
        for layer in layers:
            if 'text_vectorization' in layer.name.lower() or hasattr(layer, 'get_vocabulary'):
                text_vec_layer = layer
                break
        
        if text_vec_layer:
            self.vectorizer = text_vec_layer
            try:
                vocab = text_vec_layer.get_vocabulary()
                self.vocab_size = len(vocab)
                print(f"Loaded vocabulary with {self.vocab_size} tokens")
                # Print first few tokens for verification
                print(f"First 10 tokens: {vocab[:10]}")
            except Exception as e:
                print(f"Could not extract vocabulary: {e}")
                self.vocab_size = layers[1].input_dim if len(layers) > 1 else 10000
        else:
            print("Warning: No text vectorization layer found")
            self.vocab_size = layers[1].input_dim if len(layers) > 1 else 10000
            
        # Get sequence length from vectorizer or model config
        if self.vectorizer and hasattr(self.vectorizer, 'output_sequence_length'):
            self.sequence_length = self.vectorizer.output_sequence_length
        else:
            self.sequence_length = 100  # Default from your model
            
        self.embedding_dim = 128
        self.hidden_dim = 128
        self.output_dim = 3
        
        # Load embedding weights
        for layer in layers:
            if 'embedding' in layer.name:
                self.embedding_weights = layer.get_weights()[0]
                print(f"Loaded embedding weights: {self.embedding_weights.shape}")
                break
            
        # Load SimpleRNN weights
        bidirectional_layers = [layer for layer in layers if 'bidirectional' in layer.name]
        
        if len(bidirectional_layers) > 0:
            rnn1_weights = bidirectional_layers[0].get_weights()
            self._parse_bidirectional_rnn_weights(rnn1_weights, 'rnn1')
            print(f"Loaded first SimpleRNN layer weights")
            
        if len(bidirectional_layers) > 1:
            rnn2_weights = bidirectional_layers[1].get_weights()
            self._parse_bidirectional_rnn_weights(rnn2_weights, 'rnn2')
            print(f"Loaded second SimpleRNN layer weights")
            
        # Load dense layer weights
        for layer in layers:
            if 'dense' in layer.name:
                dense_weights = layer.get_weights()
                self.dense_weights = dense_weights[0]
                self.dense_bias = dense_weights[1]
                print(f"Loaded dense layer weights: {self.dense_weights.shape}")
                break
                
    def _parse_bidirectional_rnn_weights(self, weights, layer_name):
        """Parse bidirectional SimpleRNN weights into forward and backward components"""
        # SimpleRNN has only 3 weight matrices per direction:
        # - kernel: input to hidden weights
        # - recurrent_kernel: hidden to hidden weights  
        # - bias: bias vector
        
        if layer_name == 'rnn1':
            self.rnn1_forward_weights = {
                'kernel': weights[0],           # Input to hidden weights
                'recurrent_kernel': weights[1], # Hidden to hidden weights
                'bias': weights[2]              # Bias
            }
            self.rnn1_backward_weights = {
                'kernel': weights[3],           # Input to hidden weights
                'recurrent_kernel': weights[4], # Hidden to hidden weights
                'bias': weights[5]              # Bias
            }
        elif layer_name == 'rnn2':
            self.rnn2_forward_weights = {
                'kernel': weights[0],           # Input to hidden weights
                'recurrent_kernel': weights[1], # Hidden to hidden weights
                'bias': weights[2]              # Bias
            }
            self.rnn2_backward_weights = {
                'kernel': weights[3],           # Input to hidden weights
                'recurrent_kernel': weights[4], # Hidden to hidden weights
                'bias': weights[5]              # Bias
            }
    
    def tanh(self, x):
        """Tanh activation function"""
        return np.tanh(np.clip(x, -500, 500))
    
    def softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def simple_rnn_cell_forward(self, x, h_prev, weights):
        """Single SimpleRNN cell forward pass"""
        # SimpleRNN formula: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
        # where W_ih is kernel, W_hh is recurrent_kernel, b is bias
        
        # Input transformation
        input_contrib = np.dot(x, weights['kernel'])
        
        # Recurrent transformation
        recurrent_contrib = np.dot(h_prev, weights['recurrent_kernel'])
        
        # Combine and apply activation
        h_new = self.tanh(input_contrib + recurrent_contrib + weights['bias'])
        
        return h_new
    
    def simple_rnn_forward(self, embeddings, weights, return_sequences=True):
        """Forward pass through SimpleRNN layer"""
        batch_size, seq_len, input_dim = embeddings.shape
        hidden_dim = weights['bias'].shape[0]
        
        # Initialize hidden state
        h = np.zeros((batch_size, hidden_dim))
        
        outputs = []
        
        for t in range(seq_len):
            h = self.simple_rnn_cell_forward(embeddings[:, t, :], h, weights)
            outputs.append(h)
            
        if return_sequences:
            return np.stack(outputs, axis=1)  # (batch_size, seq_len, hidden_dim)
        else:
            return h  # (batch_size, hidden_dim) - only last output
    
    def simple_rnn_backward(self, embeddings, weights, return_sequences=True):
        """Backward pass through SimpleRNN layer (reverse sequence)"""
        batch_size, seq_len, input_dim = embeddings.shape
        hidden_dim = weights['bias'].shape[0]
        
        # Initialize hidden state
        h = np.zeros((batch_size, hidden_dim))
        
        outputs = []
        
        # Process sequence in reverse order
        for t in range(seq_len-1, -1, -1):
            h = self.simple_rnn_cell_forward(embeddings[:, t, :], h, weights)
            outputs.append(h)
            
        # Reverse outputs to match forward direction
        outputs = outputs[::-1]
        
        if return_sequences:
            return np.stack(outputs, axis=1)  # (batch_size, seq_len, hidden_dim)
        else:
            return outputs[0]  # (batch_size, hidden_dim) - last processed (first in sequence)
    
    def forward(self, input_ids):
        """Complete forward propagation"""
        # Ensure input is 2D (batch_size, sequence_length)
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
            
        batch_size = input_ids.shape[0]
        
        # 1. Embedding layer
        embeddings = self.embedding_weights[input_ids]  # (batch_size, seq_len, embedding_dim)
        
        # 2. First Bidirectional SimpleRNN layer
        rnn1_forward = self.simple_rnn_forward(embeddings, self.rnn1_forward_weights, return_sequences=True)
        rnn1_backward = self.simple_rnn_backward(embeddings, self.rnn1_backward_weights, return_sequences=True)
        
        # Concatenate forward and backward outputs
        rnn1_output = np.concatenate([rnn1_forward, rnn1_backward], axis=2)
        
        # Apply dropout (in inference, just pass through)
        # dropout1_output = rnn1_output  # No dropout during inference
        
        # 3. Second Bidirectional SimpleRNN layer (return_sequences=False for last layer)
        rnn2_forward = self.simple_rnn_forward(rnn1_output, self.rnn2_forward_weights, return_sequences=False)
        rnn2_backward = self.simple_rnn_backward(rnn1_output, self.rnn2_backward_weights, return_sequences=False)
        
        # Concatenate forward and backward outputs  
        rnn2_output = np.concatenate([rnn2_forward, rnn2_backward], axis=1)
        
        # Apply dropout (in inference, just pass through)
        # dropout2_output = rnn2_output  # No dropout during inference
        
        # 4. Dense layer
        dense_output = np.dot(rnn2_output, self.dense_weights) + self.dense_bias
        
        # 5. Softmax activation for classification
        probabilities = self.softmax(dense_output)
        
        return probabilities
    
    def vectorize_text(self, texts):
        """Convert raw text to token sequences using the loaded vectorizer"""
        if self.vectorizer is None:
            raise ValueError("No text vectorizer loaded. Make sure to load a model with text vectorization layer.")
        
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Convert to tensor and vectorize
        text_tensor = tf.constant(texts)
        vectorized = self.vectorizer(text_tensor)
        
        # Convert back to numpy
        return vectorized.numpy()
    
    def predict_from_text(self, texts):
        """Direct prediction from raw text input"""
        # Vectorize the text
        vectorized_input = self.vectorize_text(texts)
        
        # Make predictions
        predictions, probabilities = self.predict(vectorized_input)
        
        return predictions, probabilities
    
    def predict(self, input_ids):
        """Make predictions on tokenized input"""
        probabilities = self.forward(input_ids)
        predictions = np.argmax(probabilities, axis=1)
        return predictions, probabilities

def test_vanilla_simple_rnn_with_text():
    """Test cases with real text input using the built-in vectorizer"""
    
    print("=== VanillaSimpleRNN Text Processing Test Cases ===\n")
    
    # Initialize SimpleRNN (you would load your actual model here)
    rnn = VanillaSimpleRNN()
    rnn.load_weights_from_model(MODEL_NAME)
    
    # Test Case 1: Single text prediction
    print("Test Case 1: Single text prediction")
    single_text = "restoran ini sangat enak"
    
    try:
        # Method 1: Direct text prediction
        predictions, probabilities = rnn.predict_from_text(single_text)
        print(f"Input text: '{single_text}'")
        print(f"Predicted class: {predictions[0]}")
        print(f"Class probabilities: {probabilities[0]}")
        print(f"Confidence: {np.max(probabilities[0]):.4f}")
        
        # Method 2: Manual vectorization then prediction
        vectorized = rnn.vectorize_text(single_text)
        print(f"Vectorized shape: {vectorized.shape}")
        print(f"First 10 tokens: {vectorized[0][:10]}")
        
    except Exception as e:
        print(f"Error in single text test: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test Case 2: Batch text prediction
    print("Test Case 2: Batch text prediction")
    batch_texts = [
        "makanan sangat lezat dan pelayanan bagus",
        "pelayanan buruk dan makanan tidak enak", 
        "biasa saja, tidak istimewa"
    ]
    
    try:
        predictions, probabilities = rnn.predict_from_text(batch_texts)
        
        for i, text in enumerate(batch_texts):
            print(f"Text {i+1}: '{text}'")
            print(f"  Predicted class: {predictions[i]}")
            print(f"  Class probabilities: {probabilities[i]}")
            print(f"  Confidence: {np.max(probabilities[i]):.4f}")
            print()
            
    except Exception as e:
        print(f"Error in batch text test: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test Case 3: Edge cases
    print("Test Case 3: Edge cases")
    edge_cases = [
        "",                    # Empty string
        "hanya satu kata",     # Short text
        "ini adalah teks yang sangat panjang dengan banyak kata untuk menguji kemampuan model dalam menangani input yang lebih kompleks dan panjang"  # Long text
    ]
    
    for i, text in enumerate(edge_cases):
        try:
            predictions, probabilities = rnn.predict_from_text(text)
            print(f"Edge case {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print(f"  Predicted class: {predictions[0]}")
            print(f"  Confidence: {np.max(probabilities[0]):.4f}")
            print()
        except Exception as e:
            print(f"Error with edge case {i+1}: {e}")
            print()

# Run the test
if __name__ == "__main__":
    test_vanilla_simple_rnn_with_text()