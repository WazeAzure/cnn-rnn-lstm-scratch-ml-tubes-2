import numpy as np
import h5py
import time
from sklearn.metrics import f1_score, accuracy_score
import sys
from scipy.signal import correlate 

def relu_manual(x):
    return np.maximum(0, x)

def softmax_manual(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def conv2d_scipy(input_batch, W, b, stride=1, padding_mode='same'):
    """
    Optimized 2D convolution using scipy.signal.correlate.
    input_batch: (batch_size, height, width, in_channels)
    W: (kernel_h, kernel_w, in_channels, out_channels)
    b: (out_channels,)
    Keras Conv2D is cross-correlation, which scipy.signal.correlate performs.
    """
    if stride != 1:
        raise NotImplementedError("Stride > 1 not implemented for conv2d_scipy yet.")
    if padding_mode != 'same':
        raise NotImplementedError("Only 'same' padding implemented for conv2d_scipy.")

    (batch_size, n_H_prev, n_W_prev, n_C_prev) = input_batch.shape
    (f_h, f_w, _, n_C_out) = W.shape 

    # Output dimensions for 'same' padding and stride 1
    n_H_out = n_H_prev
    n_W_out = n_W_prev
    
    Z = np.zeros((batch_size, n_H_out, n_W_out, n_C_out))

    for i in range(batch_size):          
        for c_out_idx in range(n_C_out): 
            W_one_filter = W[:, :, :, c_out_idx]
       
            convolved_channel_sum = np.zeros((n_H_out, n_W_out))
            
            for c_in_idx in range(n_C_prev):
                input_slice = input_batch[i, :, :, c_in_idx] 
                kernel_slice = W_one_filter[:, :, c_in_idx]
                
                convolved_channel_sum += correlate(input_slice, kernel_slice, mode='same', method='auto')
            
            Z[i, :, :, c_out_idx] = convolved_channel_sum + b[c_out_idx]
    return Z

def pool2d_reshaping(input_batch, pool_size=(2, 2), stride_val=None, mode='max'):
    """
    Optimized 2D Pooling using NumPy reshaping.
    This version assumes stride_val is equal to pool_size (non-overlapping pooling).
    input_batch: (batch_size, height, width, channels)
    """
    if stride_val is None:
        stride_val = pool_size 
    
    if stride_val != pool_size:
        raise NotImplementedError("pool2d_reshaping currently only supports stride == pool_size.")

    B, H_in, W_in, C = input_batch.shape
    pH, pW = pool_size 

    if H_in % pH != 0 or W_in % pW != 0:
     
        print(f"Warning: Pooling input dimensions H_in={H_in}, W_in={W_in} "
              f"not perfectly divisible by pool_size pH={pH}, pW={pW}. "
              "Results might be incorrect if Keras applied padding for pooling.")
    
    H_out = H_in // pH
    W_out = W_in // pW

    reshaped = input_batch.reshape(B, H_out, pH, W_out, pW, C)
    transposed = reshaped.transpose(0, 1, 3, 2, 4, 5)
    
    if mode == 'max':
        output = transposed.max(axis=(3, 4)) 
    elif mode == 'average':
        output = transposed.mean(axis=(3, 4))
    else:
        raise ValueError("Pooling mode must be 'max' or 'average'")
    return output


def flatten_manual(input_batch):
    return input_batch.reshape(input_batch.shape[0], -1)

def dense_manual(A_prev_flattened, W, b): 
    return np.dot(A_prev_flattened, W) + b


class NumpyCNNConfigurable:
    def __init__(self, architecture_config, weights_path):
        self.config = architecture_config
        self.weights = {}
        self.class_names = self.config.get('class_names', [])
        print(f"Initializing NumPy CNN for weights: {weights_path}")
        self._load_weights(weights_path) 

    def _load_weights(self, weights_path):
        print(f"Loading weights from {weights_path}...")
        try:
            with h5py.File(weights_path, 'r') as f:
                print(f"  HDF5 file opened. Top-level keys: {list(f.keys())}")
                if 'layers' not in f:
                    raise KeyError("'layers' group not found in HDF5 file.")
                layers_hdf5_group = f['layers']
                print(f"  Found 'layers' top-level group.")
                layer_group_keys = list(layers_hdf5_group.keys())
                print(f"  Keys under 'layers' (sample): {layer_group_keys[:10]}" + ("..." if len(layer_group_keys) > 10 else ""))

                for i, layer_spec in enumerate(self.config['layers']):
                    keras_layer_group_name_from_config = layer_spec.get('keras_name') 
                    if not keras_layer_group_name_from_config: continue
                    
                    weight_key_w = f'layer_{i}_w'; weight_key_b = f'layer_{i}_b'
                    print(f"\n  Processing Keras layer from config: '{keras_layer_group_name_from_config}' (NumPy layer index {i})")
                    try:
                        if keras_layer_group_name_from_config not in layers_hdf5_group:
                            raise KeyError(f"Keras layer group '{keras_layer_group_name_from_config}' not found under 'layers'")
                        current_layer_hdf5_group = layers_hdf5_group[keras_layer_group_name_from_config]
                      
                        if 'vars' in current_layer_hdf5_group and isinstance(current_layer_hdf5_group['vars'], h5py.Group):
                            final_vars_group_for_weights = current_layer_hdf5_group['vars']
                        else:
                            final_vars_group_for_weights = current_layer_hdf5_group
                        
                        kernel_dataset_name = '0'; bias_dataset_name = '1'

                        if kernel_dataset_name not in final_vars_group_for_weights:
                            raise KeyError(f"Dataset '{kernel_dataset_name}' not found in {final_vars_group_for_weights.name}")
                        if bias_dataset_name not in final_vars_group_for_weights:
                            raise KeyError(f"Dataset '{bias_dataset_name}' not found in {final_vars_group_for_weights.name}")

                        self.weights[weight_key_w] = final_vars_group_for_weights[kernel_dataset_name][:]
                        self.weights[weight_key_b] = final_vars_group_for_weights[bias_dataset_name][:]
                        print(f"      Successfully loaded weights for '{keras_layer_group_name_from_config}'.")
                    except Exception as e_load_layer:
                        print(f"    ERROR during weight loading for Keras layer '{keras_layer_group_name_from_config}': {e_load_layer}")
                        raise Exception(f"Critical weight loading failure for layer '{keras_layer_group_name_from_config}'. Original error: {e_load_layer}")
            print("\nNumPy model weights loaded successfully.")
        except Exception as e:
            print(f"Fatal error during HDF5 file processing or weight loading: {e}")
            raise

    def predict_batch(self, image_batch):
        x = image_batch.astype(np.float32)
        
        for i, layer_spec in enumerate(self.config['layers']):
            layer_type = layer_spec['type']

            if layer_type == 'conv':
                W = self.weights[f'layer_{i}_w']
                b = self.weights[f'layer_{i}_b']
                x = conv2d_scipy(x, W, b, 
                                 stride=layer_spec.get('stride', 1), 
                                 padding_mode=layer_spec.get('padding', 'same'))
                if layer_spec.get('activation') == 'relu':
                    x = relu_manual(x)
            
            elif layer_type == 'pool':
                x = pool2d_reshaping(x, 
                                     pool_size=layer_spec['pool_size'], 
                                     mode=layer_spec['pool_type'])
            
            elif layer_type == 'flatten':
                x = flatten_manual(x)
            
            elif layer_type == 'dense':
                W = self.weights[f'layer_{i}_w']
                b = self.weights[f'layer_{i}_b']
                x = dense_manual(x, W, b)
                if layer_spec.get('activation') == 'relu':
                    x = relu_manual(x)
                elif layer_spec.get('activation') == 'softmax':
                    x = softmax_manual(x)
        return x

    def evaluate(self, x_test_normalized, y_test_true, batch_size=32): 
        print(f"\nEvaluating NumPy model on {len(x_test_normalized)} samples...")
        y_pred_all = []
        num_batches = (len(x_test_normalized) + batch_size - 1) // batch_size
        start_time_eval = time.time()
        bar_length = 50
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(x_test_normalized))
            batch_x = x_test_normalized[start_idx:end_idx]
            batch_probs = self.predict_batch(batch_x)
            batch_preds = np.argmax(batch_probs, axis=1)
            y_pred_all.extend(batch_preds)
            progress = (i + 1) / num_batches
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            status_text = f"Batch {i + 1}/{num_batches}"
            sys.stdout.write(f'\rProgress: [{bar}] {progress*100:.1f}% ({status_text})  ')
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        end_time_eval = time.time()
        total_eval_time = end_time_eval - start_time_eval
        print(f"Evaluation completed in {total_eval_time:.2f} seconds.")
        y_pred_all = np.array(y_pred_all)
        y_test_true_np = np.array(y_test_true)
        acc = accuracy_score(y_test_true_np, y_pred_all)
        f1 = f1_score(y_test_true_np, y_pred_all, average='macro', zero_division=0) 
        print(f"NumPy Model Results:")
        print(f"  Test Accuracy: {acc:.4f}")
        print(f"  Macro F1-Score: {f1:.4f}")
        return acc, f1, y_pred_all

if __name__ == "__main__":
    config_model = {
        'input_shape': (32, 32, 3),
        'layers': [
            {'type': 'conv', 'filters': 32, 'kernel_size': (3,3), 'padding': 'same', 
             'activation': 'relu', 'keras_name': 'conv2d', 'stride': 1}, # Assuming stride 1 for conv
            {'type': 'pool', 'pool_size': (2,2), 'stride': (2,2), 'pool_type': 'max'}, # stride must match pool_size for pool2d_reshaping
            
            {'type': 'conv', 'filters': 64, 'kernel_size': (3,3), 'padding': 'same', 
             'activation': 'relu', 'keras_name': 'conv2d_1', 'stride': 1}, # Assuming stride 1 for conv
            {'type': 'pool', 'pool_size': (2,2), 'stride': (2,2), 'pool_type': 'max'}, # stride must match pool_size for pool2d_reshaping
            
            {'type': 'flatten'},
            {'type': 'dense', 'units': 128, 'activation': 'relu', 'keras_name': 'dense'},
            {'type': 'dense', 'units': 10, 'activation': 'softmax', 'keras_name': 'dense_1'}
        ],
        'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck']
    }
    
    WEIGHTS_FILE_TO_TEST = 'saved_models/conv_layers_2.weights.h5'

    try:
        numpy_cnn_model = NumpyCNNConfigurable(config_model, WEIGHTS_FILE_TO_TEST)
    except Exception as e:
        print(f"Failed to initialize NumPy model: {e}")
        exit()
