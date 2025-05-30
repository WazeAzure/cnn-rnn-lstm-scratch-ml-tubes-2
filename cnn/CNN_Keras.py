import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
import pickle
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CNNTrainer:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
    def load_and_prepare_data(self):
        """Load CIFAR-10 dataset and create train/validation/test splits"""
        print("Loading CIFAR-10 dataset...")
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train_full = x_train_full.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Flatten labels
        y_train_full = y_train_full.flatten()
        y_test = y_test.flatten()
        
        # Split training data into train (40k) and validation (10k) with ratio 4:1
        val_size = 10000
        indices = np.random.permutation(len(x_train_full))
        
        self.x_train = x_train_full[indices[val_size:]]
        self.y_train = y_train_full[indices[val_size:]]
        self.x_val = x_train_full[indices[:val_size]]
        self.y_val = y_train_full[indices[:val_size]]
        self.x_test = x_test
        self.y_test = y_test
        
        print(f"Training data: {self.x_train.shape[0]} samples")
        print(f"Validation data: {self.x_val.shape[0]} samples")
        print(f"Test data: {self.x_test.shape[0]} samples")
        
    def create_model(self, conv_layers=3, filters_config=[32, 64, 128], 
                    kernel_sizes=[3, 3, 3], pooling_type='max'):
        """Create CNN model with specified configuration"""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(32, 32, 3)))
        
        # Convolutional layers
        for i in range(conv_layers):
            model.add(layers.Conv2D(
                filters=filters_config[i % len(filters_config)],
                kernel_size=kernel_sizes[i % len(kernel_sizes)],
                activation='relu',
                padding='same'
            ))
            
            # Pooling layer
            if pooling_type == 'max':
                model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            else:
                model.add(layers.AveragePooling2D(pool_size=(2, 2)))
        
        # Flatten layer
        model.add(layers.Flatten())
        
        # Dense layers
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model, model_name, epochs=20):
        """Train the model and return history"""
        print(f"\nTraining {model_name}...")
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            self.x_train, self.y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(self.x_val, self.y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save model weights
        os.makedirs('saved_models', exist_ok=True)
        model.save_weights(f'saved_models/{model_name}.weights.h5')
        
        # Save model architecture
        with open(f'saved_models/{model_name}_architecture.json', 'w') as f:
            f.write(model.to_json())
        
        return history
    
    def evaluate_model(self, model, model_name):
        """Evaluate model and calculate F1-score"""
        # Predictions on test set
        y_pred_probs = model.predict(self.x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate macro F1-score
        f1_macro = f1_score(self.y_test, y_pred, average='macro')
        
        # Test accuracy
        test_loss, test_acc = model.evaluate(self.x_test, self.y_test, verbose=0)
        
        print(f"\n{model_name} Results:")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Macro F1-Score: {f1_macro:.4f}")
        
        return f1_macro, test_acc, y_pred
    
    def plot_training_history(self, histories, experiment_name):
        """Plot training and validation loss for comparison"""
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        for name, history in histories.items():
            plt.plot(history.history['loss'], label=f'{name} - Train Loss')
            plt.plot(history.history['val_loss'], label=f'{name} - Val Loss', linestyle='--')
        
        plt.title(f'{experiment_name} - Training vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        for name, history in histories.items():
            plt.plot(history.history['accuracy'], label=f'{name} - Train Acc')
            plt.plot(history.history['val_accuracy'], label=f'{name} - Val Acc', linestyle='--')
        
        plt.title(f'{experiment_name} - Training vs Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'plots_{experiment_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def experiment_conv_layers(self):
        """Experiment with different numbers of convolutional layers"""
        print("\n" + "="*60)
        print("EXPERIMENT 1: Effect of Number of Convolutional Layers")
        print("="*60)
        
        configurations = [
            (2, "2 Conv Layers"),
            (3, "3 Conv Layers"), 
            (4, "4 Conv Layers")
        ]
        
        histories = {}
        results = {}
        
        for conv_layers, name in configurations:
            model = self.create_model(conv_layers=conv_layers)
            history = self.train_model(model, f"conv_layers_{conv_layers}", epochs=15)
            f1_score, test_acc, _ = self.evaluate_model(model, name)
            
            histories[name] = history
            results[name] = {'f1_score': f1_score, 'test_acc': test_acc}
        
        # Plot comparison
        self.plot_training_history(histories, "Convolutional Layers Comparison")
        
        # Print summary
        print("\nSUMMARY - Convolutional Layers:")
        for name, metrics in results.items():
            print(f"{name}: F1-Score = {metrics['f1_score']:.4f}, Accuracy = {metrics['test_acc']:.4f}")
        
        return results
    
    def experiment_filter_numbers(self):
        """Experiment with different numbers of filters per layer"""
        print("\n" + "="*60)
        print("EXPERIMENT 2: Effect of Number of Filters per Layer")
        print("="*60)
        
        configurations = [
            ([16, 32, 64], "Small Filters (16-32-64)"),
            ([32, 64, 128], "Medium Filters (32-64-128)"),
            ([64, 128, 256], "Large Filters (64-128-256)")
        ]
        
        histories = {}
        results = {}
        
        for filters_config, name in configurations:
            model = self.create_model(filters_config=filters_config)
            history = self.train_model(model, f"filters_{'-'.join(map(str, filters_config))}", epochs=15)
            f1_score, test_acc, _ = self.evaluate_model(model, name)
            
            histories[name] = history
            results[name] = {'f1_score': f1_score, 'test_acc': test_acc}
        
        # Plot comparison
        self.plot_training_history(histories, "Filter Numbers Comparison")
        
        # Print summary
        print("\nSUMMARY - Filter Numbers:")
        for name, metrics in results.items():
            print(f"{name}: F1-Score = {metrics['f1_score']:.4f}, Accuracy = {metrics['test_acc']:.4f}")
        
        print("\nCONCLUSION:")
        print("- More filters can capture more diverse features")
        print("- But increase computational cost and may cause overfitting")
        print("- Medium-sized filters often provide best balance")
        
        return results
    
    def experiment_kernel_sizes(self):
        """Experiment with different kernel sizes"""
        print("\n" + "="*60)
        print("EXPERIMENT 3: Effect of Kernel Sizes")
        print("="*60)
        
        configurations = [
            ([3, 3, 3], "Small Kernels (3x3)"),
            ([5, 5, 5], "Medium Kernels (5x5)"),
            ([3, 5, 7], "Mixed Kernels (3x3, 5x5, 7x7)")
        ]
        
        histories = {}
        results = {}
        
        for kernel_sizes, name in configurations:
            model = self.create_model(kernel_sizes=kernel_sizes)
            history = self.train_model(model, f"kernels_{'-'.join(map(str, kernel_sizes))}", epochs=15)
            f1_score, test_acc, _ = self.evaluate_model(model, name)
            
            histories[name] = history
            results[name] = {'f1_score': f1_score, 'test_acc': test_acc}
        
        # Plot comparison
        self.plot_training_history(histories, "Kernel Sizes Comparison")
        
        # Print summary
        print("\nSUMMARY - Kernel Sizes:")
        for name, metrics in results.items():
            print(f"{name}: F1-Score = {metrics['f1_score']:.4f}, Accuracy = {metrics['test_acc']:.4f}")
        
        print("\nCONCLUSION:")
        print("- Smaller kernels (3x3) capture fine details")
        print("- Larger kernels capture broader patterns")
        print("- Mixed sizes can provide comprehensive feature extraction")
        
        return results
    
    def experiment_pooling_types(self):
        """Experiment with different pooling types"""
        print("\n" + "="*60)
        print("EXPERIMENT 4: Effect of Pooling Types")
        print("="*60)
        
        configurations = [
            ('max', "Max Pooling"),
            ('average', "Average Pooling")
        ]
        
        histories = {}
        results = {}
        
        for pooling_type, name in configurations:
            model = self.create_model(pooling_type=pooling_type)
            history = self.train_model(model, f"pooling_{pooling_type}", epochs=15)
            f1_score, test_acc, _ = self.evaluate_model(model, name)
            
            histories[name] = history
            results[name] = {'f1_score': f1_score, 'test_acc': test_acc}
        
        # Plot comparison
        self.plot_training_history(histories, "Pooling Types Comparison")
        
        # Print summary
        print("\nSUMMARY - Pooling Types:")
        for name, metrics in results.items():
            print(f"{name}: F1-Score = {metrics['f1_score']:.4f}, Accuracy = {metrics['test_acc']:.4f}")
        
        print("\nCONCLUSION:")
        print("- Max pooling preserves strongest features (good for edge detection)")
        print("- Average pooling provides smoother feature maps")
        print("- Max pooling typically performs better for image classification")
        
        return results
    
    def run_all_experiments(self):
        """Run all hyperparameter experiments"""
        self.load_and_prepare_data()
        
        # Create directories
        os.makedirs('saved_models', exist_ok=True)
        
        # Run experiments
        # conv_results = self.experiment_conv_layers()
        filter_results = self.experiment_filter_numbers()
        kernel_results = self.experiment_kernel_sizes()
        pooling_results = self.experiment_pooling_types()
        
        # Save all results
        all_results = {
            # 'conv_layers': conv_results,
            'filter_numbers': filter_results,
            'kernel_sizes': kernel_results,
            'pooling_types': pooling_results
        }
        
        with open('experiment_results.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        
        print("\n" + "="*60)
        print("ALL EXPERIMENTS COMPLETED!")
        print("Model weights saved in 'saved_models/' directory")
        print("Results saved in 'experiment_results.pkl'")
        print("="*60)
        
        return all_results

# Run the experiments
if __name__ == "__main__":
    trainer = CNNTrainer()
    results = trainer.run_all_experiments()