import pickle
import sys
import time
import warnings
import numpy as np
import cupy as cp
from sklearn.metrics import classification_report
from .gpu import gpu
from .layers import PositionalEmbeddingLayer
from .layers import EmbeddingLayer
from .layers import TransformerBlock
from .layers import DenseLayer
from .layers import Flatten
from .layers import DebugShape
from .layers import Conv2DLayer
from .layers import MaxPool2DLayer
from .layers import MultiHeadAttentionLayer
from .layers import TransformerFeedForward
from .lossfunctions import MSE
from .utils import DebugLogger
from .utils.regularization import l1_regularization, l2_regularization
class LayerConfig:
    def __init__(self, layer_class, activation=None, **kwargs):
        self.layer_class = layer_class
        self.activation = activation
        self.kwargs = kwargs
class Model:
    def __init__(self, layer_configs, input_size=None, regularization=None):
        self.layers = []
        self.regularization = regularization
        self.logger= DebugLogger(enabled=True, to_file=True)
        in_features = input_size
        self.logger.log(f"Initializing model with input size: {in_features}, regularization: {regularization}, layers: {len(layer_configs)}", "INFO")
        for i, config in enumerate(layer_configs):
            layer_cls = config.layer_class
            activation = config.activation
            kwargs = config.kwargs
            
            if layer_cls == DenseLayer:
                if in_features is None:
                    raise ValueError("input_size must be provided for Dense layers")
                out_features = kwargs["output_size"]
                layer = DenseLayer(in_features, out_features, activation, regularization)
                in_features = out_features
            elif layer_cls == Conv2DLayer:
                kwargs["input_shape"] = in_features   
                kwargs["activation"] = activation
                layer = Conv2DLayer(**kwargs)
                in_features = layer.output_shape 
            elif layer_cls == Flatten:
                layer = Flatten(input_shape=in_features)
                in_features = layer.output_size
            elif layer_cls == MaxPool2DLayer:
                layer = MaxPool2DLayer(**kwargs)
                self.layers.append(layer)
                c, h, w = in_features
                k = kwargs.get("kernel_size", 2)
                s = kwargs.get("stride", 2)
                h = (h - k) // s + 1
                w = (w - k) // s + 1
                in_features = (c, h, w)
                continue 
            elif layer_cls == DebugShape:
                layer = DebugShape()
                in_features = in_features
            elif layer_cls == MultiHeadAttentionLayer:
                num_heads = kwargs["num_heads"]
                output_size = kwargs["output_size"]
                layer = MultiHeadAttentionLayer(input_size=in_features, output_size=output_size, num_heads=num_heads, activation=activation)
                in_features = output_size
            elif layer_cls == TransformerFeedForward:
                hidden_size = kwargs["hidden_size"]
                layer = TransformerFeedForward(
                    input_size=in_features,
                    hidden_size=hidden_size,
                    activation=activation,
                    regularization=regularization
                )
                in_features = in_features  
                self.layers.append(layer)
            elif layer_cls == PositionalEmbeddingLayer:
                max_len = kwargs["max_len"]
                embedding_dim = kwargs["embedding_dim"]
                layer = PositionalEmbeddingLayer(max_len=max_len, embedding_dim=embedding_dim)
            elif layer_cls == EmbeddingLayer:
                vocab_size = kwargs["vocab_size"]
                embedding_dim = kwargs["embedding_dim"]
                layer = EmbeddingLayer(vocab_size=vocab_size, embedding_dim=embedding_dim)
                in_features = embedding_dim
                self.layers.append(layer)
            elif layer_cls == TransformerBlock:
                num_heads = kwargs["num_heads"]
                hidden_size = kwargs["hidden_size"]
                layer = TransformerBlock(input_size=in_features, num_heads=num_heads, hidden_size=hidden_size)
                self.layers.append(layer)
            else:
                raise ValueError(f"Unsupported layer class: {layer_cls}")
            self.layers.append(layer)           
    def forward(self, x):
        out = gpu.to_gpu(x)
        for layer in self.layers:
            out = layer.forward(out)
        return out
    def backward(self, loss_grad, learning_rate,lambda_=0.0):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate,lambda_)
    def predict(self, x):
        return gpu.to_cpu(self.forward(x))
    def custom_warning(self,message, category, filename,lineno, file=None, line=None):
        self.logger.log(f"[WARNING] {message} ({category.__name__}) at {filename}:{lineno}", "WARNING")
    def validate(self, X_train, y_train, X_test, y_test,
             learning_rate, epochs, batch_size,
             verbose, loss_fn,
             lambda_, warmup_epochs,
             early_stopping, patience):
        xp = gpu.xp
        # Check input types and shapes
        self.logger.log("Validating input data types and shapes...", "INFO")
        assert isinstance(X_train, (np.ndarray, cp.ndarray)), "X_train must be a NumPy or CuPy array"
        assert isinstance(y_train, (np.ndarray, cp.ndarray)), "y_train must be a NumPy or CuPy array"
        assert isinstance(X_test, (np.ndarray, cp.ndarray)), "X_test must be a NumPy or CuPy array"
        assert isinstance(y_test, (np.ndarray, cp.ndarray)), "y_test must be a NumPy or CuPy array"
        self.logger.log("Validating input data shapes...", "INFO")
        assert X_train.shape[0] == y_train.shape[0], "X_train and y_train must have the same number of samples"
        assert X_test.shape[0] == y_test.shape[0], "X_test and y_test must have the same number of samples"
        # Check valid range for hyperparameters
        self.logger.log("Validating hyperparameters...", "INFO")
        assert isinstance(learning_rate, (float, int)) and learning_rate > 0, "learning_rate must be a positive float"
        assert isinstance(epochs, int) and epochs > 0, "epochs must be a positive integer"
        assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be a positive integer"
        assert isinstance(verbose, bool), "verbose must be a boolean"
        assert callable(loss_fn) or hasattr(loss_fn, '__call__'), "loss_fn must be callable"
        assert isinstance(lambda_, (float, int)) and lambda_ >= 0, "lambda_ must be a non-negative float"
        assert isinstance(warmup_epochs, int) and warmup_epochs >= 0, "warmup_epochs must be a non-negative integer"
        assert isinstance(early_stopping, bool), "early_stopping must be a boolean"
        assert isinstance(patience, int) and patience >= 0, "patience must be a non-negative integer"
    def train(self, X_train, y_train, X_test, y_test,
          learning_rate=0.001, epochs=100, batch_size=32,
          verbose=False, loss_fn=MSE,
          lambda_=0.001, warmup_epochs=5,
          early_stopping=False, patience=10,debug=False):
        self.logger.set_enabled(debug)
        warnings.showwarning = self.custom_warning
        if not gpu._has_cuda:
            self.logger.log("CUDA not available. Using CPU for training.", "WARNING")
        else:
            self.logger.log("CUDA available. Using GPU for training.", "INFO")
        self.validate(X_train, y_train, X_test, y_test,
                  learning_rate, epochs, batch_size,
                  verbose, loss_fn,
                  lambda_, warmup_epochs,
                  early_stopping, patience)
        self.loss_fn = loss_fn
        n_samples = X_train.shape[0]
        best_loss = float('inf')
        epochs_no_improve = 0
        X_train = gpu.to_gpu(X_train)
        y_train = gpu.to_gpu(y_train)
        X_test = gpu.to_gpu(X_test)
        y_test = gpu.to_gpu(y_test)

        orig_batch_size = batch_size

        for epoch in range(epochs):
            start_time = time.time()
            if verbose:
                self.logger.log(f"Starting epoch {epoch+1}/{epochs} with batch size {batch_size}", "INFO")
            indices = gpu.xp.arange(n_samples)
            indices = gpu.to_cpu(indices)
            np.random.shuffle(indices)

            total_loss = 0
            num_batches = int(np.ceil(n_samples / batch_size))
            batch_size_ok = False

            while not batch_size_ok:
                try:
                    print(f"\nEpoch {epoch+1}/{epochs} - Using batch size: {batch_size}")
                    for batch_idx, start in enumerate(range(0, n_samples, batch_size)):
                        end = start + batch_size
                        batch_indices = indices[start:end]
                        X_batch = X_train[batch_indices]
                        y_batch = y_train[batch_indices]
                        output = self.forward(X_batch)
                        loss = self.loss_fn(y_batch, output)
                        total_loss += loss
                        if epoch >= warmup_epochs:
                            if self.regularization == 'l2':
                                loss += l2_regularization([layer.w for layer in self.layers if hasattr(layer, 'w')], lambda_)
                            elif self.regularization == 'l1':
                                loss += l1_regularization([layer.w for layer in self.layers if hasattr(layer, 'w')], lambda_)
                        grad_loss = loss_fn.derivative(y_batch, output)
                        self.backward(grad_loss, learning_rate, lambda_)
                        percent = (batch_idx + 1) / num_batches
                        bar = '=' * int(30 * percent) + '-' * (30 - int(30 * percent))
                        elapsed = time.time() - start_time
                        eta = (elapsed / (batch_idx + 1)) * (num_batches - (batch_idx + 1))
                        sys.stdout.write(f"\rEpoch {epoch+1}/{epochs} [{bar}] {batch_idx+1}/{num_batches} - Loss: {gpu.to_cpu(loss):.4f} - ETA: {eta:.1f}s")
                        sys.stdout.flush()
                    batch_size_ok = True

                except Exception as e:
                    import traceback
                    if 'out of memory' in str(e).lower() or 'cudaErrorMemoryAllocation' in str(e):
                        print(f"\nCUDA OOM detected at batch size {batch_size}. Reducing batch size...")
                        batch_size = max(1, batch_size // 2)
                        if batch_size == 1:
                            print("Batch size reduced to 1 but still OOM. Exiting training.")
                            raise e
                        continue
                    else:
                        traceback.print_exc()
                        raise e

            avg_loss = total_loss / num_batches
            epoch_time = time.time() - start_time
            # === Batched inference on test set ===
            y_pred_batches = []
            for i in range(0, X_test.shape[0], batch_size):
                Xb = X_test[i:i+batch_size]
                try:
                    out = self.forward(Xb)
                    y_pred_batches.append(out)
                    del Xb, out
                except Exception as e:
                    print(f"[ERROR] during test batch {i}: {e}")
                    continue
            y_pred = gpu.xp.concatenate(y_pred_batches, axis=0)
            y_true_cpu = gpu.to_cpu(y_test)
            y_pred_cpu = gpu.to_cpu(y_pred)
            if y_true_cpu.shape[1] > 1:
                true_labels = np.argmax(y_true_cpu, axis=1)
                pred_labels = np.argmax(y_pred_cpu, axis=1)
                accuracy = np.mean(true_labels == pred_labels)
            else:
                accuracy = np.mean(np.abs(y_true_cpu - y_pred_cpu) < 0.5)

            sys.stdout.write(f"\rEpoch {epoch+1}/{epochs} [{'='*30}] - {num_batches}/{num_batches} - {epoch_time:.1f}s - Loss: {gpu.to_cpu(avg_loss):.4f} - Acc: {accuracy:.4f}\n")
            sys.stdout.flush()
            if early_stopping:
                if avg_loss < best_loss - 1e-4:
                    best_loss = avg_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Stopping early at epoch {epoch+1} due to no improvement.")
                        self.logger.log(f"Early stopping at epoch {epoch+1} - No improvement for {patience} epochs", "INFO")
                        break
            self.logger.log(f"Epoch {epoch+1}/{epochs} completed - Loss: {gpu.to_cpu(avg_loss):.4f}, Accuracy: {accuracy:.4f}, Time: {epoch_time:.1f}s", "INFO")
        y_pred_batches = []
        for i in range(0, X_test.shape[0], batch_size):
            Xb = X_test[i:i+batch_size]
            try:
                out = self.forward(Xb)
                y_pred_batches.append(out)
                del Xb, out
                if gpu._has_cuda:
                    cp.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                print(f"[ERROR] during final test batch {i}: {e}")
                self.logger.log(f"[ERROR] during final test batch {i}: {e}", "ERROR")
                continue
        y_pred = gpu.xp.concatenate(y_pred_batches, axis=0)
        self.accuracy(y_test, y_pred)
    def train_batch(self, X_batch, y_batch, learning_rate=0.01, lambda_=0.0, loss_fn=None):
        X_batch = gpu.to_gpu(X_batch)
        y_batch = gpu.to_gpu(y_batch)
        output = self.forward(X_batch)
        loss = loss_fn(y_batch, output)
        if self.regularization == 'l2':
            loss += l2_regularization([layer.w for layer in self.layers if hasattr(layer, 'w')], lambda_)
        elif self.regularization == 'l1':
            loss += l1_regularization([layer.w for layer in self.layers if hasattr(layer, 'w')], lambda_)
        grad_loss = loss_fn.derivative(y_batch, output)
        self.backward(grad_loss, learning_rate, lambda_)
        return gpu.to_cpu(loss)
    def accuracy(self,y_true, y_pred):
        y_true_cpu = gpu.to_cpu(y_true)
        y_pred_cpu = gpu.to_cpu(y_pred)
        print(classification_report(y_true_cpu.argmax(axis=1), y_pred_cpu.argmax(axis=1)))
        self.logger.log(classification_report(y_true_cpu.argmax(axis=1), y_pred_cpu.argmax(axis=1)), "INFO")
        print(f"Test set size: {y_true_cpu.shape[0]}")
        if y_true_cpu.shape[1] > 1: 
            true_labels = np.argmax(y_true_cpu, axis=1)
            pred_labels = np.argmax(y_pred_cpu, axis=1)
            accuracy = np.mean(true_labels == pred_labels)
            print(f"Test Accuracy: {accuracy:.4f}")
            self.logger.log(f"Test Accuracy: {accuracy:.4f}", "INFO")
            cm = np.zeros((y_true_cpu.shape[1], y_true_cpu.shape[1]), dtype=int)
            for t, p in zip(true_labels, pred_labels):
                cm[t, p] += 1
            print("Confusion Matrix:")
            self.logger.log("Confusion Matrix:", "INFO")
            self.logger.log(str(cm), "INFO")
            print(cm)
        else:
            mse = np.mean((y_true_cpu - y_pred_cpu)**2)
            mae = np.mean(np.abs(y_true_cpu - y_pred_cpu))
            print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")
            self.logger.log(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}", "INFO")
        if self.loss_fn:
            xp = gpu.xp
            y_true_gpu = xp.array(y_true_cpu)
            y_pred_gpu = xp.array(y_pred_cpu)
            loss = self.loss_fn(y_true_gpu, y_pred_gpu)
            print(f"Final Loss: {gpu.to_cpu(loss):.4f}")

    def save(self, path):
        logger_backup = getattr(self, 'logger', None)
        if logger_backup:
            self.logger = None
        for layer in self.layers:
            for attr in ['w', 'b', 'filters', 'biases']:
                if hasattr(layer, attr):
                    arr = getattr(layer, attr)
                    if 'cupy' in str(type(arr)):
                        try:
                            setattr(layer, attr, arr.get())
                        except Exception as e:
                            if logger_backup:
                                logger_backup.log(f"[WARNING] Could not move {attr} to CPU for layer {layer.__class__.__name__}: {e}", "WARNING")
                            setattr(layer, attr, None)
                        try:
                            gpu.clear_memory()
                        except Exception:
                            pass
        try:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            if logger_backup:
                logger_backup.log(f"✅ Model saved to {path}", "INFO")
        except Exception as e:
            print(f"[ERROR] Could not save model: {e}")
            if logger_backup:
                logger_backup.log(f"[ERROR] Could not save model: {e}", "ERROR")
        finally:
            if logger_backup:
                self.logger = logger_backup
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        for layer in model.layers:
            for attr in ['w', 'b', 'filters', 'biases']:
                if hasattr(layer, attr):
                    arr = getattr(layer, attr)
                    if gpu._has_cuda and 'numpy' in str(type(arr)):
                        try:
                            setattr(layer, attr, gpu.to_gpu(arr))
                        except Exception as e:
                            print(f"[WARNING] Could not move {attr} to GPU: {e}")
        return model