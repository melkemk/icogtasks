import sys
import numpy as np
import tensorflow as tf
from ngclearn.utils.config import Config
from ngclearn.utils.io_utils import deserialize
from ngclearn.utils.transform_utils import binarize

if len(sys.argv) < 2:
    print("Usage: python simple_classification.py config.yaml")
    sys.exit(1)

# Load data and config
cfg = Config(sys.argv[1])
X_train = np.load(cfg.getArg("train_xfname"))[:10000]
raw_Y_train = np.load(cfg.getArg("train_yfname"))[:10000]
X_test  = np.load(cfg.getArg("test_xfname"))
raw_Y_test = np.load(cfg.getArg("test_yfname"))

# Preprocess: binarize and flatten
X_train = binarize(tf.cast(X_train, tf.float32)).numpy().reshape(len(X_train), -1)
X_test  = binarize(tf.cast(X_test,  tf.float32)).numpy().reshape(len(X_test),  -1)

# One-hot encode labels only if needed
if raw_Y_train.ndim > 1:
    Y_train = raw_Y_train
else:
    Y_train = tf.keras.utils.to_categorical(raw_Y_train)

if raw_Y_test.ndim > 1:
    Y_test = raw_Y_test
else:
    Y_test = tf.keras.utils.to_categorical(raw_Y_test)

# Load pretrained NGC model
agent = deserialize(cfg.getArg("model_fname"))
batch_sz = agent.ngc_model.batch_size

# Trim or pad data to full batches
def trim_to_batches(X, Y, bs):
    n_full = (len(X) // bs) * bs
    if n_full > 0:
        return X[:n_full], Y[:n_full]
    # pad if fewer than one batch
    pad = bs - len(X)
    X_pad = np.concatenate([X, X[:pad]], axis=0)
    Y_pad = np.concatenate([Y, Y[:pad]], axis=0)
    return X_pad, Y_pad

X_train, Y_train = trim_to_batches(X_train, Y_train, batch_sz)
X_test,  Y_test  = trim_to_batches(X_test,  Y_test,  batch_sz)

# Extract latent features
def extract_latents(X):
    parts = []
    total = len(X)
    for i in range(0, total, batch_sz):
        batch = X[i:i+batch_sz]
        agent.settle(batch, calc_update=False)
        z = agent.ngc_model.extract("z3", "phi(z)")
        parts.append(z.numpy())
        agent.clear()
        print(f"Extracted latents: {min(i+batch_sz, total)}/{total}", end='\r')
    print()
    return np.vstack(parts)

Z_train = extract_latents(X_train)
Z_test  = extract_latents(X_test)

# Build and train classifier
num_classes = Y_train.shape[1]
model = tf.keras.Sequential([tf.keras.layers.Dense(num_classes, activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(Z_train, Y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate and print error
_, acc = model.evaluate(Z_test, Y_test, verbose=1)
print(f"Classification error: {1-acc:.4f}") 
