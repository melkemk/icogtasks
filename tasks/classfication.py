import os
import sys, getopt
import numpy as np
import tensorflow as tf
from ngclearn.utils.config import Config
import ngclearn.utils.transform_utils as transform
from ngclearn.utils.data_utils import DataLoader
from ngclearn.utils.io_utils import deserialize

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def eval_model_classification(agent, dataset, verbose=False):
    total_misclassified = 0
    total_samples = 0

    for batch in dataset:
        _, x = batch[0]
        _, y = batch[1]
        
        if y.ndim == 1:
            y_onehot = one_hot_encode(y, num_classes=10)
        else:
            y_onehot = y
        
        x_2d = [img.reshape(28, 28) if img.ndim == 1 else img for img in x]
        batch_size = len(x_2d)
        total_samples += batch_size
        
        x_flat = np.array([img.reshape(-1) for img in x_2d])
        
        x_hat_flat = agent.settle(x_flat, calc_update=False)
        logits = x_hat_flat.numpy()
        probs = tf.nn.softmax(logits, axis=1)
        preds = tf.argmax(probs, axis=1, output_type=tf.int32).numpy()
        true_labels = tf.argmax(y_onehot, axis=1, output_type=tf.int32).numpy()
        
        batch_errors = np.sum(preds != true_labels)
        print(preds, true_labels)
        total_misclassified += batch_errors
        
        agent.clear()
        
        if verbose:
            print(f"Processed {total_samples} samples - Batch error rate: {batch_errors/batch_size:.4f}")

    overall_error_rate = total_misclassified / total_samples
    return overall_error_rate

options, remainder = getopt.getopt(sys.argv[1:], '', ["config=", "gpu_id="])
cfg_fname = None
for opt, arg in options:
    if opt == "--config":
        cfg_fname = arg.strip()

args = Config(cfg_fname)
dev_batch_size = int(args.getArg("dev_batch_size"))

xfname = args.getArg("test_xfname")
yfname = args.getArg("test_yfname")
X = transform.binarize(tf.cast(np.load(xfname), dtype=tf.float32)).numpy().reshape(-1, 28, 28)
y = np.load(yfname)
dev_set = DataLoader(design_matrices=[("z0", X), ("labels", y)], batch_size=dev_batch_size, disable_shuffle=True)

pretrained_model_path = os.path.join(os.path.dirname(__file__), "./gncn_pdh/model0.ngc")
agent = deserialize(pretrained_model_path)

error_rate = eval_model_classification(agent, dev_set, verbose=True)
print(f"\nOverall Classification Error Rate: {error_rate:.4f}")
