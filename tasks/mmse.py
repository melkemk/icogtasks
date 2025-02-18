import os
import sys, getopt
import numpy as np
import tensorflow as tf
from ngclearn.utils.config import Config
import ngclearn.utils.transform_utils as transform
from ngclearn.utils.data_utils import DataLoader
from ngclearn.utils.io_utils import deserialize
import jax.numpy as jnp

def mask_image(image, mask_side="left"):
    img = image.copy()
    if len(img.shape) == 1:
        img = img.reshape(28, 28)
    w = img.shape[1]
    if mask_side == "left":
        img[:, :w//2] = 0
    else:
        img[:, w//2:] = 0
    return img

def mse(mu, x):
    diff = mu - x
    se = tf.math.square(diff)
    return tf.reduce_sum(se, axis=[1,2])

def masked_mean_squared_error(Y_pred, Y_true, mask):
    mask_batch = np.expand_dims(mask, axis=0)
    Y_pred_tf = tf.convert_to_tensor(Y_pred, dtype=tf.float32)
    Y_true_tf = tf.convert_to_tensor(Y_true, dtype=tf.float32)
    mask_tf = tf.convert_to_tensor(mask_batch, dtype=tf.float32)
    masked_pred = Y_pred_tf * mask_tf
    masked_true = Y_true_tf * mask_tf
    mse_per_sample = mse(masked_pred, masked_true)
    mask_sum = tf.reduce_sum(mask_tf, axis=[1,2])
    mse_normalized = mse_per_sample / mask_sum
    return tf.reduce_mean(mse_normalized).numpy() 

def eval_modelmmse(agent, dataset, mask_side="left", verbose=False):
    total_MMSE = 0.0
    N = 0

    for batch in dataset:
        _, x = batch[0]
        x_2d = [img.reshape(28, 28) if len(img.shape) == 1 else img for img in x]
        masked_images = np.array([mask_image(img, mask_side) for img in x_2d])
        batch_size = masked_images.shape[0]
        N += batch_size
        masked_images_flat = masked_images.reshape(batch_size, -1)
        x_hat_flat = agent.settle(masked_images_flat, calc_update=False)
        x_hat = x_hat_flat.numpy().reshape(batch_size, 28, 28)
        sample_img = x_2d[0]
        w = sample_img.shape[1]
        mask_np = np.zeros_like(sample_img)
        if mask_side == "left":
            mask_np[:, :w//2] = 1
        else:
            mask_np[:, w//2:] = 1
        batch_MMSE = 0.0
        for i in range(batch_size):
            batch_MMSE += masked_mean_squared_error(
                np.expand_dims(x_hat[i], axis=0), 
                np.expand_dims(x_2d[i], axis=0), 
                mask_np
            )
        total_MMSE += batch_MMSE
        agent.clear()

        if verbose:
            print("\rMMSE {:.4f} over {} samples...".format(total_MMSE / N, N), end="")

    if verbose:
        print()

    avg_MMSE = total_MMSE / N
    return avg_MMSE

options, remainder = getopt.getopt(sys.argv[1:], '', ["config=", "gpu_id=", "n_trials="])
cfg_fname = None
for opt, arg in options:
    if opt == "--config":
        cfg_fname = arg.strip()

args = Config(cfg_fname)
batch_size = int(args.getArg("batch_size"))
dev_batch_size = int(args.getArg("dev_batch_size"))

xfname = args.getArg("dev_xfname")
X = transform.binarize(tf.cast(np.load(xfname), dtype=tf.float32)).numpy()
dev_set = DataLoader(design_matrices=[("z0", X)], batch_size=dev_batch_size, disable_shuffle=True)

pretrained_model_path = os.path.join(os.path.dirname(__file__), "./gncn_pdh/model0.ngc")
agent = deserialize(pretrained_model_path)

avg_MMSE = eval_modelmmse(agent, dev_set, mask_side="left", verbose=True)
print("\nAverage Masked Mean Squared Error (MMSE): {:.4f}".format(avg_MMSE)) 
