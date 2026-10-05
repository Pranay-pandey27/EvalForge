import numpy as np
def spatial_channel_mean(images):
    values = np.asarray(images, dtype=float)
    return values.mean(axis=(1, 2))
