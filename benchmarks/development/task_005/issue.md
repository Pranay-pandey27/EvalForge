`train_batch` accumulates stale gradients because the optimizer is never cleared. Follow standard PyTorch batch ordering.
