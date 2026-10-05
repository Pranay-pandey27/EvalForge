def train_epoch(model, batches, update_batch):
    losses = []
    for batch in batches:
        losses.append(update_batch(model, batch))
    return losses
