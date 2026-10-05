def train_accumulated(batches, optimizer, backward_batch, accumulation_steps):
    for batch in batches:
        optimizer.zero_grad()
        backward_batch(batch)
        optimizer.step()
