def train_batch(model, inputs, targets, optimizer, loss_fn):
    predictions = model(inputs)
    loss = loss_fn(predictions, targets)
    loss.backward()
    optimizer.step()
    return loss
