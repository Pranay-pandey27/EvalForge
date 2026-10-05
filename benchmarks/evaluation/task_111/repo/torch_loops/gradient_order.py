def optimize_batch(model, batch, optimizer, loss_fn):
    loss = loss_fn(model(batch))
    loss.backward()
    optimizer.zero_grad()
    optimizer.step()
    return loss
