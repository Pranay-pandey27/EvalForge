def evaluate(model, batches, score_batch, no_grad):
    model.train()
    return [score_batch(model, batch) for batch in batches]
