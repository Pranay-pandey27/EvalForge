def prepare(model, batch, device):
    return model.to(device), batch
