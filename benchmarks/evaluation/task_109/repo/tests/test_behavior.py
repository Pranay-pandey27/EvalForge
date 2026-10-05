from torch_loops.train_mode import train_epoch
class Model:
    def __init__(self): self.train_calls = 0
    def train(self): self.train_calls += 1
def test_model_enters_train_mode_once():
    model = Model(); train_epoch(model, [1,2], lambda m,b: b)
    assert model.train_calls == 1
def test_returns_one_loss_per_batch():
    assert train_epoch(Model(), [1,2], lambda m,b: b * 2) == [2,4]
