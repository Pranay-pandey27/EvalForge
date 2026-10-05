from torch_loop.training import train_batch

class Optimizer:
    def __init__(self): self.zero_calls = 0; self.steps = 0
    def zero_grad(self): self.zero_calls += 1
    def step(self): self.steps += 1
class Loss:
    def backward(self): pass

def test_evaluator_clears_once_per_batch():
    optimizer = Optimizer()
    train_batch(lambda x: x, 0, 0, optimizer, lambda p, t: Loss())
    assert (optimizer.zero_calls, optimizer.steps) == (1, 1)

def test_evaluator_model_receives_input():
    seen = []
    train_batch(lambda x: seen.append(x) or x, 7, 0, Optimizer(), lambda p, t: Loss())
    assert seen == [7]
