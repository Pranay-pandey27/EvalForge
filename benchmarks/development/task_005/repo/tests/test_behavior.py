from torch_loop.training import train_batch

class Recorder:
    def __init__(self, events): self.events = events
    def zero_grad(self): self.events.append('zero_grad')
    def step(self): self.events.append('step')

class Loss:
    def __init__(self, events): self.events = events
    def backward(self): self.events.append('backward')

def test_zero_grad_precedes_backward_and_step():
    events = []
    model = lambda x: x
    loss = Loss(events)
    train_batch(model, 1, 1, Recorder(events), lambda p, t: loss)
    assert events == ['zero_grad', 'backward', 'step']

def test_returns_loss_object():
    events, loss = [], Loss([])
    assert train_batch(lambda x: x, 1, 1, Recorder(events), lambda p, t: loss) is loss
