from tensor_device.batch import move_batch

class Tensor:
    def __init__(self): self.calls = 0
    def to(self, device): self.calls += 1; self.device = device; return self

def test_evaluator_moves_each_tensor_exactly_once():
    x, y = Tensor(), Tensor()
    move_batch(x, y, 'mps')
    assert (x.calls, y.calls) == (1, 1)

def test_evaluator_uses_requested_device():
    x, y = Tensor(), Tensor()
    move_batch(x, y, 'accelerator:1')
    assert x.device == y.device == 'accelerator:1'
