from tensor_device.batch import move_batch

class Tensor:
    def __init__(self, name): self.name = name; self.devices = []
    def to(self, device): self.devices.append(device); return self

def test_moves_inputs_and_targets():
    x, y = Tensor('x'), Tensor('y')
    move_batch(x, y, 'cuda')
    assert x.devices == ['cuda'] and y.devices == ['cuda']

def test_preserves_tuple_order():
    x, y = Tensor('x'), Tensor('y')
    assert move_batch(x, y, 'cpu') == (x, y)
