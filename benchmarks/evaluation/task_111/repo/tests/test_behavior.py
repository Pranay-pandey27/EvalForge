from torch_loops.gradient_order import optimize_batch
class Optimizer:
    def __init__(self,e): self.e=e
    def zero_grad(self): self.e.append('zero')
    def step(self): self.e.append('step')
class Loss:
    def __init__(self,e): self.e=e
    def backward(self): self.e.append('backward')
def test_gradient_operation_order():
    e=[]; optimize_batch(lambda x:e.append('forward') or x,1,Optimizer(e),lambda x:Loss(e))
    assert e==['zero','forward','backward','step']
def test_returns_loss():
    e=[]; loss=Loss(e)
    assert optimize_batch(lambda x:x,1,Optimizer(e),lambda x:loss) is loss
