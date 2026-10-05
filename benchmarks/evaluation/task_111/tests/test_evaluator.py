from torch_loops.gradient_order import optimize_batch
class O:
    def __init__(self): self.zero=0; self.steps=0
    def zero_grad(self): self.zero+=1
    def step(self): self.steps+=1
class L:
    def backward(self): pass
def test_evaluator_one_clear_and_step():
    o=O(); optimize_batch(lambda x:x,0,o,lambda x:L()); assert (o.zero,o.steps)==(1,1)
def test_evaluator_batch_reaches_model():
    seen=[]; optimize_batch(lambda x:seen.append(x) or x,9,O(),lambda x:L()); assert seen==[9]
