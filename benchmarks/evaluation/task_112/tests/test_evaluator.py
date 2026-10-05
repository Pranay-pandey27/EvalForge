from torch_loops.accumulation import train_accumulated
class O:
    def __init__(self): self.zeros=0; self.steps=0
    def zero_grad(self): self.zeros+=1
    def step(self): self.steps+=1
def test_evaluator_exact_full_windows():
    o=O(); train_accumulated([1,2,3,4],o,lambda b:None,2); assert (o.zeros,o.steps)==(2,2)
def test_evaluator_empty_batches_do_nothing():
    o=O(); train_accumulated([],o,lambda b:None,2); assert (o.zeros,o.steps)==(0,0)
