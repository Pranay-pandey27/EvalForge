import pytest
from torch_loops.accumulation import train_accumulated
class O:
    def __init__(self,e): self.e=e
    def zero_grad(self): self.e.append('zero')
    def step(self): self.e.append('step')
def test_two_batch_windows():
    e=[]; train_accumulated([1,2,3],O(e),lambda b:e.append(f'b{b}'),2)
    assert e==['zero','b1','b2','step','zero','b3','step']
def test_rejects_nonpositive_window():
    with pytest.raises(ValueError): train_accumulated([],O([]),lambda b:None,0)
