from contextlib import contextmanager
from torch_loops.evaluation import evaluate
class Model:
    def __init__(self): self.mode=None
    def train(self): self.mode='train'
    def eval(self): self.mode='eval'
def test_uses_eval_mode_and_no_grad():
    events=[]
    @contextmanager
    def no_grad(): events.append('enter'); yield; events.append('exit')
    model=Model(); evaluate(model,[1],lambda m,b: events.append('score') or b,no_grad)
    assert model.mode=='eval' and events==['enter','score','exit']
def test_returns_scores():
    @contextmanager
    def no_grad(): yield
    assert evaluate(Model(),[2,3],lambda m,b:b+1,no_grad)==[3,4]
