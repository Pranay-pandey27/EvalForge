from contextlib import contextmanager
from torch_loops.evaluation import evaluate
class Model:
    def __init__(self): self.events=[]
    def train(self): self.events.append('train')
    def eval(self): self.events.append('eval')
def test_evaluator_eval_precedes_context():
    model=Model()
    @contextmanager
    def no_grad(): model.events.append('enter'); yield
    evaluate(model,[],lambda m,b:0,no_grad)
    assert model.events==['eval','enter']
def test_evaluator_context_exits_on_error():
    events=[]
    @contextmanager
    def no_grad():
        events.append('enter')
        try: yield
        finally: events.append('exit')
    try: evaluate(Model(),[1],lambda m,b: (_ for _ in ()).throw(RuntimeError()),no_grad)
    except RuntimeError: pass
    assert events==['enter','exit']
