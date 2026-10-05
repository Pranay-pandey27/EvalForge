from torch_loops.train_mode import train_epoch
class Model:
    def __init__(self): self.events=[]
    def train(self): self.events.append('train')
def test_evaluator_mode_set_before_first_update():
    model=Model(); train_epoch(model,[1],lambda m,b: m.events.append('update') or 0)
    assert model.events == ['train','update']
def test_evaluator_empty_epoch_still_sets_mode():
    model=Model(); assert train_epoch(model,[],lambda m,b: 0)==[] and model.events==['train']
