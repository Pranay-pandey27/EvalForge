from tensor_runtime.device import prepare
class Movable:
    def __init__(self): self.device=None
    def to(self,d): self.device=d; return self
def test_model_and_batch_share_device():
    m,x,y=Movable(),Movable(),Movable(); prepared=prepare(m,(x,y),'cuda')
    assert m.device==x.device==y.device=='cuda' and prepared==(m,(x,y))
def test_batch_tuple_type_preserved():
    m,x=Movable(),Movable(); assert isinstance(prepare(m,(x,),'cpu')[1],tuple)
