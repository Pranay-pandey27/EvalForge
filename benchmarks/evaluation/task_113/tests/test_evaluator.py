from tensor_runtime.device import prepare
class T:
    def __init__(self): self.calls=[]
    def to(self,d): self.calls.append(d); return self
def test_evaluator_each_object_moves_once():
    objects=[T(),T(),T()]; prepare(objects[0],tuple(objects[1:]),'mps'); assert [o.calls for o in objects]==[['mps'],['mps'],['mps']]
def test_evaluator_empty_batch_supported():
    model=T(); assert prepare(model,tuple(),'cpu')==(model,tuple())
