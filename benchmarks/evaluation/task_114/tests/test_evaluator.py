from tensor_runtime.labels import prepare_class_labels
class L:
    def __init__(self): self.used=[]
    def float(self): self.used.append('float'); return self
    def long(self): self.used.append('long'); return self
def test_evaluator_never_calls_float():
    labels=L(); prepare_class_labels(labels); assert labels.used==['long']
def test_evaluator_identity_returned():
    labels=L(); assert prepare_class_labels(labels) is labels
