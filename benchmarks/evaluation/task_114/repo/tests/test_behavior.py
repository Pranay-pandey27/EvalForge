from tensor_runtime.labels import prepare_class_labels
class Labels:
    def __init__(self): self.dtype='initial'
    def float(self): self.dtype='float'; return self
    def long(self): self.dtype='long'; return self
def test_labels_are_long_integers():
    labels=Labels(); assert prepare_class_labels(labels) is labels and labels.dtype=='long'
def test_conversion_happens_once():
    class L(Labels):
        def __init__(self): super().__init__(); self.calls=0
        def long(self): self.calls+=1; return super().long()
    labels=L(); prepare_class_labels(labels); assert labels.calls==1
