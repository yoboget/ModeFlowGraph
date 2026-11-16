from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

# PyG >= 2.4 makes BaseTransform.forward abstract (__call__ calls it); older versions only have __call__.
# The transforms implement forward and keep __call__, so they work with both.


class Qm9Transform(BaseTransform):
    def __init__(self):
        super().__init__()

    def forward(self, data: Data):
        data.x = data.x[..., :5]
        return data

    def __call__(self, data: Data):
        return self.forward(data)


class Qm9HTransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def forward(self, data: Data):
        data.x = data.x[..., :5]
        return data

    def __call__(self, data: Data):
        return self.forward(data)
