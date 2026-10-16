from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

class Qm9Transform(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data: Data):
        data.x = data.x[..., :5]
        return data

class Qm9HTransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data: Data):
        data.x = data.x[..., :5]
        return data
