import torch

def get_input_node_attrs(batch, extra_features, pos=True):
    batch.x = extra_features(batch.x, batch.extra_feat, batch=batch)
    if pos:
        print(batch.pos)
        batch.x = torch.cat([batch.x, batch.extra_feat['positional_encoding']], dim=-1)
    return batch