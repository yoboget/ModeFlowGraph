import os
import torch
from torch import optim
from models.models import DenseGNN
from models.digress_model import GraphTransformer


def load_denoiser(config, loader, extra_features, device, prior,
                  denoiser_dir=None):
    """
    Loads the denoiser model, along with its optimizer.

    Args:
        config: Configuration object.
        loader: Data loader containing training data.
        extra_features: Object containing extra features information.
        device: Device to load models onto (e.g., 'cuda' or 'cpu').
        prior: Noise prior ('marginal' or 'masking').
        denoiser_dir: Directory to load pre-trained denoiser model from (optional).

    Returns:
        Tuple: (denoiser, optimizer)
    """
    input_sizes, output_sizes, hidden_size = get_input_sizes(config, loader, extra_features, prior)

    # --- Denoiser ---
    print("Loading Denoiser...")
    print(input_sizes)
    if config.model.architecture == 'digress':
        denoiser = GraphTransformer(config, *input_sizes, *output_sizes, hidden_size).to(device)
    else:
        denoiser = DenseGNN(config, *input_sizes, *output_sizes, hidden_size,
                            norm_out=True).to(device)
    params = list(denoiser.parameters())
    opt = optim.AdamW(params, lr=config.training.learning_rate, amsgrad=True, weight_decay=10e-12)
    n_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    print(f'Number of parameters in the encoder: {n_params}')

    if denoiser_dir is not None:
        if config.dataset in ['qm9', 'zinc', 'qm9H']:
            filename = 'best_run_fcd_ema.pt'
        else:
            filename = 'best_run_avg_ema.pt'
        denoiser = load_trained_model(denoiser, 'denoiser', denoiser_dir, filename, device)

    return denoiser, opt


def load_trained_model(model, model_name, model_dir, filename, device):
    model_path = os.path.join(model_dir, filename)
    # our own checkpoints: older ones store the metric as a numpy scalar, which weights_only=True rejects
    saved_model = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(saved_model[model_name])
    return model

def get_input_sizes(config, loader, extra_features, prior):
    n_extra_feat = extra_features.n_features
    batch = next(iter(loader['train']))
    n_node_attr = batch.x.size(-1)
    n_edge_attr = batch.edge_attr.size(1)
    nhf = config.model.nhf

    print(f"Node features: {n_node_attr}, Extra features: {n_extra_feat}")
    nnf_in = n_node_attr + n_extra_feat
    nef_in = n_edge_attr + 1

    if prior == 'masking':
        if n_node_attr > 1:
            nnf_in += 1
        nef_in += 1
    else:
        nnf_in += 2  # alpha_t, 1 - alpha_t

    if extra_features.rrwp:
        nef_in += extra_features.rrwp_k

    nnf_out = n_node_attr
    nef_out = n_edge_attr
    if n_node_attr > 1:
        nef_out += 1
    return (nnf_in, nef_in), (nnf_out, nef_out), nhf
