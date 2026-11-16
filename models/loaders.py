import os
import warnings
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
        prior: Noise prior ('marginal', 'masking' or 'dirichlet').
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
        denoiser = load_trained_model(denoiser, 'denoiser', denoiser_dir, filename, device,
                                      meta=checkpoint_meta(config))

    return denoiser, opt


def load_trained_model(model, model_name, model_dir, filename, device, meta=None):
    """Loads the weights of `model`; with `meta` (see checkpoint_meta), first checks that the checkpoint was trained
    with a compatible setup."""
    model_path = os.path.join(model_dir, filename)
    # our own checkpoints: older ones store the metric as a numpy scalar, which weights_only=True rejects
    saved_model = torch.load(model_path, map_location=device, weights_only=False)
    if meta is not None:
        check_meta(saved_model.get('meta'), meta, model_path)
    model.load_state_dict(saved_model[model_name])
    return model


def checkpoint_meta(config):
    """The training setup saved with each checkpoint, checked against the configuration when it is loaded."""
    m = config.model
    return {'prior': m.prior, 'input': m.get('input', 'state'), 'time_input': bool(m.get('time_input', False)),
            'num_modes': m.get('num_modes', 1), 'training_method': m.get('training_method', 'denoiser'),
            'schedule_c': m.get('schedule_c', 2.), 'dirichlet_scale': m.get('dirichlet_scale', 3.)}


def compatibility_keys(meta):
    """
    What a checkpoint must share with the configuration to be sampled. With input 'sufficient' and no time input,
    the denoiser does not depend on the noising process. Otherwise it gets the noise level as input: the prior must
    match, and for 'dirichlet' the scale too (the input t means alpha(t) = -scale log(1 - t)). For 'marginal', the
    input is alpha_bar itself, so schedule_c only changes the sampling grid and may differ.
    """
    keys = ['input', 'time_input', 'num_modes', 'training_method']
    if meta['input'] == 'state' or meta['time_input']:
        keys.append('prior')
        if meta['prior'] == 'dirichlet':
            keys.append('dirichlet_scale')
    return keys


def check_meta(saved, current, path):
    if saved is None:
        # checkpoints from before the metadata: all trained with the 'state' input
        warnings.warn(f'{path} has no training metadata: only the model input is checked (assumed \'state\'). '
                      f'Make sure prior, schedule and num_modes match the training.')
        saved, keys = {'input': 'state'}, ['input']
    else:
        keys = compatibility_keys(current)
    mismatch = {k: (saved.get(k), current[k]) for k in keys if saved.get(k) != current[k]}
    if mismatch:
        details = ', '.join(f'{k}: checkpoint {v[0]} vs config {v[1]}' for k, v in mismatch.items())
        raise ValueError(f'{path} was trained with a different setup ({details}).')


def get_input_sizes(config, loader, extra_features, prior):
    n_extra_feat = extra_features.n_features
    batch = next(iter(loader['train']))
    n_node_attr = batch.x.size(-1)
    n_edge_attr = batch.edge_attr.size(1)
    nhf = config.model.nhf

    print(f"Node features: {n_node_attr}, Extra features: {n_extra_feat}")
    nnf_in = n_node_attr + n_extra_feat
    nef_in = n_edge_attr + 1

    if config.model.get('input', 'state') == 'sufficient':
        # phi over the data classes; the noise level only as an ablation (none for 'masking', see add_noise_level)
        if config.model.get('time_input', False) and prior != 'masking':
            nnf_in += 2
    elif prior == 'masking':
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
