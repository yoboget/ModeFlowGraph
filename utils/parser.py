import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str,
        default='zinc',
        help="Name of the dataset. Available: qm9, qm9H, zinc, planar, sbm, enzymes"
    )

    parser.add_argument(
        "--work_type", type=str,
        default='train', help="Options: train or sample"
    )

    parser.add_argument(
        "--wandb", type=str,
        default='on', help="If W&B is online, offline or disabled"
    )

    parser.add_argument(
        "--denoiser_dir", type=str,
        default='./wandb/run-20261017_122547-dztbwtog/files/', help="Path to the model directory"
    )

    parser.add_argument(
        "--num_modes", type=int,
        default=32, help="Number of modes M predicted by the denoiser (1: standard single-mode denoiser)"
    )

    parser.add_argument(
        "--hyp_loss", type=str, choices=['wta', 'mixture'],
        default='mixture', help="num_modes > 1: winner-take-all + commitment loss, or the likelihood of the mixture"
    )

    parser.add_argument(
        "--mi_weight", type=float,
        default=0., help="num_modes > 1: weight lam of the regularizer -lam * (H(k) - H(k | G_0)), which rewards "
                         "confident responsibilities and a balanced usage of the modes"
    )

    parser.add_argument(
        "--div_weight", type=float,
        default=0., help="num_modes > 1: weight lam of the regularizer -lam * D, with D the divergence between the "
                         "modes' predictions per element (pushes the modes to specialize)"
    )

    parser.add_argument(
        "--hyp_selection", type=str, choices=['sample', 'average'],
        default='sample', help="num_modes > 1: sample one mode per graph at each step, or use the expectation "
                               "of the mixture (recovers the standard formulation)"
    )

    return parser.parse_args()
