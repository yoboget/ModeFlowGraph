import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str,
        default='planar',
        help="Name of the dataset. Available: qm9, qm9H, zinc, planar, sbm, enzymes"
    )

    parser.add_argument(
        "--work_type", type=str,
        default='train', help="Options: train, sample, or train_vs_test (metrics of n_samples_generation "
                              "training graphs against as many test graphs)"
    )

    parser.add_argument(
        "--wandb", type=str,
        default='on', help="If W&B is online, offline or disabled"
    )

    parser.add_argument(
        "--denoiser_dir", type=str,
        default='./wandb/run-20261019_170353-03tajjr5/files/', help="Path to the model directory"
    )

    parser.add_argument(
        "--num_modes", type=int,
        default=32, help="Number of modes M predicted by the denoiser (1: standard single-mode denoiser)"
    )

    parser.add_argument(
        "--hyp_loss", type=str, choices=['wta', 'mixture', 'tempered'],
        default='mixture', help="num_modes > 1: winner-take-all + commitment loss, the likelihood of the mixture, "
                            "or an interpolation between them with temperature --hyp_temperature"
    )

    parser.add_argument(
        "--hyp_temperature", type=float,
        default=1, help="hyp_loss tempered: 0 behaves as wta, 1 as mixture, > 1 is softer than the mixture"
    )

    parser.add_argument(
        "--mi_weight", type=float,
        default=0.0, help="num_modes > 1: weight lam of the regularizer -lam * (H(k) - H(k | G_0)), which rewards "
                         "confident responsibilities and a balanced usage of the modes"
    )

    parser.add_argument(
        "--div_weight", type=float,
        default=0.0, help="num_modes > 1: weight lam of the regularizer -lam * D, with D the divergence between the "
                         "modes' predictions per element (pushes the modes to specialize)"
    )

    parser.add_argument(
        "--hyp_selection", type=str, choices=['sample', 'average'],
        default='sample', help="num_modes > 1: sample one mode per graph at each step, or use the expectation "
                               "of the mixture (recovers the standard formulation)"
    )

    return parser.parse_args()
