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
        default='train', help="Options: train, sample, or train_vs_test (metrics of n_samples_generation "
                              "training graphs against as many test graphs)"
    )

    parser.add_argument(
        "--wandb", type=str,
        default='on', help="If W&B is online, offline or disabled"
    )

    parser.add_argument(
        "--denoiser_dir", type=str,
        default=None, help="Path to the model directory"
    )

    parser.add_argument(
        "--num_modes", type=int,
        default=1, help="Number of modes M predicted by the denoiser (1: standard single-mode denoiser)"
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
        "--training_method", type=str, choices=['denoiser', 'cavity'],
        default='denoiser', help="What the denoiser outputs: 'denoiser' p(z_1 | G_t) or 'cavity' p(z_1 | G_t without "
                                 "z_t), the prediction without the element's own noisy value. Both are trained with "
                                 "the cross-entropy (the cavity through its conversion to the denoiser); must match "
                                 "the checkpoint when sampling"
    )

    parser.add_argument(
        "--sampling_method", type=str, choices=['denoiser', 'cavity'],
        default='denoiser', help="'dif' sampling: posterior sum_z1 q(z_u | z_t, z_1) p(z_1 | G_t) with the denoiser, or "
                                 "the plug-in q(z_t | z_u) sum_z1 c(z_1) q(z_u | z_1) with the cavity; the output is "
                                 "converted from --training_method when they differ (sid / dfm always use the "
                                 "denoiser)"
    )

    parser.add_argument(
        "--greedy_last_step", type=bool,
         default=True, help="Last sampling step (sid / dif / dfm): take the most likely class of each node / edge under the "
             "prediction p(z_1 | G_t) instead of sampling it"
    )

    parser.add_argument(
        "--temperature", type=float,
        default=1, help="Sampling temperature tau: the prediction p(z_1 | G_t) is replaced by "
                         "p(z_1 | G_t)^(1 / tau), renormalized (< 1 sharper, > 1 flatter, 1: the model), for all "
                         "samplers"
    )

    parser.add_argument(
        "--pgd", action=argparse.BooleanOptionalAction,
        default=True, help="Also compute the PolyGraph Discrepancy (eval/pgd.py, Krimmel et al. 2025): a TabPFN "
                           "classifier's lower bound on the JS distance between generated and reference graphs. Only when sampling "
                           "(work_type sample / train_vs_test), never during training"
    )

    parser.add_argument(
        "--dirichlet_scale", type=float,
        default=6, help="prior 'dirichlet': scale of the concentration alpha(t) = -scale log(1 - t) of the path "
                         "z_t ~ Dir(1 + alpha(t) z_1)"
    )

    parser.add_argument(
        "--input", type=str, choices=['state', 'sufficient'],
        default='sufficient', help="Denoiser input: the noisy 'state' with the noise level, or the 'sufficient' statistics "
                              "phi(z_t, t)_b ∝ q(z_t | z_1 = b) of each node / edge, the same for every noising process "
                              "(see utils/model_input.py)"
    )

    parser.add_argument(
        "--time_input", action=argparse.BooleanOptionalAction,
        default=False, help="--input sufficient: also give the noise level to the denoiser (ablation)"
    )

    parser.add_argument(
        "--schedule_c", type=float,
        default=2, help="Exponent c of the noise schedule alpha_bar(t) = cos(pi t / 2)^c (2: standard cosine)"
    )

    parser.add_argument(
        "--hyp_selection", type=str, choices=['sample', 'average'],
        default='sample', help="num_modes > 1: sample one mode per graph at each step, or use the expectation "
                               "of the mixture (recovers the standard formulation)"
    )

    return parser.parse_args()
