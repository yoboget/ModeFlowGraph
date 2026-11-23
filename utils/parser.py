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
        default='sample', help="Options: train, sample, or train_vs_test (metrics of n_samples_generation "
                              "training graphs against as many test graphs)"
    )

    parser.add_argument(
        "--wandb", type=str,
        default='on', help="If W&B is online, offline or disabled"
    )

    parser.add_argument(
        "--denoiser_dir", type=str,
        default='./checkpoints/zinc/dirichlet6_state-a73worv0/files/', help="Path to the model directory"
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
        "--nfe", type=int, nargs='+',
        default=[128], help="Numbers of sampling steps T (NFE) to sample with, e.g. --nfe 32 64 128 (work_type sample "
                           "and sample_tuned_sweep.py). Default: model.T of the config"
    )

    parser.add_argument(
        "--lambd", type=float,
        default=0., help="'tuned' sampling ('dirichlet' prior): how the per-class quantile maps are combined. < 0 "
                          "(-1, lambda = infinity): average weighted by the prediction p(z_1 | G_t) (deterministic), "
                          "0: one class drawn from p(z_1 | G_t), > 0: Dirichlet(lambda p(z_1 | G_t)) mixing weights"
    )

    parser.add_argument(
        "--eta", type=float,
        default=1., help="'tuned' sampling ('dirichlet' prior), in [0, 1]: how much each conditional map is resampled "
                         "from its target Dirichlet. 0: the deterministic quantile map, 1: a fresh conditional sample, "
                         "in between: partial resampling (rho = 1 - eta). eta = 0 with lambd < 0 is 'quantile_map'"
    )

    parser.add_argument(
        "--pgd", action=argparse.BooleanOptionalAction,
        default=True, help="Also compute the PolyGraph Discrepancy (eval/pgd.py, Krimmel et al. 2025): a TabPFN "
                           "classifier's lower bound on the JS distance between generated and reference graphs. Only when sampling "
                           "(work_type sample / train_vs_test), never during training"
    )

    parser.add_argument(
        "--fast_eval", action=argparse.BooleanOptionalAction,
        default=False, help="work_type sample: evaluate as during training, i.e. log.n_val_samples graphs against the "
                            "validation set, without NSPDK and PGD"
    )

    parser.add_argument(
        "--dirichlet_scale", type=float,
        default=6, help="prior 'dirichlet': scale of the schedule, alpha(t) = -scale log(1 - t) ('cosine') or "
                         "scale t ('linear') in z_t ~ Dir(1 + alpha(t) z_1), the exponent in "
                         "alpha(t) = 1 - (1 - t)^scale ('sparse')"
    )

    parser.add_argument(
        "--input", type=str, choices=['state', 'sufficient'],
        default='state', help="Denoiser input: the noisy 'state' with the noise level, or the 'sufficient' statistics "
                              "phi(z_t, t)_b ∝ q(z_t | z_1 = b) of each node / edge, the same for every noising process "
                              "(see utils/model_input.py)"
    )

    parser.add_argument(
        "--time_input", action=argparse.BooleanOptionalAction,
        default=False, help="--input sufficient: also give the noise level to the denoiser (ablation)"
    )

    parser.add_argument(
        "--schedule", type=str, choices=['cosine', 'linear', 'sparse'],
        default='cosine', help="Noise schedule. 'cosine': alpha_bar(t) = cos(pi (1 - t) / 2)^c (discrete priors), "
                               "alpha(t) = -scale log(1 - t) (dirichlet). 'linear': alpha_bar(t) = t (discrete priors), "
                               "alpha(t) = scale t (dirichlet, alpha_max = 1 + scale as in Dirichlet flow matching). "
                               "'sparse' (dirichlet only): z_t ~ Dir(alpha(t) z_1 + (1 - alpha(t)) 1) with "
                               "alpha(t) = 1 - (1 - t)^scale, uniform at t = 0 and exactly z_1 at t = 1 (scale ~2: "
                               "information about z_1 spread evenly over t; larger: earlier)"
    )

    parser.add_argument(
        "--schedule_c", type=float,
        default=2, help="Exponent c of the cosine schedule alpha_bar(t) = cos(pi (1 - t) / 2)^c (2: standard cosine)"
    )

    parser.add_argument(
        "--hyp_selection", type=str, choices=['sample', 'average'],
        default='sample', help="num_modes > 1: sample one mode per graph at each step, or use the expectation "
                               "of the mixture (recovers the standard formulation)"
    )

    return parser.parse_args()
