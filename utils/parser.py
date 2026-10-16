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
        default=None, help="Path to the model directory"
    )

    return parser.parse_args()
