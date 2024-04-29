from argparse import ArgumentParser


def add_argument(parser: ArgumentParser):
    parser.add_argument(
        "--experiment_name",
        default="new",
        type=str,
    )
    parser.add_argument(
        "--loss",
        default="dice",
        choices=["dice", "bce", "focal", "diceCE"],
        type=str,
    )
    parser.add_argument(
        "--diceCE_lambda_dice",
        default=1,
        type=float,
    )
    parser.add_argument(
        "--diceCE_lambda_CE",
        default=1,
        type=float,
    )
