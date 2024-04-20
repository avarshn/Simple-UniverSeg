from argparse import ArgumentParser


def add_argument(parser: ArgumentParser):
    parser.add_argument(
        "--experiment_name",
        default="new",
        type=str,
    )
