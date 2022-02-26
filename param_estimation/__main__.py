import os
from argparse import ArgumentParser
from param_estimation import CWD
from param_estimation.raw_data_transform import get_intermediate_data


def generate_intermediate_data():
    get_intermediate_data().to_csv(
        os.path.join(CWD, "./data/IntermediateDataForParamEstimation.csv"),
        index=False,
    )


def main():
    parser = ArgumentParser()
    parser.add_argument(
        dest="action", metavar="action", type=str, help="Action to execute"
    )
    args = parser.parse_args()
    if args.action == "generate_intermediate_data":
        generate_intermediate_data()
    else:
        pass


if __name__ == "__main__":
    main()
