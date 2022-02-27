import os
from argparse import ArgumentParser
from param_estimation import CWD
from param_estimation.param_estimation import estimate_params_by_subjects
from param_estimation.raw_data_transform import get_intermediate_data


def generate_intermediate_data(experiment_number: int):
    get_intermediate_data(experiment_number).to_csv(
        os.path.join(
            CWD, f"./data/IntermediateDataExperiment{experiment_number}.csv"
        ),
        index=False,
    )


def generate_estimated_parameters(filename: str):
    estimate_params_by_subjects(filename, "1param").to_csv(
        os.path.join(
            CWD, f"./data/EstimatedParameters__{filename.split('.')[0]}.csv"
        )
    )


def main():
    parser = ArgumentParser()
    parser.add_argument(dest="action", type=str, help="Command to execute")
    parser.add_argument(
        "--experiment-number",
        dest="experiment_number",
        type=int,
        choices=[1, 2, 3],
        help="Experiment number",
    )
    args = parser.parse_args()
    if args.action == "generate_intermediate_data":
        generate_intermediate_data(args.experiment_number)
    elif args.action == "estimate_params":
        generate_estimated_parameters(
            f"IntermediateDataExperiment{args.experiment_number}.csv"
        )


if __name__ == "__main__":
    main()
