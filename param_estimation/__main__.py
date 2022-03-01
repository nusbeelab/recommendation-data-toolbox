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


def generate_estimated_parameters(experiment_number: int, model: str):
    intermediate_data_filename = (
        f"IntermediateDataExperiment{experiment_number}.csv"
    )
    estimate_params_by_subjects(intermediate_data_filename, model).to_csv(
        os.path.join(
            CWD,
            f"./data/mle{model.capitalize()}_Experiment{experiment_number}.csv",
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
    parser.add_argument(
        "--model", dest="model", type=str, choices=["crra1param", "crra3params"]
    )
    args = parser.parse_args()
    if args.action == "generate_intermediate_data":
        generate_intermediate_data(args.experiment_number)
    elif args.action == "estimate_params":
        generate_estimated_parameters(args.experiment_number, args.model)


if __name__ == "__main__":
    main()
