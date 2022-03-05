import os
from argparse import ArgumentParser
from param_estimation import CWD
from param_estimation.param_estimation import estimate_params
from param_estimation.raw_data_transform import get_intermediate_data


def generate_intermediate_data(experiment_number: int):
    results = get_intermediate_data(experiment_number)
    output_filepath = (
        f"./data/IntermediateDataExperiment{experiment_number}.csv"
    )
    results.to_csv(
        os.path.join(CWD, output_filepath),
        index=False,
    )


def snakecase_to_camelcase(snakecase: str):
    words = snakecase.split("_")
    words = [
        word.capitalize() if i > 0 else word for i, word in enumerate(words)
    ]
    return "".join(words)


def generate_estimated_parameters(
    experiment_number: int,
    model: str,
    is_neg_domain_included: bool,
    is_with_constraints: bool,
    is_per_subject: bool,
):
    intermediate_data_filename = (
        f"IntermediateDataExperiment{experiment_number}.csv"
    )

    results = estimate_params(
        intermediate_data_filename,
        model,
        is_neg_domain_included,
        is_with_constraints,
        is_per_subject,
    )
    experiment_label = f"experiment{experiment_number}"
    model_label = f"{snakecase_to_camelcase(model)}Model"
    domain_label = "realDomain" if is_neg_domain_included else "gainDomainOnly"
    constraints_label = (
        "withConstraints" if is_with_constraints else "withoutConstraints"
    )
    per_subject_label = (
        "perSubject" if is_per_subject else "representativeSubject"
    )
    output_filepath = f"./data/mle_{experiment_label}_{model_label}_{domain_label}_{constraints_label}_{per_subject_label}.csv"
    results.to_csv(os.path.join(CWD, output_filepath))


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
        "--model",
        dest="model",
        type=str,
        choices=[
            "expected_utility",
            "prospect_theory",
            "cumulative_prospect_theory",
        ],
    )
    parser.add_argument(
        "--include-neg-domain", dest="isNegDomainIncluded", action="store_true"
    )
    parser.add_argument(
        "--per-subject", dest="isPerSubject", action="store_true"
    )
    parser.add_argument(
        "--with-constraints", dest="isWithConstraints", action="store_true"
    )
    args = parser.parse_args()
    if args.action == "generate_intermediate_data":
        generate_intermediate_data(args.experiment_number)
    elif args.action == "estimate_params":
        generate_estimated_parameters(
            args.experiment_number,
            args.model,
            args.isNegDomainIncluded,
            args.isWithConstraints,
            args.isPerSubject,
        )


if __name__ == "__main__":
    main()
