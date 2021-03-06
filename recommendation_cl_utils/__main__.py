from argparse import ArgumentParser
from recommendation_cl_utils.mock_data import get_mock_data, get_mock_split_data

from recommendation_cl_utils.utils import (
    get_fullpath_to_datafile,
    snakecase_to_camelcase,
)
from recommendation_cl_utils.raw_data_transform import get_intermediate_data
from recommendation_cl_utils.param_estimation import estimate_params
from recommendation_cl_utils.rec_benchmarking.benchmark import benchmark_model


def generate_intermediate_data(experiment_number: int):
    results = get_intermediate_data(experiment_number)
    output_filename = "IntermediateDataExperiment{experiment_number}.csv"
    results.to_csv(
        get_fullpath_to_datafile(output_filename),
        index=False,
    )


def generate_estimated_parameters(
    experiment_number: int,
    model: str,
    is_neg_domain_included: bool,
    is_with_constraints: bool,
    is_per_subject: bool,
):

    results = estimate_params(
        experiment_number,
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
    output_filename = f"mle_{experiment_label}_{model_label}_{domain_label}_{constraints_label}_{per_subject_label}.csv"
    results.to_csv(get_fullpath_to_datafile(output_filename))


def generate_mock_experiment_data(dataset: str, is_split: bool):
    if is_split:
        preexperiment_data, experiment_data = get_mock_split_data(dataset)
        preexperiment_data.to_csv(
            get_fullpath_to_datafile(f"MockPreexperimentData_{dataset}.csv"),
            index=False,
        )
        experiment_data.to_csv(
            get_fullpath_to_datafile(f"MockExperimentData_{dataset}.csv"),
            index=False,
        )
    else:
        get_mock_data(dataset).to_csv(
            get_fullpath_to_datafile(f"Data_{dataset}.csv"), index=False
        )


def generate_benchmark_results(model: str, dataset: str):
    benchmark_model(model, dataset).to_csv(
        get_fullpath_to_datafile(f"benchmark_{model}_{dataset}.csv"),
        index=False,
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
        "--model",
        dest="model",
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

    parser.add_argument("--dataset", dest="dataset")
    parser.add_argument("--split", dest="isSplit", action="store_true")

    parser.add_argument("--rec-model", dest="recModel")
    parser.add_argument("--experiment-filename", dest="experimentFilename")
    parser.add_argument(
        "--preexperiment-filename", dest="preexperimentFilename"
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
    elif args.action == "generate_mock_data":
        generate_mock_experiment_data(args.dataset, args.isSplit)
    elif args.action == "benchmark":
        generate_benchmark_results(args.recModel, args.dataset)


if __name__ == "__main__":
    main()
