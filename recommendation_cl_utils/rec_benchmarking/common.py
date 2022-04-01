import pandas as pd


def convert_subj_data_to_rating_vector(subj_experiment_data: pd.DataFrame):
    decisions = subj_experiment_data["decision"].to_list()
    problem_ids = subj_experiment_data["problem_id"].to_list()
    problem_ids, decisions = zip(*sorted(zip(problem_ids, decisions)))
    return pd.Series(
        decisions, index=problem_ids, name=subj_experiment_data.index[0]
    )


def get_rating_matrix_df(df: pd.DataFrame):
    return df.groupby("subj_id").apply(convert_subj_data_to_rating_vector)
