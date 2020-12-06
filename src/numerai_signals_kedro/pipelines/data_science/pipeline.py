from kedro.pipeline import Pipeline, node

from .nodes import (
    predict, report, train_model, make_submit
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                train_model,
                ["train_data",
                 "feature_names"],
                "trained_model",
                tags="train_model"
            ),
            node(
                predict,
                ["trained_model",
                 "test_data",
                 "full_data_with_features",
                 "feature_names"],
                dict(
                    predicted_live_data="predicted_live_data",
                    predicted_test_data="predicted_test_data"
                ),
                tags="predict"
            ),
            node(report, "predicted_test_data", None, tags="report"),
            node(make_submit,
                 ["predicted_test_data",
                  "predicted_live_data"],
                 "submit",
                 tags="make_submit")
        ]
    )
