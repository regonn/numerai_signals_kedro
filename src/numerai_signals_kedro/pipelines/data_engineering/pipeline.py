from kedro.pipeline import Pipeline, node

from .nodes import (
    download_yfinance,
    make_features,
    make_input
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                download_yfinance,
                "ticker_map",
                "full_data",
                tags="download_yfinance"
            ),
            node(
                make_features,
                "full_data",
                dict(
                    full_data_with_features="full_data_with_features",
                    feature_names="feature_names"
                ),
                tags="make_features"
            ),
            node(
                make_input,
                ["full_data_with_features",
                 "feature_names",
                 "historical_targets"],
                dict(
                    train_data="train_data",
                    test_data="test_data",
                ),
                tags="make_input"
            )
        ]
    )
