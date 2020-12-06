from typing import Any, Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta, FR

TARGET_NAME = 'target'
PREDICTION_NAME = 'signal'


def train_model(train_data: pd.DataFrame, feature_names: List[str]):
    lgb_train = lgb.Dataset(train_data[feature_names], train_data[TARGET_NAME])
    params = {}
    gbm = lgb.train(params, lgb_train)

    return gbm


def predict(gbm: lgb.Booster, test_data: pd.DataFrame, full_data: pd.DataFrame, feature_names: List[str]):
    last_friday = datetime.now() + relativedelta(weekday=FR(-1))
    date_string = last_friday.strftime('%Y-%m-%d')
    print(date_string)
    live_data = full_data.loc[date_string].copy()
    live_data.dropna(subset=feature_names, inplace=True)
    live_data[PREDICTION_NAME] = gbm.predict(live_data[feature_names])
    test_data[PREDICTION_NAME] = gbm.predict(test_data[feature_names])
    return dict(
        predicted_live_data=live_data,
        predicted_test_data=test_data
    )


def report(test_data: pd.DataFrame):
    score_data = test_data[["friday_date", "target", "signal"]]
    score_data_input = score_data.reset_index().drop(
        ["date", "target"], axis=1)
    new_score = score_data_input.groupby("friday_date")["signal"].apply(
        lambda x: x.rank(pct=True, method="first"))
    score_data.reset_index(inplace=True)
    score_data["rank_score"] = new_score
    scores = []
    for friday in score_data["friday_date"].unique():
        score = np.corrcoef(score_data[score_data["friday_date"] == friday]["target"],
                            score_data[score_data["friday_date"] == friday]["rank_score"])[0, 1]
        scores.append(score)
    print(f"score: {np.mean(scores)}")


def make_submit(test_data: pd.DataFrame, live_data: pd.DataFrame):
    last_friday = datetime.now() + relativedelta(weekday=FR(-1))
    diagnostic_df = pd.concat([test_data, live_data])
    diagnostic_df['friday_date'] = diagnostic_df.friday_date.fillna(
        last_friday.strftime('%Y%m%d')).astype(int)
    diagnostic_df['data_type'] = diagnostic_df.data_type.fillna('live')
    return diagnostic_df[['bloomberg_ticker', 'friday_date',
                          'data_type', 'signal']].reset_index(drop=True)
