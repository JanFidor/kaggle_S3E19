# TFTModel_2023-07-13_20_49_37

from darts import TimeSeries
from torchmetrics import SymmetricMeanAbsolutePercentageError
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
import pandas as pd
import os
from data import load_train, train_model, light_gbm
import numpy as np


def get_order():
    df = pd.read_csv(os.path.join(os.path.curdir, "data", "test.csv"))
    df["key"] = df["country"] + "," + df["store"] + "," + df["product"]
    return df["key"].unique()


def submission(model):
    test = load_train("train.csv")
    predictions = model.predict(365, test)

    ts_series = []

    for i, pred in enumerate(predictions):
        values = pred.pd_series().tolist()
        print(len(values))
        index = 136950 + i + np.arange(0, 365 * 75, 75)
        ts_series.append(pd.Series(values, index=index))

    series = pd.concat(ts_series).sort_index()
    print(len(series))
    series.to_csv("submission.csv")
    print(pd.read_csv("submission.csv").size)


if __name__ == '__main__':
    model = light_gbm()
    submission(model)


