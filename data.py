from darts import TimeSeries
from torchmetrics import SymmetricMeanAbsolutePercentageError
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel, LightGBMModel
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
import itertools

def load_train(filename):
    df = pd.read_csv(os.path.join(os.path.curdir, "data", filename))

    countries = df["country"].unique()
    stores = df["store"].unique()
    products = df["product"].unique()

    encoder = OneHotEncoder()

    ts_products = list(itertools.product(countries, stores, products))
    encoder.fit(ts_products)

    ts_list = []

    for data in ts_products:
        country, store, product = data
        one_hot = pd.Series(encoder.transform([data]).toarray()[0])

        ts_df = df[(df["country"] == country) & (df["store"] == store) & (df["product"] == product)]
        ts_curr = TimeSeries.from_dataframe(df=ts_df, time_col="date", value_cols="num_sold", static_covariates=one_hot)
        ts_list.append(ts_curr)
    return ts_list

def train_model(model, is_deterministic=False):
    train, valid = zip(*(ts.split_after(0.8) for ts in load_train("train.csv")))
    if not is_deterministic:
        model.fit(series=train, val_series=valid, epochs=10, num_loader_workers=4)
    else:
        model.fit(series=train, val_series=valid)
    return model

def tft():
    pl_trainer_kwargs = {
        "accelerator": "gpu"
    }
    model = TFTModel(
        input_chunk_length=124,
        output_chunk_length=31,
        loss_fn=SymmetricMeanAbsolutePercentageError(),
        add_encoders={
            "cyclic": {"future": ["month", "day"]},
            "transformer": Scaler(),
        },
        pl_trainer_kwargs=pl_trainer_kwargs
    )
    train_model(model)

def light_gbm():
    model = LightGBMModel(
        lags=124,
        output_chunk_length=31,
        add_encoders={
            "cyclic": {"future": ["month", "day"]},
            "transformer": Scaler(),
        }
    )
    return train_model(model, True)

