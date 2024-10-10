import random

import numpy as np
import pandas as pd


def encode_binary_columns(df, columns):
    mapping_dict = {"yes": 1, "no": 0}
    for col in columns:
        df[col] = df[col].str.lower().map(mapping_dict)

    return df


def encode_date_columns(df, columns):
    for col in columns:
        df[col] = pd.to_datetime(df[col])

        df["DayOfYear"] = df[col].dt.dayofyear
        df["DaysInYear"] = df[col].dt.is_leap_year.apply(
            lambda leap: 366 if leap else 365
        )

        df["Angle"] = 2 * np.pi * (df["DayOfYear"] - 1) / (df["DaysInYear"])

        df[f"DayCos"] = np.cos(df["Angle"])
        df[f"DaySin"] = np.sin(df["Angle"])

        df = df.drop(columns=["DayOfYear", "DaysInYear", "Angle"])

    return df


def encode_dir_columns(df, columns):
    dirs = [
        "E",
        "ENE",
        "NE",
        "NNE",
        "N",
        "NNW",
        "NW",
        "WNW",
        "W",
        "WSW",
        "SW",
        "SSW",
        "S",
        "SSE",
        "SE",
        "ESE",
    ]
    angles = np.radians(np.arange(0, 360, 22.5))
    mapping_dict = {d: a for (d, a) in zip(dirs, angles)}

    for col in columns:
        df[f"{col}Angle"] = df[col].map(mapping_dict)

        df[f"{col}Cos"] = np.cos(df[f"{col}Angle"])
        df[f"{col}Sin"] = np.sin(df[f"{col}Angle"])

        df = df.drop(columns=f"{col}Angle")

    return df


def split_by_date(df, train_ratio=0.85, test_ratio=0.15):
    if train_ratio + test_ratio != 1.0:
        raise ValueError("The sum of the split ratios must be 1.0")

    list_dates = df["Date"].unique().tolist()

    random.shuffle(list_dates)

    total_size = len(list_dates)
    train_size = int(total_size * train_ratio)

    train_dates = list_dates[:train_size]
    test_dates = list_dates[train_size:]

    train_filt = df["Date"].isin(train_dates)
    train_set = df.loc[train_filt, :].reset_index(drop=True)

    test_filt = df["Date"].isin(test_dates)
    test_set = df.loc[test_filt, :].reset_index(drop=True)

    return train_set, test_set
