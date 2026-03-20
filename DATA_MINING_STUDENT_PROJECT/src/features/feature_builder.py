import pandas as pd


def build_features(df):

    df = df.copy()

    df["absences_bin"] = pd.cut(
        df["absences"],
        bins=[-1,5,15,100],
        labels=["low","medium","high"]
    )

    df["studytime_bin"] = pd.cut(
        df["studytime"],
        bins=[0,2,4],
        labels=["low","high"]
    )

    return df