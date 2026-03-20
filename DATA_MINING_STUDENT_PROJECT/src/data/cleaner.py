def clean_data(df):

    df = df.copy()

    df = df.drop_duplicates()

    df["pass"] = (df["G3"] >= 10).astype(int)

    return df