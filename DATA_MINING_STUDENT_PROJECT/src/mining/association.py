import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def build_transactions(df):

    cols = [
        "studytime_bin",
        "absences_bin",
        "internet",
        "romantic",
        "goout",
        "failures",
        "pass"
    ]

    transactions = df[cols].astype(str).values.tolist()

    return transactions


def run_apriori(transactions, min_support, min_confidence):

    te = TransactionEncoder()

    te_data = te.fit(transactions).transform(transactions)

    df = pd.DataFrame(te_data, columns=te.columns_)

    freq = apriori(df, min_support=min_support, use_colnames=True)

    rules = association_rules(freq, metric="confidence",
                              min_threshold=min_confidence)

    return rules.sort_values("lift", ascending=False)