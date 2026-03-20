import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import joblib

from src.data.loader import load_dataset
from src.data.cleaner import clean_data
from src.features.feature_builder import build_features

from src.mining.association import build_transactions, run_apriori
from src.mining.clustering import run_kmeans

from src.models.supervised import train_model
from src.models.semi_supervised import train_label_spreading

from src.evaluation.metrics import compute_f1


with open("configs/params.yaml") as f:
    params = yaml.safe_load(f)


print("Loading dataset")

df = load_dataset(params["data"]["raw_path"])


print("Cleaning data")

df = clean_data(df)


print("Building features")

df = build_features(df)


print("Running association rules")

transactions = build_transactions(df)

rules = run_apriori(
    transactions,
    params["association"]["min_support"],
    params["association"]["min_confidence"]
)

rules.to_csv("outputs/tables/association_rules.csv")


print("Running clustering")

features = [
    "studytime",
    "failures",
    "absences",
    "goout",
    "Dalc",
    "Walc"
]

X = df[features]

labels, model = run_kmeans(X, params["clustering"]["max_k"])

df["cluster"] = labels


print("Running classification")

y = df["pass"]

model, X_test, y_test = train_model(
    X,
    y,
    params["classification"]["test_size"],
    params["classification"]["random_state"]
)

pred = model.predict(X_test)

f1 = compute_f1(y_test, pred)

print("F1 Score:", f1)

joblib.dump(model, "outputs/models/random_forest.pkl")


print("Running semi supervised learning")

semi_pred = train_label_spreading(
    X.values,
    y.values,
    params["semi_supervised"]["labeled_ratio"]
)

print("Pipeline completed")