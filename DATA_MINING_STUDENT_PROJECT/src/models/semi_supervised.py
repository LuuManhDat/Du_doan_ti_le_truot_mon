import numpy as np
from sklearn.semi_supervised import LabelSpreading


def simulate_low_labels(y, ratio):

    y_semi = y.copy()

    mask = np.random.rand(len(y)) > ratio

    y_semi[mask] = -1

    return y_semi


def train_label_spreading(X, y, ratio):

    y_semi = simulate_low_labels(y, ratio)

    model = LabelSpreading()

    model.fit(X, y_semi)

    preds = model.transduction_

    return preds