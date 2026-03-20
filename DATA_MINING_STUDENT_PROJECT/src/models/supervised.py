from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_model(X, y, test_size, random_state):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    model = RandomForestClassifier(n_estimators=200)

    model.fit(X_train, y_train)

    return model, X_test, y_test