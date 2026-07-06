import joblib
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "model/model.pkl")

    return model