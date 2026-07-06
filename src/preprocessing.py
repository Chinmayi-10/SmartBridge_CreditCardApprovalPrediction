import pandas as pd

def load_data():
    app = pd.read_csv("dataset/application_record.csv")
    credit = pd.read_csv("dataset/credit_record.csv")
    return app, credit


def preprocess_data(app, credit):
    # Remove duplicates
    app = app.drop_duplicates()

    # Remove unnecessary column
    if "OCCUPATION_TYPE" in app.columns:
        app = app.drop("OCCUPATION_TYPE", axis=1)

    # Convert negative values to positive
    app["DAYS_BIRTH"] = app["DAYS_BIRTH"].abs()
    app["DAYS_EMPLOYED"] = app["DAYS_EMPLOYED"].abs()

    return app, credit