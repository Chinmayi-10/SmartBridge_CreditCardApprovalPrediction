from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("model/model.pkl")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict_page():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():

    features = [
        float(request.form["FLAG_OWN_REALTY"]),
        float(request.form["CNT_CHILDREN"]),
        float(request.form["AMT_INCOME_TOTAL"]),
        float(request.form["NAME_INCOME_TYPE"]),
        float(request.form["NAME_EDUCATION_TYPE"]),
        float(request.form["DAYS_BIRTH"]),
        float(request.form["DAYS_EMPLOYED"]),
        float(request.form["FLAG_WORK_PHONE"]),
        float(request.form["FLAG_PHONE"]),
        float(request.form["FLAG_EMAIL"]),
        float(request.form["CNT_FAM_MEMBERS"]),
        float(request.form["TOTAL_FAMILY_MEMBERS"]),
        float(request.form["open_month"]),
        float(request.form["end_month"]),
        float(request.form["window"]),
        float(request.form["CODE_GENDER_1"]),
        float(request.form["FLAG_OWN_CAR_1"]),
        float(request.form["NAME_FAMILY_STATUS_1"]),
        float(request.form["NAME_FAMILY_STATUS_2"]),
        float(request.form["NAME_FAMILY_STATUS_3"]),
        float(request.form["NAME_FAMILY_STATUS_4"]),
        float(request.form["NAME_HOUSING_TYPE_1"]),
        float(request.form["NAME_HOUSING_TYPE_2"]),
        float(request.form["NAME_HOUSING_TYPE_3"]),
        float(request.form["NAME_HOUSING_TYPE_4"]),
        float(request.form["NAME_HOUSING_TYPE_5"])
    ]

    df = pd.DataFrame([features], columns=[
        'FLAG_OWN_REALTY',
        'CNT_CHILDREN',
        'AMT_INCOME_TOTAL',
        'NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE',
        'DAYS_BIRTH',
        'DAYS_EMPLOYED',
        'FLAG_WORK_PHONE',
        'FLAG_PHONE',
        'FLAG_EMAIL',
        'CNT_FAM_MEMBERS',
        'TOTAL_FAMILY_MEMBERS',
        'open_month',
        'end_month',
        'window',
        'CODE_GENDER_1',
        'FLAG_OWN_CAR_1',
        'NAME_FAMILY_STATUS_1',
        'NAME_FAMILY_STATUS_2',
        'NAME_FAMILY_STATUS_3',
        'NAME_FAMILY_STATUS_4',
        'NAME_HOUSING_TYPE_1',
        'NAME_HOUSING_TYPE_2',
        'NAME_HOUSING_TYPE_3',
        'NAME_HOUSING_TYPE_4',
        'NAME_HOUSING_TYPE_5'
    ])

    prediction = model.predict(df)[0]

    if prediction == 1:
        result = "Approved"
    else:
        result = "Rejected"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)