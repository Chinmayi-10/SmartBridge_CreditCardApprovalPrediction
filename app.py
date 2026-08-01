from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("model/model.pkl")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict_page():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():

    # ---------------- Family Status ----------------
    if request.form["NAME_FAMILY_STATUS_1"] == "1":
        family_status = 0
    elif request.form["NAME_FAMILY_STATUS_2"] == "1":
        family_status = 1
    elif request.form["NAME_FAMILY_STATUS_3"] == "1":
        family_status = 2
    elif request.form["NAME_FAMILY_STATUS_4"] == "1":
        family_status = 3
    else:
        family_status = 0

    # ---------------- Housing Type ----------------
    if request.form["NAME_HOUSING_TYPE_1"] == "1":
        housing_type = 0
    elif request.form["NAME_HOUSING_TYPE_2"] == "1":
        housing_type = 1
    elif request.form["NAME_HOUSING_TYPE_3"] == "1":
        housing_type = 2
    elif request.form["NAME_HOUSING_TYPE_4"] == "1":
        housing_type = 3
    elif request.form["NAME_HOUSING_TYPE_5"] == "1":
        housing_type = 4
    else:
        housing_type = 0

    # ---------------- Convert Years → Days ----------------
    age_years = int(request.form["DAYS_BIRTH"])
    work_years = int(request.form["DAYS_EMPLOYED"])

    days_birth = age_years * 365
    days_employed = work_years * 365

    # ---------------- Occupation ----------------
    # Temporary default because your current form doesn't have this field.
    occupation = 0

    # ---------------- DataFrame ----------------
    df = pd.DataFrame([{
        "CODE_GENDER": int(request.form["CODE_GENDER_1"]),
        "FLAG_OWN_CAR": int(request.form["FLAG_OWN_CAR_1"]),
        "FLAG_OWN_REALTY": int(request.form["FLAG_OWN_REALTY"]),
        "CNT_CHILDREN": int(request.form["CNT_CHILDREN"]),
        "AMT_INCOME_TOTAL": float(request.form["AMT_INCOME_TOTAL"]),
        "NAME_INCOME_TYPE": int(request.form["NAME_INCOME_TYPE"]),
        "NAME_EDUCATION_TYPE": int(request.form["NAME_EDUCATION_TYPE"]),
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing_type,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "FLAG_MOBIL": 1,
        "FLAG_WORK_PHONE": int(request.form["FLAG_WORK_PHONE"]),
        "FLAG_PHONE": int(request.form["FLAG_PHONE"]),
        "FLAG_EMAIL": int(request.form["FLAG_EMAIL"]),
        "OCCUPATION_TYPE": occupation,
        "CNT_FAM_MEMBERS": float(request.form["CNT_FAM_MEMBERS"]),
        "open_month": float(request.form["open_month"]),
        "end_month": float(request.form["end_month"]),
        "window": float(request.form["window"])
    }])

    # Arrange columns exactly as the model expects
    df = df[model.feature_names_in_]

    proba = model.predict_proba(df)

    print("Prediction probabilities:", proba)

    prediction = model.predict(df)[0]

    if prediction == 1:
        result = "✅ Credit Card Approved"
    else:
        result = "❌ Credit Card Rejected"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)