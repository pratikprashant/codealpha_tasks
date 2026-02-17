from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('credit_risk_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get data from the HTML form
        income = float(request.form['income'])
        loan_amount = float(request.form['loan_amount'])
        credit_years = float(request.form['credit_history'])
        home_ownership = request.form['home_ownership']

        input_data = pd.DataFrame({
            'annual_inc': [income],
            'loan_amnt': [loan_amount],
            'credit_history_years': [credit_years],
            'home_ownership': [home_ownership]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        result_text = "High Credit Risk (Likely to Default)" if prediction == 1 else "Low Credit Risk (Likely to Pay)"
        risk_color = "red" if prediction == 1 else "green"

        return render_template(
            'index.html',
            prediction_text=result_text,
            probability_text=f'Default Probability: {round(probability*100, 2)}%',
            risk_color=risk_color
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)