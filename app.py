from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
model = joblib.load('diabetes_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['HighBP']),
        float(request.form['HighChol']),
        float(request.form['CholCheck']),
        float(request.form['BMI']),
        float(request.form['Smoker']),
        float(request.form['Stroke']),
        float(request.form['HeartDiseaseorAttack']),
        float(request.form['PhysActivity']),
        float(request.form['Fruits']),
        float(request.form['Veggies']),
        float(request.form['HvyAlcoholConsump']),
        float(request.form['AnyHealthcare']),
        float(request.form['NoDocbcCost']),
        float(request.form['GenHlth']),
        float(request.form['MentHlth']),
        float(request.form['PhysHlth']),
        float(request.form['DiffWalk']),
        float(request.form['Sex']),
        float(request.form['Age']),
        float(request.form['Education']),
        float(request.form['Income']),
    ]

    feature_names = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                     'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                     'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                     'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']

    input_df = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    risk_pct = round(probability[1] * 100, 1)

    if prediction == 0:
        result = f"LOW RISK - based on your responses, you are not at risk for developing diabetes ({risk_pct}% risk score)."
        tips = []
    else:
        result = f"AT RISK - based on your responses, you are at risk for developing diabetes! ({risk_pct}% risk score)."
        tips = []
        if float(request.form['HighBP']) == 1:
            tips.append("Managing your blood pressure can significantly reduce diabetes risk.")
        if float(request.form['HighChol']) == 1:
            tips.append("High cholesterol is closely linked to insulin resistance")
        if float(request.form['BMI']) > 25:
            tips.append("Weight reduction can lower diabetes risk.")
        if float(request.form['PhysActivity']) == 0:
            tips.append("Physical activity improves insulin sensitivity.")
        if float(request.form['Fruits']) == 0 or float(request.form['Veggies']) == 0:
            tips.append("Fruits and vegetables support blood sugar regulation.")

    return render_template('index.html', result=result, tips=tips)

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
