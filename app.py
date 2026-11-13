from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("house_price_linear_regression_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    data = {
        'area': [float(request.form['area'])],
        'bedrooms': [int(request.form['bedrooms'])],
        'bathrooms': [int(request.form['bathrooms'])],
        'stories': [int(request.form['stories'])],
        'mainroad': [request.form['mainroad']],
        'guestroom': [request.form['guestroom']],
        'basement': [request.form['basement']],
        'hotwaterheating': [request.form['hotwaterheating']],
        'airconditioning': [request.form['airconditioning']],
        'parking': [int(request.form['parking'])],
        'prefarea': [request.form['prefarea']],
        'furnishingstatus': [request.form['furnishingstatus']]
    }

    df = pd.DataFrame(data)

    prediction = model.predict(df)[0]

    return render_template(
        'index.html',
        prediction=f"₹ {prediction:,.2f}"
    )

if __name__ == '__main__':
    app.run(debug=True)
