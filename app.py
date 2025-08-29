import flask
from flask import render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

app = flask.Flask(__name__)

MODEL_PATH = 'best_student_performance_model.pkl'
COLUMNS_FILE = 'columns.txt'

# Load model and columns
with open(MODEL_PATH, 'rb') as f:
    model = joblib.load(f)

with open(COLUMNS_FILE, 'r') as f:
    training_columns = [line.strip() for line in f if line.strip()]

# Dummy scaler to keep your current approach (ideally save the real scaler)
scaler = MinMaxScaler().fit(pd.DataFrame(np.zeros((1, len(training_columns))), columns=training_columns))

@app.route('/', methods=['GET'])
def home():
    # Render template with no result initially
    return render_template('form.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1) Read form inputs
        data = request.form.to_dict()

        # 2) Validate inputs
        if int(data['age']) < 6:
            return render_template('form.html', result="Error: Age must be at least 6.")
        if float(data['Attendance_Ratio']) < 0 or float(data['Attendance_Ratio']) > 100:
            return render_template('form.html', result="Error: Attendance Ratio must be between 0 and 100.")
        for grade in ['G1', 'G2', 'G3', 'Average_Grade']:
            if float(data[grade]) < 0 or float(data[grade]) > 20:
                return render_template('form.html', result=f"Error: {grade} must be between 0 and 20.")
        if int(data['studytime']) < 0:
            return render_template('form.html', result="Error: Study Time must be at least 0.")
        if int(data['absences']) < 0:
            return render_template('form.html', result="Error: Absences must be at least 0.")

        # 3) To numeric where possible (strings -> numbers)
        for k, v in data.items():
            if isinstance(v, str) and v.strip() != '':
                try:
                    if '.' in v:
                        data[k] = float(v)  # Convert to float for decimal values
                    else:
                        data[k] = int(v)  # Convert to int for whole numbers
                except ValueError:
                    # Leave as string if not numeric
                    pass

        # 4) Align with training columns (missing -> 0)
        aligned = pd.DataFrame(np.zeros((1, len(training_columns))), columns=training_columns)

        for col in data.keys():
            if col in aligned.columns:
                aligned[col] = data[col]  # Assign provided values

        # 5) Scale then predict
        X = scaler.transform(aligned)
        y_pred = model.predict(X)
        predicted_risk_category = str(y_pred[0])

        # 6) Re-render the same page with result clearly shown under the table
        return render_template('form.html', result=predicted_risk_category)

    except Exception as e:
        # Show error nicely on the page (optional)
        return render_template('form.html', result=f"Error: {e}")

if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
