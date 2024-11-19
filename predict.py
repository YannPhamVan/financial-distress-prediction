import pickle

from flask import Flask, request, jsonify
import xgboost as xgb

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('financial-distress')

@app.route('/predict', methods=['POST'])
def predict():
    company = request.get_json()
    X = dv.transform([company])
    feature_names = list(dv.get_feature_names_out())
    dX = xgb.DMatrix(X, feature_names=feature_names)
    y_pred = model.predict(dX)[0]
    bankrupt = y_pred >= 0.5
    result = {
        'bankrupt_probability': float(y_pred),
        'bankrupt': bool(bankrupt)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)