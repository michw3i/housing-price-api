from flask import Flask, request, jsonify
import joblib
import pandas as pd 

#initalizing Flask app
app = Flask(__name__)
#creates a web application

#load trained model once at startup
model = joblib.load("models/best_model.pkl")
feature_names = list(model.feature_names_in_)

#define prediction endpoints
@app.route("/predict", methods = ["POST"])
def predict():
	data = request.get_json(force=True)
	df = pd.DataFrame([data])
	df = df.reindex(columns=feature_names, fill_value = 0)
	prediction = model.predict(df)[0]
	return jsonify({"predicted_price": float(prediction)})

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5001, debug=True)