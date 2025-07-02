import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from app import app

@pytest.fixture
def client():
	app.config["TESTING"] = True 
	with app.test_client() as client:
		yield client 

def text_predict_minimal_payload(client):
	payload = {
		"OverallQual":7,
		"GrLivArea": 1710,
		"GarageCars":2,
		"GarageArea":548,
		"TotalBsmtSF":856
	}

	response = client.post("/predict", json=payload)
	assert response.status_code==200
	data = response.get_json()
	assert "predicted_price" in data
	assert isinstance(data["predicted_price"], float)

def test_predict_empty_payload(client):
	response = client.post("/predict", json={})
	assert response.status_code == 200
	data = response.get_json()
	assert isinstance(data["predicted_price"], float)