# HealthWeb
# HealthWeb – Disease Prediction API 

A simple Flask + Machine Learning system that predicts diseases from symptoms.
_______________________________________________________________________________________________________________________
## Installation
git clone https://github.com/FenuPhilip/HealthWeb.git
cd HealthWeb

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
_______________________________________________________________________________________________________________________

Run Backend
python backend/app.py

_______________________________________________________________________________________________________________________

API Usage
POST /predict
Body:
{
  "symptoms": ["fever", "cough", "fatigue"]
}

_______________________________________________________________________________________________________________________

Train Model
python train.py
