from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# Charger le modèle
MODEL_PATH = "model.joblib"

if not os.path.exists(MODEL_PATH):
    raise Exception(f"❌ Fichier modèle {MODEL_PATH} non trouvé. Entraînez le modèle avant de lancer l'API.")

model = joblib.load(MODEL_PATH)

# Initialisation de l'application FastAPI
app = FastAPI(title="Churn Prediction API", description="API pour prédire le churn des clients.")

# Définition du format des données d'entrée avec les variables spécifiques
class PredictionInput(BaseModel):
    account_length: int = Field(..., alias="Account length")
    number_vmail_messages: int = Field(..., alias="Number vmail messages")
    total_day_calls: int = Field(..., alias="Total day calls")
    total_day_charge: float = Field(..., alias="Total day charge")
    total_eve_calls: int = Field(..., alias="Total eve calls")
    total_eve_charge: float = Field(..., alias="Total eve charge")
    total_night_calls: int = Field(..., alias="Total night calls")
    total_night_charge: float = Field(..., alias="Total night charge")
    total_intl_calls: int = Field(..., alias="Total intl calls")
    total_intl_charge: float = Field(..., alias="Total intl charge")
    customer_service_calls: int = Field(..., alias="Customer service calls")
    international_plan: int = Field(..., alias="International plan")
    voice_mail_plan: int = Field(..., alias="Voice mail plan")

    class Config:
        # Permet de spécifier l'utilisation de l'alias pour l'input et l'output
        allow_population_by_field_name = True

@app.post("/predict")
def predict(data: PredictionInput):
    """Effectue une prédiction en utilisant le modèle ML."""
    try:

        # Préparer les features à partir des valeurs reçues dans la requête
        features = np.array([[
            data.account_length,
            data.number_vmail_messages,
            data.total_day_calls,
            data.total_day_charge,
            data.total_eve_calls,
            data.total_eve_charge,
            data.total_night_calls,
            data.total_night_charge,
            data.total_intl_calls,
            data.total_intl_charge,
            data.customer_service_calls,
            data.international_plan,
            data.voice_mail_plan
        ]])

        # Prédiction avec le modèle
        prediction = model.predict(features)[0]

        # Retourner seulement la prédiction
        return {"prediction": "Churn" if prediction == 1 else "No Churn"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {str(e)}")

# Route de test
@app.get("/")
def root():
    return {"message": "🚀 L'API est en ligne ! Utilisez /predict pour effectuer des prédictions."}

