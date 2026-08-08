"""
API de Scoring de Crédit v2.0 — Mapping métier → features modèle
FastAPI endpoint pour la prédiction de probabilité de défaut
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import json
from pathlib import Path

app = FastAPI(
    title="API Scoring Crédit — Banque Centrale",
    description="Prédiction de la probabilité de défaut d'un emprunteur",
    version="2.0.0"
)

# ==============================================================================
# CHARGEMENT DES MODÈLES
# ==============================================================================
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# Détection automatique : on préfère GB + preprocessor, sinon RF pipeline
MODEL_PATH = MODELS_DIR / "model_gradient_boosting.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
META_PATH = MODELS_DIR / "metadata_gb.json"
SHAP_PATH = MODELS_DIR / "shap_explainer.pkl"

# Fallback sur Random Forest (pipeline complet) si GB n'existe pas
if not MODEL_PATH.exists():
    MODEL_PATH = MODELS_DIR / "model_random_forest.pkl"
    META_PATH = MODELS_DIR / "metadata_rf.json"
    PREPROCESSOR_PATH = None

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH) if PREPROCESSOR_PATH and PREPROCESSOR_PATH.exists() else None
shap_explainer = joblib.load(SHAP_PATH) if SHAP_PATH.exists() else None

with open(META_PATH, "r") as f:
    metadata = json.load(f)

MODEL_TYPE = metadata.get("model_type", "Unknown")
USE_PIPELINE = preprocessor is None  # True si RF (pipeline intégré), False si GB (preprocessor externe)

print(f"✅ Modèle chargé : {MODEL_TYPE} | Pipeline intégré : {USE_PIPELINE} | SHAP : {shap_explainer is not None}")

# ==============================================================================
# SCHÉMAS
# ==============================================================================
class CreditRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Âge de l'emprunteur")
    revenu_mensuel: float = Field(..., gt=0, description="Revenu mensuel en FCFA")
    montant_credit: float = Field(..., gt=0, description="Montant du crédit demandé")
    duree_credit_mois: int = Field(..., ge=6, le=120, description="Durée du crédit en mois")
    anciennete_emploi_mois: int = Field(..., ge=0, le=600, description="Ancienneté dans l'emploi (mois)")
    nb_credits_en_cours: int = Field(0, ge=0, le=20, description="Nombre de crédits en cours")
    taux_endettement: float = Field(..., ge=0, le=100, description="Taux d'endettement actuel (%)")
    historique_defaut: int = Field(0, ge=0, le=1, description="1 si défaut historique, 0 sinon")
    type_emploi: str = Field(..., description="CDI, CDD, Independant, Sans_emploi")
    niveau_etude: str = Field(..., description="Primaire, Secondaire, Superieur, Aucun")
    secteur_activite: str = Field(..., description="Agriculture, Commerce, Services, Industrie, Public")
    compte_bancaire_anciennete_mois: float = Field(..., ge=0, description="Ancienneté du compte bancaire (mois)")
    nb_incidents_paiement_12m: int = Field(0, ge=0, le=50, description="Incidents de paiement sur 12 mois")
    utilisation_credit_revolving: float = Field(0, ge=0, le=100, description="Taux d'utilisation du crédit revolving (%)")
    nombre_enfants: int = Field(0, ge=0, le=20, description="Nombre d'enfants à charge")
    situation_matrimoniale: str = Field(..., description="Celibataire, Marie, Divorce, Veuf")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "revenu_mensuel": 450000,
                "montant_credit": 2500000,
                "duree_credit_mois": 36,
                "anciennete_emploi_mois": 48,
                "nb_credits_en_cours": 1,
                "taux_endettement": 25.5,
                "historique_defaut": 0,
                "type_emploi": "CDI",
                "niveau_etude": "Superieur",
                "secteur_activite": "Services",
                "compte_bancaire_anciennete_mois": 36,
                "nb_incidents_paiement_12m": 0,
                "utilisation_credit_revolving": 15.0,
                "nombre_enfants": 2,
                "situation_matrimoniale": "Marie"
            }
        }

class CreditResponse(BaseModel):
    probabilite_defaut: float
    score_credit: int
    decision: str
    risque: str
    seuil_utilise: float
    features_utilisees: list
    model_type: str

class ExplainResponse(BaseModel):
    probabilite_defaut: float
    shap_values: dict
    top_positive: list
    top_negative: list

# ==============================================================================
# MAPPING MÉTIER → FEATURES MODÈLE
# ==============================================================================
def map_request_to_model_features(req: CreditRequest) -> pd.DataFrame:
    """
    Transforme les champs métier du frontend en les 11 features
    attendues par le modèle entraîné sur credit_risk_dataset.csv
    """
    # --- Numériques ---
    person_age = req.age
    person_income = float(req.revenu_mensuel)
    person_emp_length = max(0, int(req.anciennete_emploi_mois / 12))
    loan_amnt = float(req.montant_credit)
    cb_person_cred_hist_length = max(0, int(req.compte_bancaire_anciennete_mois / 12))

    monthly_payment = req.montant_credit / req.duree_credit_mois
    loan_percent_income = (monthly_payment / req.revenu_mensuel) * 100
    loan_percent_income = min(loan_percent_income, 100.0)

    # --- loan_int_rate (estimation basée sur le profil) ---
    base_rate = 8.0
    risk_premium = (req.taux_endettement / 100) * 4.0
    default_premium = 5.0 if req.historique_defaut == 1 else 0.0
    incident_premium = req.nb_incidents_paiement_12m * 0.8
    revolving_premium = (req.utilisation_credit_revolving / 100) * 2.0
    loan_int_rate = base_rate + risk_premium + default_premium + incident_premium + revolving_premium
    loan_int_rate = round(min(loan_int_rate, 25.0), 2)

    # --- loan_grade ---
    if loan_int_rate < 9:
        loan_grade = "A"
    elif loan_int_rate < 11:
        loan_grade = "B"
    elif loan_int_rate < 13:
        loan_grade = "C"
    elif loan_int_rate < 15:
        loan_grade = "D"
    elif loan_int_rate < 18:
        loan_grade = "E"
    elif loan_int_rate < 21:
        loan_grade = "F"
    else:
        loan_grade = "G"

    # --- Catégorielles ---
    home_map = {
        "Celibataire": "RENT",
        "Marie": "OWN",
        "Divorce": "RENT",
        "Veuf": "OWN"
    }
    person_home_ownership = home_map.get(req.situation_matrimoniale, "RENT")

    intent_map = {
        "Agriculture": "VENTURE",
        "Commerce": "BUSINESS",
        "Services": "PERSONAL",
        "Industrie": "BUSINESS",
        "Public": "EDUCATION"
    }
    loan_intent = intent_map.get(req.secteur_activite, "PERSONAL")

    cb_person_default_on_file = "Y" if req.historique_defaut == 1 else "N"

    data = {
        "person_age": [person_age],
        "person_income": [person_income],
        "person_emp_length": [person_emp_length],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length],
        "person_home_ownership": [person_home_ownership],
        "loan_intent": [loan_intent],
        "loan_grade": [loan_grade],
        "cb_person_default_on_file": [cb_person_default_on_file]
    }

    return pd.DataFrame(data)

# ==============================================================================
# ENDPOINTS
# ==============================================================================
@app.get("/")
def root():
    return {
        "message": "API Scoring Crédit — Banque Centrale",
        "version": "2.0.0",
        "model_type": MODEL_TYPE,
        "model_auc": metadata.get("auc"),
        "model_gini": metadata.get("gini")
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": MODEL_TYPE,
        "preprocessor_loaded": preprocessor is not None,
        "shap_loaded": shap_explainer is not None
    }

@app.post("/predict", response_model=CreditResponse)
def predict(request: CreditRequest):
    try:
        df_model = map_request_to_model_features(request)

        if USE_PIPELINE:
            proba = model.predict_proba(df_model)[0, 1]
        else:
            if preprocessor is None:
                raise HTTPException(status_code=500, detail="Preprocessor manquant pour le modèle Gradient Boosting")
            X_prep = preprocessor.transform(df_model)
            proba = model.predict_proba(X_prep)[0, 1]

        score_credit = int((1 - proba) * 1000)
        seuil = metadata.get("best_threshold", 0.45)

        if proba < 0.25:
            risque = "Faible"
            decision = "ACCEPTÉ"
        elif proba < seuil:
            risque = "Modéré"
            decision = "ACCEPTÉ AVEC CAUTION"
        elif proba < 0.65:
            risque = "Élevé"
            decision = "REFUSÉ"
        else:
            risque = "Très élevé"
            decision = "REFUSÉ"

        return CreditResponse(
            probabilite_defaut=round(float(proba), 4),
            score_credit=score_credit,
            decision=decision,
            risque=risque,
            seuil_utilise=seuil,
            features_utilisees=df_model.columns.tolist(),
            model_type=MODEL_TYPE
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(requests: list[CreditRequest]):
    results = []
    for req in requests:
        result = predict(req)
        results.append(result)
    return {"predictions": results, "count": len(results)}

@app.post("/explain", response_model=ExplainResponse)
def explain(request: CreditRequest):
    if shap_explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer non disponible")

    try:
        df_model = map_request_to_model_features(request)

        if USE_PIPELINE:
            X_prep = model.named_steps['prep'].transform(df_model)
        else:
            X_prep = preprocessor.transform(df_model)

        shap_values = shap_explainer.shap_values(X_prep)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        sv = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        feature_names = metadata.get("feature_names", [])

        shap_dict = {name: round(float(val), 4) for name, val in zip(feature_names, sv)}

        sorted_shap = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
        top_positive = [{"feature": k, "impact": v} for k, v in sorted_shap if v > 0][:5]
        top_negative = [{"feature": k, "impact": v} for k, v in sorted_shap if v < 0][:5]

        proba = model.predict_proba(X_prep)[0, 1] if not USE_PIPELINE else model.predict_proba(df_model)[0, 1]

        return ExplainResponse(
            probabilite_defaut=round(float(proba), 4),
            shap_values=shap_dict,
            top_positive=top_positive,
            top_negative=top_negative
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))