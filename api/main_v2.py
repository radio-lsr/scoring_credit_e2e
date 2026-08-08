# =============================================================================
#  API de Scoring de Credit v2.0 — Production Ready
# =============================================================================
#  Auteur      : Karl Bifu Batunguni - Data scientist
#  Organisation: Bifu Albert Bank
#  Date        : 2026-08-07
#  Version     : 2.0.0
#  Licence     : MIT
#
#  Copyright (c) 2026 Bifu Albert Bank. Tous droits reserves.
# =============================================================================
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import json
from pathlib import Path

# ─── Configuration ──────────────────────────────────────────
SECRET_KEY = "votre-cle-super-secrete-a-changer-en-prod"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

DATABASE_URL = "sqlite:///./scoring_audit.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ─── Modèles DB ───────────────────────────────────────────
class PredictionLog(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    username = Column(String)
    probabilite_defaut = Column(Float)
    score_credit = Column(Integer)
    decision = Column(String)
    risque = Column(String)
    features = Column(JSON)
    shap_values = Column(JSON)
    model_version = Column(String)

Base.metadata.create_all(bind=engine)

# ─── Auth ─────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

USERS_DB = {
    "analyste": {
        "username": "analyste",
        "hashed_password": pwd_context.hash("analyste123"),
        "role": "analyste",
        "permissions": ["predict", "batch"]
    },
    "superviseur": {
        "username": "superviseur",
        "hashed_password": pwd_context.hash("superviseur123"),
        "role": "superviseur",
        "permissions": ["predict", "batch", "audit", "admin"]
    }
}

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None or username not in USERS_DB:
            raise HTTPException(status_code=401, detail="Token invalide")
        return USERS_DB[username]
    except JWTError:
        raise HTTPException(status_code=401, detail="Token invalide")

def require_permission(permission: str):
    def checker(user: dict = Depends(verify_token)):
        if permission not in user.get("permissions", []):
            raise HTTPException(status_code=403, detail="Permission insuffisante")
        return user
    return checker

# ─── Chargement des modèles ───────────────────────────────
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "model_gradient_boosting.pkl"
META_PATH = MODELS_DIR / "metadata_gb.json"
PREP_PATH = MODELS_DIR / "preprocessor.pkl"
SHAP_PATH = MODELS_DIR / "shap_explainer.pkl"

if not MODEL_PATH.exists():
    MODEL_PATH = MODELS_DIR / "model_random_forest.pkl"
    META_PATH = MODELS_DIR / "metadata_rf.json"
    PREP_PATH = None

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREP_PATH) if PREP_PATH and PREP_PATH.exists() else None
explainer = joblib.load(SHAP_PATH) if SHAP_PATH.exists() else None

with open(META_PATH, "r") as f:
    metadata = json.load(f)

feature_names = metadata.get("feature_names", [])
MODEL_TYPE = metadata.get("model_type", "Unknown")
USE_PIPELINE = preprocessor is None

print(f"✅ Modele charge : {MODEL_TYPE} | Pipeline : {USE_PIPELINE} | SHAP : {explainer is not None}")

# ─── FastAPI App ──────────────────────────────────────────
app = FastAPI(
    title="API Scoring Credit v2.0 — Bifu Albert Bank",
    description="XGBoost + SHAP + JWT + Audit | (c) 2026 Bifu Albert Bank",
    version="2.0.0"
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─── Schémas ──────────────────────────────────────────────
class CreditRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    revenu_mensuel: float = Field(..., gt=0)
    montant_credit: float = Field(..., gt=0)
    duree_credit_mois: int = Field(..., ge=6, le=120)
    anciennete_emploi_mois: int = Field(..., ge=0, le=600)
    nb_credits_en_cours: int = Field(0, ge=0, le=20)
    taux_endettement: float = Field(..., ge=0, le=100)
    historique_defaut: int = Field(0, ge=0, le=1)
    type_emploi: str = Field(...)
    niveau_etude: str = Field(...)
    secteur_activite: str = Field(...)
    compte_bancaire_anciennete_mois: float = Field(..., ge=0)
    nb_incidents_paiement_12m: int = Field(0, ge=0, le=50)
    utilisation_credit_revolving: float = Field(0, ge=0, le=100)
    nombre_enfants: int = Field(0, ge=0, le=20)
    situation_matrimoniale: str = Field(...)

class LoginRequest(BaseModel):
    username: str
    password: str

class CreditResponse(BaseModel):
    probabilite_defaut: float
    score_credit: int
    decision: str
    risque: str
    shap_explanation: dict
    model_version: str
    seuil_utilise: float
    features_utilisees: list
    model_type: str

# ─── MAPPING METIER -> FEATURES MODELE ─────────────────────
def map_request_to_model_features(req: CreditRequest) -> pd.DataFrame:
    """Transforme les champs metier en features du modele credit_risk_dataset."""
    person_age = req.age
    person_income = float(req.revenu_mensuel)
    person_emp_length = max(0, int(req.anciennete_emploi_mois / 12))
    loan_amnt = float(req.montant_credit)
    cb_person_cred_hist_length = max(0, int(req.compte_bancaire_anciennete_mois / 12))

    monthly_payment = req.montant_credit / req.duree_credit_mois
    loan_percent_income = (monthly_payment / req.revenu_mensuel) * 100
    loan_percent_income = min(loan_percent_income, 100.0)

    base_rate = 8.0
    risk_premium = (req.taux_endettement / 100) * 4.0
    default_premium = 5.0 if req.historique_defaut == 1 else 0.0
    incident_premium = req.nb_incidents_paiement_12m * 0.8
    revolving_premium = (req.utilisation_credit_revolving / 100) * 2.0
    loan_int_rate = base_rate + risk_premium + default_premium + incident_premium + revolving_premium
    loan_int_rate = round(min(loan_int_rate, 25.0), 2)

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

    home_map = {"Celibataire": "RENT", "Marie": "OWN", "Divorce": "RENT", "Veuf": "OWN"}
    person_home_ownership = home_map.get(req.situation_matrimoniale, "RENT")

    intent_map = {"Agriculture": "VENTURE", "Commerce": "BUSINESS", "Services": "PERSONAL",
                  "Industrie": "BUSINESS", "Public": "EDUCATION"}
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

# ─── Endpoints ────────────────────────────────────────────
@app.post("/login")
def login(req: LoginRequest):
    user = USERS_DB.get(req.username)
    if not user or not pwd_context.verify(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Identifiants invalides")
    token = create_access_token(
        {"sub": req.username, "role": user["role"]},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": token, "token_type": "bearer", "role": user["role"]}

@app.get("/")
def root():
    return {
        "api": "Scoring Credit v2.0",
        "copyright": "(c) 2026 Bifu Albert Bank",
        "author": metadata.get("author", "Karl Bifu Batunguni - Data scientist"),
        "model": MODEL_TYPE,
        "auc": metadata.get("auc"),
        "gini": metadata.get("gini"),
        "version": metadata.get("version")
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=CreditResponse)
def predict(
    request: CreditRequest,
    user: dict = Depends(require_permission("predict")),
    db: Session = Depends(get_db)
):
    try:
        df_model = map_request_to_model_features(request)

        if USE_PIPELINE:
            proba = model.predict_proba(df_model)[0, 1]
            X_prep = model.named_steps['prep'].transform(df_model)
        else:
            if preprocessor is None:
                raise HTTPException(status_code=500, detail="Preprocessor manquant")
            X_prep = preprocessor.transform(df_model)
            proba = model.predict_proba(X_prep)[0, 1]

        shap_dict = {}
        if explainer is not None:
            shap_vals = explainer.shap_values(X_prep)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            sv = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
            shap_dict = {name: float(val) for name, val in zip(feature_names, sv)}

        score_credit = int((1 - proba) * 1000)
        seuil = metadata.get("best_threshold", 0.45)

        if proba < 0.25:
            risque, decision = "Faible", "ACCEPTE"
        elif proba < seuil:
            risque, decision = "Modere", "ACCEPTE AVEC CAUTION"
        elif proba < 0.65:
            risque, decision = "Eleve", "REFUSE"
        else:
            risque, decision = "Tres eleve", "REFUSE"

        log = PredictionLog(
            username=user["username"],
            probabilite_defaut=float(proba),
            score_credit=score_credit,
            decision=decision,
            risque=risque,
            features=df_model.to_dict(orient='records')[0],
            shap_values=shap_dict,
            model_version=metadata.get("version", "unknown")
        )
        db.add(log)
        db.commit()

        return CreditResponse(
            probabilite_defaut=round(float(proba), 4),
            score_credit=score_credit,
            decision=decision,
            risque=risque,
            shap_explanation=shap_dict,
            model_version=metadata.get("version", "unknown"),
            seuil_utilise=seuil,
            features_utilisees=df_model.columns.tolist(),
            model_type=MODEL_TYPE
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(
    requests: list[CreditRequest],
    user: dict = Depends(require_permission("batch")),
    db: Session = Depends(get_db)
):
    results = []
    for req in requests:
        result = predict(req, user, db)
        results.append(result)
    return {"predictions": results, "count": len(results)}

@app.get("/audit/predictions")
def get_audit(
    limit: int = 100,
    user: dict = Depends(require_permission("audit")),
    db: Session = Depends(get_db)
):
    logs = db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": log.id,
            "timestamp": log.timestamp.isoformat() if log.timestamp else None,
            "username": log.username,
            "probabilite_defaut": log.probabilite_defaut,
            "score_credit": log.score_credit,
            "decision": log.decision,
            "risque": log.risque,
            "model_version": log.model_version
        }
        for log in logs
    ]

@app.get("/audit/stats")
def get_stats(
    user: dict = Depends(require_permission("audit")),
    db: Session = Depends(get_db)
):
    total = db.query(PredictionLog).count()
    accepted = db.query(PredictionLog).filter(PredictionLog.decision.like("ACCEPTE%")).count()
    refused = db.query(PredictionLog).filter(PredictionLog.decision == "REFUSE").count()
    avg_score = db.query(func.avg(PredictionLog.score_credit)).scalar()

    return {
        "total_predictions": total,
        "acceptes": accepted,
        "refuses": refused,
        "taux_acceptation": round(accepted / total, 4) if total else 0,
        "score_moyen": round(float(avg_score), 2) if avg_score else 0
    }