"""
Tests de l'API Scoring de Crédit
pytest -v tests/test_api.py
"""
import pytest
import requests

BASE_URL = "http://localhost:8000"

@pytest.fixture
def sample_payload():
    return {
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

@pytest.fixture
def auth_token():
    """Obtient un token JWT pour les tests"""
    resp = requests.post(f"{BASE_URL}/login", json={
        "username": "analyste",
        "password": "analyste123"
    })
    assert resp.status_code == 200
    return resp.json()["access_token"]

# ─── Tests v1 (sans auth) ──────────────────────────────────
class TestV1:
    def test_health(self):
        resp = requests.get(f"{BASE_URL}/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_predict_v1(self, sample_payload):
        resp = requests.post(f"{BASE_URL}/predict", json=sample_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "probabilite_defaut" in data
        assert "score_credit" in data
        assert "decision" in data
        assert 0 <= data["probabilite_defaut"] <= 1
        assert 0 <= data["score_credit"] <= 1000

    def test_predict_invalid_age(self):
        payload = {
            "age": 15,
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
        resp = requests.post(f"{BASE_URL}/predict", json=payload)
        assert resp.status_code == 422

# ─── Tests v2 (avec auth) ──────────────────────────────────
class TestV2:
    def test_login_success(self):
        resp = requests.post(f"{BASE_URL}/login", json={
            "username": "analyste",
            "password": "analyste123"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["role"] == "analyste"

    def test_login_failure(self):
        resp = requests.post(f"{BASE_URL}/login", json={
            "username": "analyste",
            "password": "mauvais_mdp"
        })
        assert resp.status_code == 401

    def test_predict_v2(self, auth_token, sample_payload):
        headers = {"Authorization": f"Bearer {auth_token}"}
        resp = requests.post(f"{BASE_URL}/predict", json=sample_payload, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "shap_explanation" in data
        assert "model_version" in data
        assert isinstance(data["shap_explanation"], dict)

    def test_predict_unauthorized(self, sample_payload):
        resp = requests.post(f"{BASE_URL}/predict", json=sample_payload)
        assert resp.status_code == 403

    def test_audit_forbidden_for_analyste(self, auth_token):
        headers = {"Authorization": f"Bearer {auth_token}"}
        resp = requests.get(f"{BASE_URL}/audit/stats", headers=headers)
        assert resp.status_code == 403

    def test_audit_accessible_to_superviseur(self):
        resp = requests.post(f"{BASE_URL}/login", json={
            "username": "superviseur",
            "password": "superviseur123"
        })
        token = resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(f"{BASE_URL}/audit/stats", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_predictions" in data