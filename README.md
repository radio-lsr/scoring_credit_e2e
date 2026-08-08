# 🏦 Scoring de Crédit — Projet End-to-End

**Projet complet de data science appliqué à la banque centrale** : de la collecte des données au déploiement en production via API et application web.

---

## 📁 Structure du projet

```
credit_scoring_e2e/
├── api/
│   └── main.py              → API FastAPI (prédiction en temps réel)
├── frontend/
│   └── app.py               → Application Streamlit (interface web)
├── models/
│   ├── credit_scoring_model.pkl   → Modèle Random Forest entraîné
│   └── metadata.json        → Métadonnées du modèle (AUC, Gini, features)
├── data/
│   └── credits_bancaires.csv → Dataset d'entraînement (5000 lignes)
├── notebooks/
│   └── exploration.ipynb    → Notebook d'exploration et modélisation
├── tests/
│   └── test_api.py          → Tests de l'API
├── requirements.txt         → Dépendances Python
├── Dockerfile               → Conteneurisation
├── docker-compose.yml       → Orchestration multi-services
└── README.md                → Ce fichier
```

---

## 🚀 Démarrage rapide

### Option 1 : Local (Python)

```bash
# 1. Cloner et naviguer
cd credit_scoring_e2e

# 2. Environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Dépendances
pip install -r requirements.txt

# 4. Démarrer l'API (terminal 1)
uvicorn api.main:app --reload --port 8000

# 5. Démarrer le frontend (terminal 2)
streamlit run frontend/app.py
```

- API : http://localhost:8000/docs
- Frontend : http://localhost:8501

### Option 2 : Docker (recommandé)

```bash
docker-compose up --build
```

- API : http://localhost:8000
- Frontend : http://localhost:8501

---

## 📊 Résultats du modèle

| Métrique | Valeur |
|----------|--------|
| AUC | 0.730 |
| Gini | 0.461 |
| Algorithme | Random Forest (300 arbres, max_depth=8) |

### Variables les plus importantes
1. **Taux d'endettement** (24.8%)
2. **Utilisation crédit revolving** (21.3%)
3. **Ancienneté emploi** (17.0%)

---

## 🔌 API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Info et métriques du modèle |
| `/health` | GET | Health check |
| `/predict` | POST | Prédiction individuelle |
| `/predict/batch` | POST | Prédiction par lot (CSV) |

### Exemple de requête

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Réponse

```json
{
  "probabilite_defaut": 0.3124,
  "score_credit": 688,
  "decision": "ACCEPTÉ AVEC CAUTION",
  "risque": "Modéré",
  "seuil_utilise": 0.45,
  "features_utilisees": [...]
}
```

---

## 🧪 Tests

```bash
pytest tests/test_api.py -v
```

---

## 📚 Méthodologie CRISP-DM appliquée

1. **Business Understanding** : Prédire le risque de défaut d'un emprunteur pour la supervision bancaire
2. **Data Understanding** : 5000 crédits simulés avec profils réalistes (revenus FCFA, secteurs africains)
3. **Data Preparation** : Feature engineering (ratio crédit/revenu), encodage One-Hot, standardisation
4. **Modeling** : Comparaison Logistic Regression vs Random Forest, validation croisée stratifiée
5. **Evaluation** : AUC, Gini, matrice de confusion, courbe ROC
6. **Deployment** : API REST + application web conteneurisée

---

## 🏦 Contexte Banque



Applications réelles :
- **Supervision bancaire** : surveillance des ratios de défaut par établissement
- **Stabilité financière** : détection précoce des risques systémiques
- **Politique monétaire** : évaluation de la transmission du crédit

---

## 🛡️ Bonnes pratiques de production

- ✅ Modèle versionné avec métadonnées (AUC, Gini, features)
- ✅ Validation des entrées via Pydantic
- ✅ Health check et monitoring
- ✅ Conteneurisation Docker
- ✅ Tests automatisés
- ✅ Documentation interactive (Swagger UI)

---

## 📖 Prochaines étapes suggérées

1. **Ajouter SHAP** pour l'interprétabilité des prédictions individuelles
2. **MLflow** pour le tracking des expérimentations
3. **PostgreSQL** pour le stockage des prédictions en production
4. **GitHub Actions** pour le CI/CD (tests + déploiement automatique)
5. **Grafana** pour le monitoring des performances du modèle

---

*Projet généré dans le cadre du cours Data Analyse — 2026*
