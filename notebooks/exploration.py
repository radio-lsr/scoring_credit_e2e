#!/usr/bin/env python3
"""
Notebook d'Exploration — Scoring de Crédit Banque Centrale
Module 5 (Python) + Module 7 (ML) + Module 10 (Déploiement)
CRISP-DM appliqué end-to-end
"""

# %% IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
import xgboost as xgb
import shap
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("SCORING DE CREDIT — PIPELINE COMPLET")
print("="*60)

# %% 1. CHARGEMENT DES DONNEES
df = pd.read_csv("../data/credits_bancaires.csv")
print(f"Shape: {df.shape}")
print(f"Taux de defaut: {df['defaut'].mean():.2%}")
print("\nApercu:")
print(df.head())

# %% 2. DEFINITION DES FEATURES
numeric_features = ['age', 'revenu_mensuel', 'montant_credit', 'duree_credit_mois',
                    'anciennete_emploi_mois', 'nb_credits_en_cours', 'taux_endettement',
                    'compte_bancaire_anciennete_mois', 'nb_incidents_paiement_12m',
                    'utilisation_credit_revolving', 'nombre_enfants', 'ratio_credit_revenu']

categorical_features = ['type_emploi', 'niveau_etude', 'secteur_activite', 
                        'situation_matrimoniale', 'historique_defaut']

print(f"\nVariables numeriques: {len(numeric_features)}")
print(f"Variables categorielles: {len(categorical_features)}")

# %% 3. STATISTIQUES DESCRIPTIVES
print("\n=== STATISTIQUES DESCRIPTIVES ===")
print(df[numeric_features].describe().round(2))

print("\n=== TAUX DE DEFAUT PAR CATEGORIE ===")
for col in categorical_features:
    print(f"\n{col}:")
    print(df.groupby(col)['defaut'].agg(['mean', 'count']).round(3))

# %% 4. ANALYSE EXPLORATOIRE (EDA)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

df.groupby('type_emploi')['defaut'].mean().plot(kind='bar', ax=axes[0,0], color='steelblue')
axes[0,0].set_title("Taux de defaut par type d'emploi")
axes[0,0].set_ylabel('Taux de defaut')

df.groupby('niveau_etude')['defaut'].mean().plot(kind='bar', ax=axes[0,1], color='coral')
axes[0,1].set_title("Taux de defaut par niveau d'etude")

sns.boxplot(data=df, x='defaut', y='revenu_mensuel', ax=axes[1,0])
axes[1,0].set_title('Revenu mensuel vs Defaut')
axes[1,0].set_yscale('log')

corr_defaut = df[numeric_features + ['defaut']].corr()['defaut'].drop('defaut').sort_values()
corr_defaut.plot(kind='barh', ax=axes[1,1], color='teal')
axes[1,1].set_title('Correlation avec le defaut')

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Graphique EDA sauvegarde: eda_overview.png")

# %% 5. TESTS STATISTIQUES
revenu_defaut = df[df['defaut']==1]['revenu_mensuel']
revenu_non_defaut = df[df['defaut']==0]['revenu_mensuel']
t_stat, p_value = stats.ttest_ind(revenu_defaut, revenu_non_defaut)
print(f"\nTest t sur le revenu: t={t_stat:.2f}, p-value={p_value:.4f}")
print("→ Difference significative" if p_value < 0.05 else "→ Pas de difference significative")

# %% 6. PREPARATION DES DONNEES
X = df.drop('defaut', axis=1)
y = df['defaut']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
])

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

feature_names = (numeric_features + 
                 list(preprocessor.named_transformers_['cat']
                      .get_feature_names_out(categorical_features)))

print(f"Features apres preprocessing: {len(feature_names)}")

# %% 7. MODELISATION — COMPARAISON

# 7.1 Logistic Regression
pipe_logit = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
pipe_logit.fit(X_train, y_train)
proba_logit = pipe_logit.predict_proba(X_test)[:, 1]
auc_logit = roc_auc_score(y_test, proba_logit)
gini_logit = 2 * auc_logit - 1

# 7.2 Random Forest
pipe_rf = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                    class_weight='balanced', random_state=42, n_jobs=-1))
])
pipe_rf.fit(X_train, y_train)
proba_rf = pipe_rf.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, proba_rf)
gini_rf = 2 * auc_rf - 1

# 7.3 XGBoost
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='auc', random_state=42, n_jobs=-1,
    max_depth=4, learning_rate=0.05, n_estimators=200, subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
)
xgb_clf.fit(X_train_prep, y_train)
proba_xgb = xgb_clf.predict_proba(X_test_prep)[:, 1]
auc_xgb = roc_auc_score(y_test, proba_xgb)
gini_xgb = 2 * auc_xgb - 1

print("\n" + "="*60)
print("RESULTATS DE LA MODELISATION")
print("="*60)
print(f"{'Modele':<25} {'AUC':>8} {'Gini':>8}")
print("-"*60)
print(f"{'Logistic Regression':<25} {auc_logit:>8.4f} {gini_logit:>8.4f}")
print(f"{'Random Forest':<25} {auc_rf:>8.4f} {gini_rf:>8.4f}")
print(f"{'XGBoost':<25} {auc_xgb:>8.4f} {gini_xgb:>8.4f}")
print("="*60)

# %% 8. VISUALISATION DES RESULTATS
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ROC
RocCurveDisplay.from_predictions(y_test, proba_logit, ax=axes[0,0], name='Logistic')
RocCurveDisplay.from_predictions(y_test, proba_rf, ax=axes[0,0], name='Random Forest')
RocCurveDisplay.from_predictions(y_test, proba_xgb, ax=axes[0,0], name='XGBoost')
axes[0,0].plot([0,1], [0,1], 'k--', label='Aleatoire')
axes[0,0].set_title('Courbes ROC')
axes[0,0].legend()

# Feature Importance RF
importances = pd.Series(pipe_rf.named_steps['clf'].feature_importances_, index=feature_names)
top_features = importances.sort_values(ascending=False).head(10)
sns.barplot(x=top_features.values, y=top_features.index, ax=axes[0,1], palette='viridis')
axes[0,1].set_title('Top 10 — Importance Random Forest')

# Distribution des probas XGBoost
df_test = X_test.copy()
df_test['proba_defaut'] = proba_xgb
df_test['defaut'] = y_test.values
sns.histplot(data=df_test, x='proba_defaut', hue='defaut', bins=30, kde=True, ax=axes[1,0])
axes[1,0].set_title('Distribution des Probabilites (XGBoost)')

# Matrice de confusion XGBoost
cm = confusion_matrix(y_test, (proba_xgb > 0.45).astype(int))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1])
axes[1,1].set_title('Matrice de Confusion — XGBoost')
axes[1,1].set_xlabel('Prediit')
axes[1,1].set_ylabel('Reel')

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Graphiques sauvegardes: model_evaluation.png")

# %% 9. SHAP — INTERPRETABILITE
explainer = shap.TreeExplainer(xgb_clf)
shap_values = explainer.shap_values(X_test_prep)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_prep, feature_names=feature_names, show=False)
plt.title("SHAP — Importance des features (XGBoost)")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ SHAP sauvegarde: shap_summary.png")

# %% 10. SAUVEGARDE DES MODELES
joblib.dump(pipe_rf, "../models/credit_scoring_model.pkl")
joblib.dump(xgb_clf, "../models/credit_scoring_model_xgb.pkl")
joblib.dump(preprocessor, "../models/preprocessor.pkl")
joblib.dump(explainer, "../models/shap_explainer.pkl")

metadata = {
    "model_type": "RandomForest", "auc": float(auc_rf), "gini": float(gini_rf),
    "numeric_features": numeric_features, "categorical_features": categorical_features,
    "feature_names": list(feature_names), "version": "1.0.0"
}
with open("../models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

metadata_xgb = {
    "model_type": "XGBoost", "auc": float(auc_xgb), "gini": float(gini_xgb),
    "numeric_features": numeric_features, "categorical_features": categorical_features,
    "feature_names": list(feature_names), "version": "2.0.0", "shap_enabled": True
}
with open("../models/metadata_xgb.json", "w") as f:
    json.dump(metadata_xgb, f, indent=2)

print("\n✅ TOUS LES MODELES SAUVEGARDES")
print("   ../models/credit_scoring_model.pkl")
print("   ../models/credit_scoring_model_xgb.pkl")
print("   ../models/preprocessor.pkl")
print("   ../models/shap_explainer.pkl")
print("   ../models/metadata.json")
print("   ../models/metadata_xgb.json")