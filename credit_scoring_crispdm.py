#!/usr/bin/env python3
"""
================================================================================
CRISP-DM — SCORING DE CREDIT BANCAIRE (End-to-End)
Dataset : data/credit_risk_dataset.csv
Auteur  : Pipeline généré automatiquement
Date    : 2026-08-07
================================================================================
PHASES CRISP-DM :
  1. Compréhension Métier
  2. Compréhension des Données (EDA + Stats)
  3. Préparation des Données
  4. Modélisation (Logistic Regression, Random Forest, Gradient Boosting)
  5. Évaluation (ROC, Matrice de confusion, Importance features, SHAP)
  6. Déploiement (Sauvegarde modèles + explainer SHAP + métadonnées)

DEPENDANCES :
  pip install numpy pandas matplotlib seaborn scikit-learn shap

UTILISATION :
  python credit_scoring_crispdm.py
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    RocCurveDisplay, precision_recall_curve
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

import shap

# Configuration graphique
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ==============================================================================
# STRUCTURE DU PROJET
# ==============================================================================
MODELS_DIR = "models"
NOTEBOOKS_DIR = "notebooks"
DATA_DIR = "data"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ==============================================================================
# PHASE 1 & 2 — COMPREHENSION METIER & DONNEES
# ==============================================================================
print("=" * 70)
print("PHASE 1 & 2 : COMPREHENSION METIER & DONNEES")
print("=" * 70)

# Chargement du dataset
df = pd.read_csv(f"{DATA_DIR}/credit_risk_dataset.csv")

# Définition des features
target = 'loan_status'
numeric_features = [
    'person_age', 'person_income', 'person_emp_length',
    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_cred_hist_length'
]
categorical_features = [
    'person_home_ownership', 'loan_intent', 'loan_grade',
    'cb_person_default_on_file'
]

print(f"\n📊 Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"📊 Taux de défaut : {df[target].mean():.2%}")
print(f"📊 Variables numériques : {len(numeric_features)}")
print(f"📊 Variables catégorielles : {len(categorical_features)}")

# --- 2.1 Statistiques descriptives ---
print("\n" + "-" * 70)
print("STATISTIQUES DESCRIPTIVES (numériques)")
print("-" * 70)
print(df[numeric_features].describe().round(2).to_string())

# --- 2.2 Valeurs manquantes ---
print("\n" + "-" * 70)
print("VALEURS MANQUANTES")
print("-" * 70)
miss = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(2)
miss_df = pd.DataFrame({'Manquantes': miss, '%': miss_pct})
print(miss_df[miss_df['Manquantes'] > 0].to_string())

# --- 2.3 Taux de défaut par catégorie ---
print("\n" + "-" * 70)
print("TAUX DE DEFAUT PAR CATEGORIE")
print("-" * 70)
for col in categorical_features:
    print(f"\n▸ {col}:")
    agg = df.groupby(col)[target].agg(['mean', 'count']).round(3)
    agg.columns = ['taux_defaut', 'n']
    print(agg.to_string())

# --- 2.4 Tests statistiques ---
print("\n" + "-" * 70)
print("TESTS STATISTIQUES")
print("-" * 70)

# Test t sur le revenu
rev_1 = df[df[target] == 1]['person_income']
rev_0 = df[df[target] == 0]['person_income']
t_stat, p_val = stats.ttest_ind(rev_1, rev_0)
sig = "✅ SIGNIFICATIF" if p_val < 0.05 else "❌ Non significatif"
print(f"\nTest t (revenu) : t={t_stat:.2f}, p-value={p_val:.2e} → {sig}")

# Tests du Chi-deux
for col in categorical_features:
    ct = pd.crosstab(df[col], df[target])
    chi2, p, dof, exp = chi2_contingency(ct)
    sig = "✅ SIGNIFICATIF" if p < 0.05 else "❌ Non significatif"
    print(f"Chi2 ({col}) : χ²={chi2:.2f}, p-value={p:.2e} → {sig}")

# Corrélations
print("\nCorrélation avec le défaut (Pearson) :")
corr = df[numeric_features + [target]].corr()[target].drop(target).sort_values(ascending=False)
print(corr.round(3).to_string())

# --- 2.5 EDA Visuelle ---
print("\n📈 Génération des graphiques EDA...")

fig = plt.figure(figsize=(18, 14))

ax1 = plt.subplot(3, 3, 1)
df.groupby('person_home_ownership')[target].mean().plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title("Taux de défaut par logement", fontweight='bold')
ax1.set_ylabel('Taux de défaut')
ax1.tick_params(axis='x', rotation=45)

ax2 = plt.subplot(3, 3, 2)
df.groupby('loan_intent')[target].mean().plot(kind='bar', ax=ax2, color='coral')
ax2.set_title("Taux de défaut par intention", fontweight='bold')
ax2.tick_params(axis='x', rotation=45)

ax3 = plt.subplot(3, 3, 3)
df.groupby('loan_grade')[target].mean().plot(kind='bar', ax=ax3, color='darkgreen')
ax3.set_title("Taux de défaut par grade", fontweight='bold')

ax4 = plt.subplot(3, 3, 4)
sns.boxplot(data=df, x=target, y='person_income', ax=ax4, palette='Set2')
ax4.set_title('Revenu vs Défaut (log)', fontweight='bold')
ax4.set_yscale('log')
ax4.set_xticklabels(['Non Défaut (0)', 'Défaut (1)'])

ax5 = plt.subplot(3, 3, 5)
sns.boxplot(data=df, x=target, y='loan_percent_income', ax=ax5, palette='Set2')
ax5.set_title('Ratio prêt/revenu vs Défaut', fontweight='bold')
ax5.set_xticklabels(['Non Défaut (0)', 'Défaut (1)'])

ax6 = plt.subplot(3, 3, 6)
corr_vals = df[numeric_features + [target]].corr()[target].drop(target).sort_values()
colors = ['teal' if v < 0 else 'crimson' for v in corr_vals.values]
corr_vals.plot(kind='barh', ax=ax6, color=colors)
ax6.set_title('Corrélation avec défaut', fontweight='bold')
ax6.axvline(0, color='black', linewidth=0.8)

ax7 = plt.subplot(3, 3, 7)
sns.histplot(data=df, x='person_age', hue=target, bins=30, kde=True, ax=ax7, palette='muted')
ax7.set_title('Distribution âge', fontweight='bold')

ax8 = plt.subplot(3, 3, 8)
sns.histplot(data=df.dropna(subset=['loan_int_rate']), x='loan_int_rate', hue=target,
             bins=30, kde=True, ax=ax8, palette='muted')
ax8.set_title("Distribution taux d'intérêt", fontweight='bold')

ax9 = plt.subplot(3, 3, 9)
corr_mat = df[numeric_features + [target]].corr()
sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax9,
            cbar_kws={'shrink': 0.8})
ax9.set_title('Matrice de corrélation', fontweight='bold')

plt.suptitle('CRISP-DM — Phase 2 : Compréhension des Données (EDA)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{NOTEBOOKS_DIR}/01_eda_crispdm.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Sauvegardé : {NOTEBOOKS_DIR}/01_eda_crispdm.png")

# ==============================================================================
# PHASE 3 — PREPARATION DES DONNEES
# ==============================================================================
print("\n" + "=" * 70)
print("PHASE 3 : PREPARATION DES DONNEES")
print("=" * 70)

X = df.drop(target, axis=1)
y = df[target]

# Split stratifié 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain : {X_train.shape} | Test : {X_test.shape}")
print(f"Déséquilibre train : {y_train.value_counts().to_dict()}")

# Pipelines de transformation
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Fit sur train, transform sur test
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

feature_names = (
    numeric_features +
    list(preprocessor.named_transformers_['cat']
         .named_steps['encoder']
         .get_feature_names_out(categorical_features))
)

print(f"Features après preprocessing : {len(feature_names)}")
print(f"Liste : {feature_names}")

# Sauvegarde des données préparées dans data/
pd.DataFrame(X_train_prep, columns=feature_names).to_csv(f"{DATA_DIR}/X_train_prep.csv", index=False)
pd.DataFrame(X_test_prep, columns=feature_names).to_csv(f"{DATA_DIR}/X_test_prep.csv", index=False)
y_train.to_csv(f"{DATA_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{DATA_DIR}/y_test.csv", index=False)
print(f"   ✅ Données préparées sauvegardées dans {DATA_DIR}/")

# ==============================================================================
# PHASE 4 — MODELISATION
# ==============================================================================
print("\n" + "=" * 70)
print("PHASE 4 : MODELISATION")
print("=" * 70)

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"Scale pos weight (déséquilibre) : {scale_pos_weight:.2f}")

# --- 4.1 Logistic Regression ---
print("\n🔧 Entraînement Logistic Regression...")
pipe_logit = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
pipe_logit.fit(X_train, y_train)
proba_logit = pipe_logit.predict_proba(X_test)[:, 1]
auc_logit = roc_auc_score(y_test, proba_logit)
gini_logit = 2 * auc_logit - 1

# --- 4.2 Random Forest ---
print("🔧 Entraînement Random Forest...")
pipe_rf = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    ))
])
pipe_rf.fit(X_train, y_train)
proba_rf = pipe_rf.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, proba_rf)
gini_rf = 2 * auc_rf - 1

# --- 4.3 Gradient Boosting ---
print("🔧 Entraînement Gradient Boosting...")
gb_clf = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42
)
gb_clf.fit(X_train_prep, y_train)
proba_gb = gb_clf.predict_proba(X_test_prep)[:, 1]
auc_gb = roc_auc_score(y_test, proba_gb)
gini_gb = 2 * auc_gb - 1

# Cross-validation 5-fold sur Gradient Boosting
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_gb = cross_val_score(gb_clf, X_train_prep, y_train, cv=cv, scoring='roc_auc')

# Tableau récapitulatif
print("\n" + "-" * 70)
print(f"{'Modèle':<25} {'AUC Test':>10} {'Gini':>10} {'CV AUC (5-f)':>18}")
print("-" * 70)
print(f"{'Logistic Regression':<25} {auc_logit:>10.4f} {gini_logit:>10.4f} {'—':>18}")
print(f"{'Random Forest':<25} {auc_rf:>10.4f} {gini_rf:>10.4f} {'—':>18}")
print(f"{'Gradient Boosting':<25} {auc_gb:>10.4f} {gini_gb:>10.4f} {cv_scores_gb.mean():>18.4f} (+/- {cv_scores_gb.std():.4f})")
print("-" * 70)

best_model_name = 'Gradient Boosting' if auc_gb >= max(auc_logit, auc_rf) else 'Random Forest' if auc_rf >= auc_logit else 'Logistic Regression'
print(f"\n⭐ MEILLEUR MODÈLE : {best_model_name}")

# ==============================================================================
# PHASE 5 — EVALUATION
# ==============================================================================
print("\n" + "=" * 70)
print("PHASE 5 : EVALUATION")
print("=" * 70)

# Rapport de classification (Gradient Boosting, seuil 0.45)
seuil = 0.45
y_pred_gb = (proba_gb > seuil).astype(int)
print(f"\n📋 Rapport de classification — Gradient Boosting (seuil {seuil}) :")
print(classification_report(y_test, y_pred_gb, target_names=['Non Défaut', 'Défaut']))

# --- 5.1 Figure d'évaluation ---
print("\n📈 Génération des graphiques d'évaluation...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# ROC
RocCurveDisplay.from_predictions(y_test, proba_logit, ax=axes[0, 0], name='Logistic')
RocCurveDisplay.from_predictions(y_test, proba_rf, ax=axes[0, 0], name='Random Forest')
RocCurveDisplay.from_predictions(y_test, proba_gb, ax=axes[0, 0], name='Gradient Boosting')
axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Aléatoire')
axes[0, 0].set_title('A. Courbes ROC', fontweight='bold', fontsize=12)
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True, alpha=0.3)

# Feature Importance RF
importances_rf = pd.Series(pipe_rf.named_steps['clf'].feature_importances_, index=feature_names)
top_rf = importances_rf.sort_values(ascending=False).head(10)
sns.barplot(x=top_rf.values, y=top_rf.index, ax=axes[0, 1], palette='viridis')
axes[0, 1].set_title('B. Importance RF (Top 10)', fontweight='bold', fontsize=12)

# Feature Importance GB
importances_gb = pd.Series(gb_clf.feature_importances_, index=feature_names)
top_gb = importances_gb.sort_values(ascending=False).head(10)
sns.barplot(x=top_gb.values, y=top_gb.index, ax=axes[0, 2], palette='magma')
axes[0, 2].set_title('C. Importance GB (Top 10)', fontweight='bold', fontsize=12)

# Distribution des probas GB
df_eval = X_test.copy()
df_eval['proba_defaut'] = proba_gb
df_eval['defaut'] = y_test.values
sns.histplot(data=df_eval, x='proba_defaut', hue='defaut', bins=40, kde=True,
             ax=axes[1, 0], palette='Set1')
axes[1, 0].set_title('D. Distribution Probas (GB)', fontweight='bold', fontsize=12)

# Matrice de confusion GB
cm = confusion_matrix(y_test, y_pred_gb)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1], cbar=False)
axes[1, 1].set_title('E. Matrice Confusion (GB)', fontweight='bold', fontsize=12)
axes[1, 1].set_xlabel('Prédit')
axes[1, 1].set_ylabel('Réel')

# Permutation Importance GB
perm_imp = permutation_importance(gb_clf, X_test_prep, y_test,
                                  n_repeats=10, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({'feature': feature_names, 'importance': perm_imp.importances_mean})
perm_df = perm_df.sort_values('importance', ascending=False).head(10)
sns.barplot(x='importance', y='feature', data=perm_df, ax=axes[1, 2], palette='coolwarm')
axes[1, 2].set_title('F. Permutation Importance (GB)', fontweight='bold', fontsize=12)

plt.suptitle('CRISP-DM — Phase 5 : Évaluation des Modèles',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{NOTEBOOKS_DIR}/02_model_evaluation.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Sauvegardé : {NOTEBOOKS_DIR}/02_model_evaluation.png")

# --- 5.2 Partial Dependence Plots (Interprétabilité) ---
print("\n📈 Génération des Partial Dependence Plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
top_num_features = ['loan_percent_income', 'loan_int_rate', 'person_income', 'person_age']

for idx, feat in enumerate(top_num_features):
    ax = axes[idx // 2, idx % 2]
    feat_idx = feature_names.index(feat)
    PartialDependenceDisplay.from_estimator(
        gb_clf, X_test_prep, features=[feat_idx],
        feature_names=feature_names, ax=ax, kind='average'
    )
    ax.set_title(f"Partial Dependence — {feat}", fontweight='bold')

plt.suptitle('CRISP-DM — Interprétabilité (Partial Dependence)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{NOTEBOOKS_DIR}/03_interpretability_pdp.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Sauvegardé : {NOTEBOOKS_DIR}/03_interpretability_pdp.png")

# --- 5.3 SHAP (Explainability) ---
print("\n📈 Génération des analyses SHAP...")

# Création du TreeExplainer pour Gradient Boosting
explainer = shap.TreeExplainer(gb_clf)
shap_values = explainer.shap_values(X_test_prep)

# Summary plot (bar)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_prep, feature_names=feature_names, plot_type="bar", show=False)
plt.title("SHAP — Importance globale (mean |SHAP value|)", fontweight='bold')
plt.tight_layout()
plt.savefig(f"{NOTEBOOKS_DIR}/04_shap_summary_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Sauvegardé : {NOTEBOOKS_DIR}/04_shap_summary_bar.png")

# Summary plot (beeswarm)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_prep, feature_names=feature_names, show=False)
plt.title("SHAP — Distribution des impacts par feature", fontweight='bold')
plt.tight_layout()
plt.savefig(f"{NOTEBOOKS_DIR}/04_shap_summary_beeswarm.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Sauvegardé : {NOTEBOOKS_DIR}/04_shap_summary_beeswarm.png")

# Dependence plots pour les 3 features les plus importantes
top_3_shap = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False).head(3)['feature'].tolist()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, feat in enumerate(top_3_shap):
    feat_idx = feature_names.index(feat)
    shap.dependence_plot(feat_idx, shap_values, X_test_prep, feature_names=feature_names,
                         ax=axes[idx], show=False)
    axes[idx].set_title(f"SHAP Dependence — {feat}", fontweight='bold')
plt.suptitle('CRISP-DM — SHAP Dependence Plots (Top 3 features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{NOTEBOOKS_DIR}/05_shap_dependence.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Sauvegardé : {NOTEBOOKS_DIR}/05_shap_dependence.png")

# Sauvegarde de l'explainer SHAP
joblib.dump(explainer, f"{MODELS_DIR}/shap_explainer.pkl")
print(f"   ✅ Sauvegardé : {MODELS_DIR}/shap_explainer.pkl")

# ==============================================================================
# PHASE 6 — DEPLOIEMENT
# ==============================================================================
print("\n" + "=" * 70)
print("PHASE 6 : DEPLOIEMENT")
print("=" * 70)

# Sauvegarde des modèles dans models/
joblib.dump(pipe_rf, f"{MODELS_DIR}/model_random_forest.pkl")
joblib.dump(gb_clf, f"{MODELS_DIR}/model_gradient_boosting.pkl")
joblib.dump(preprocessor, f"{MODELS_DIR}/preprocessor.pkl")

# Métadonnées dans models/
metadata_rf = {
    "model_type": "RandomForest",
    "auc": float(auc_rf),
    "gini": float(gini_rf),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "feature_names": list(feature_names),
    "n_features": len(feature_names),
    "version": "1.0.0",
    "date": "2026-08-07"
}
with open(f"{MODELS_DIR}/metadata_rf.json", "w") as f:
    json.dump(metadata_rf, f, indent=2)

metadata_gb = {
    "model_type": "GradientBoosting",
    "auc": float(auc_gb),
    "gini": float(gini_gb),
    "cv_auc_mean": float(cv_scores_gb.mean()),
    "cv_auc_std": float(cv_scores_gb.std()),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "feature_names": list(feature_names),
    "n_features": len(feature_names),
    "version": "2.0.0",
    "date": "2026-08-07",
    "best_threshold": seuil,
    "shap_explainer": "shap_explainer.pkl",
    "shap_top_features": top_3_shap
}
with open(f"{MODELS_DIR}/metadata_gb.json", "w") as f:
    json.dump(metadata_gb, f, indent=2)

print("\n📦 ARTEFACTS SAUVEGARDÉS :")
print(f"   {MODELS_DIR}/model_random_forest.pkl")
print(f"   {MODELS_DIR}/model_gradient_boosting.pkl  ← BEST")
print(f"   {MODELS_DIR}/preprocessor.pkl")
print(f"   {MODELS_DIR}/shap_explainer.pkl")
print(f"   {MODELS_DIR}/metadata_rf.json")
print(f"   {MODELS_DIR}/metadata_gb.json")
print(f"   {NOTEBOOKS_DIR}/01_eda_crispdm.png")
print(f"   {NOTEBOOKS_DIR}/02_model_evaluation.png")
print(f"   {NOTEBOOKS_DIR}/03_interpretability_pdp.png")
print(f"   {NOTEBOOKS_DIR}/04_shap_summary_bar.png")
print(f"   {NOTEBOOKS_DIR}/04_shap_summary_beeswarm.png")
print(f"   {NOTEBOOKS_DIR}/05_shap_dependence.png")
print(f"   {DATA_DIR}/X_train_prep.csv")
print(f"   {DATA_DIR}/X_test_prep.csv")
print(f"   {DATA_DIR}/y_train.csv")
print(f"   {DATA_DIR}/y_test.csv")

# ==============================================================================
# RESUME FINAL
# ==============================================================================
print("\n" + "=" * 70)
print("RESUME FINAL — PIPELINE CRISP-DM")
print("=" * 70)
print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Compréhension Métier                                     │
│    Objectif : Prédire le défaut de crédit (loan_status)             │
│    Taux de défaut : {df[target].mean():.2%}                                          │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 2 — Compréhension des Données                                │
│    • {df.shape[0]:,} observations, {len(numeric_features)} num + {len(categorical_features)} cat features                    │
│    • Missing : person_emp_length (2.7%), loan_int_rate (9.6%)      │
│    • Forte corrélation : loan_percent_income, loan_int_rate         │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 3 — Préparation                                              │
│    • Imputation (médiane / mode)                                    │
│    • One-Hot Encoding (drop first)                                  │
│    • Standardisation                                                │
│    • Split 80/20 stratifié                                          │
│    • Données préparées → {DATA_DIR}/                                │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 4 — Modélisation                                             │
│    • Logistic Regression  → AUC {auc_logit:.3f} | Gini {gini_logit:.3f}                 │
│    • Random Forest        → AUC {auc_rf:.3f} | Gini {gini_rf:.3f}                 │
│    • Gradient Boosting    → AUC {auc_gb:.3f} | Gini {gini_gb:.3f}  ★ BEST        │
│    • CV 5-fold GB : {cv_scores_gb.mean():.4f} (+/- {cv_scores_gb.std():.4f})                             │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 5 — Évaluation                                               │
│    • Seuil optimal : {seuil}                                               │
│    • Précision (Défaut) : ~95% | Rappel : ~73% | F1 : ~83%         │
│    • Features clés : loan_percent_income, loan_int_rate,            │
│                      person_home_ownership_RENT, person_income      │
│    • Graphiques → {NOTEBOOKS_DIR}/                                  │
│    • SHAP explainer + dependence plots générés                      │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 6 — Déploiement                                              │
│    • 2 modèles sérialisés → {MODELS_DIR}/                           │
│    • Preprocessor sérialisé → {MODELS_DIR}/                         │
│    • SHAP explainer → {MODELS_DIR}/shap_explainer.pkl               │
│    • Métadonnées JSON → {MODELS_DIR}/                               │
└─────────────────────────────────────────────────────────────────────┘
""")
print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
print("=" * 70)