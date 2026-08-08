# =============================================================================
#  Application Web Scoring Credit v2.0 — Streamlit
# =============================================================================
#  Auteur      : Karl Bifu Batunguni - Data scientist
#  Organisation: Bifu Albert Bank
#  Date        : 2026-08-07
#  Version     : 2.0.0
#  Licence     : MIT
#
#  Copyright (c) 2026 Bifu Albert Bank. Tous droits reserves.
# =============================================================================
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Scoring Credit v2.0 — Bifu Albert Bank",
    page_icon="🏦",
    layout="wide"
)

API_URL = st.sidebar.text_input("URL API", value="https://scoring-credit-e2e.onrender.com")

# ─── AUTH ───────────────────────────────────────────────────
if "token" not in st.session_state:
    st.session_state.token = None
    st.session_state.role = None

with st.sidebar:
    st.header("🔐 Authentification")
    if not st.session_state.token:
        username = st.text_input("Utilisateur", value="analyste")
        password = st.text_input("Mot de passe", type="password", value="analyste123")
        if st.button("Se connecter"):
            try:
                resp = requests.post(f"{API_URL}/login", json={"username": username, "password": password})
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.token = data["access_token"]
                    st.session_state.role = data["role"]
                    st.success(f"Connecte en tant que {data['role']}")
                    st.rerun()
                else:
                    st.error("Identifiants invalides")
            except Exception as e:
                st.error(f"Erreur: {e}")
    else:
        st.success(f"Connecte ({st.session_state.role})")
        if st.button("Deconnexion"):
            st.session_state.token = None
            st.session_state.role = None
            st.rerun()

    st.markdown("---")
    st.caption("© 2026 Bifu Albert Bank | v2.0.0")

headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}

tabs_list = ["🏠 Accueil", "📝 Evaluation", "📁 Batch", "📊 Audit"]
if st.session_state.role == "superviseur":
    tabs_list.append("🔧 Admin")

tab = st.tabs(tabs_list)

# ─── TAB 0 : ACCUEIL ───────────────────────────────────────
with tab[0]:
    st.title("🏦 Scoring de Credit — Bifu Albert Bank v2.0")
    st.markdown("""
    ### Systeme de scoring de credit en production

    **Fonctionnalites :**
    - ✅ Modele Gradient Boosting (AUC optimise)
    - ✅ Explicabilite SHAP pour chaque prediction
    - ✅ Authentification JWT (Analyste / Superviseur)
    - ✅ Audit complet des predictions (SQLite)
    - ✅ Traitement batch par fichier CSV
    - ✅ Dashboard de monitoring
    """)

    try:
        resp = requests.get(f"{API_URL}/")
        if resp.status_code == 200:
            info = resp.json()
            col1, col2, col3 = st.columns(3)
            col1.metric("Modele", info.get("model", "N/A"))
            col2.metric("AUC", f"{info.get('auc', 0):.3f}")
            col3.metric("Gini", f"{info.get('gini', 0):.3f}")
            st.caption(f"© {info.get('copyright', '2026 Bifu Albert Bank')} | Auteur: {info.get('author', 'Karl Bifu Batunguni - Data scientist')}")
    except Exception:
        st.warning("API non accessible")

# ─── TAB 1 : EVALUATION ──────────────────────────────────
with tab[1]:
    if not st.session_state.token:
        st.warning("Veuillez vous connecter")
        st.stop()

    st.header("Evaluation individuelle avec SHAP")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 18, 100, 35)
        situation = st.selectbox("Situation", ["Celibataire", "Marie", "Divorce", "Veuf"])
        enfants = st.number_input("Enfants", 0, 20, 0)
        niveau = st.selectbox("Niveau", ["Primaire", "Secondaire", "Superieur", "Aucun"])
    with col2:
        revenu = st.number_input("Revenu mensuel (CDF)", 1000, value=450000, step=1000)
        emploi = st.selectbox("Emploi", ["CDI", "CDD", "Independant", "Sans_emploi"])
        anciennete = st.number_input("Anciennete emploi (mois)", 0, 600, 48)
        secteur = st.selectbox("Secteur", ["Agriculture", "Commerce", "Services", "Industrie", "Public"])
    with col3:
        montant = st.number_input("Montant credit", 1000, value=2500000, step=1000)
        duree = st.selectbox("Duree (mois)", [12,24,36,48,60,72], index=2)
        endettement = st.slider("Taux endettement (%)", 0.0, 100.0, 25.0)
        credits_encours = st.number_input("Credits en cours", 0, 20, 0)
        historique = st.selectbox("Historique defaut", [0,1], format_func=lambda x: "Non" if x==0 else "Oui")
        compte_anc = st.number_input("Anciennete compte (mois)", 0, value=36)
        incidents = st.number_input("Incidents 12m", 0, 50, 0)
        revolving = st.slider("Utilisation revolving (%)", 0.0, 100.0, 15.0)

    if st.button("🔍 Evaluer", type="primary", use_container_width=True):
        payload = {
            "age": age, "revenu_mensuel": revenu, "montant_credit": montant,
            "duree_credit_mois": duree, "anciennete_emploi_mois": anciennete,
            "nb_credits_en_cours": credits_encours, "taux_endettement": endettement,
            "historique_defaut": historique, "type_emploi": emploi,
            "niveau_etude": niveau, "secteur_activite": secteur,
            "compte_bancaire_anciennete_mois": compte_anc,
            "nb_incidents_paiement_12m": incidents,
            "utilisation_credit_revolving": revolving,
            "nombre_enfants": enfants, "situation_matrimoniale": situation
        }

        with st.spinner("Analyse..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, headers=headers, timeout=10)
                if resp.status_code == 200:
                    r = resp.json()

                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Score", f"{r['score_credit']}/1000")
                    k2.metric("Probabilite defaut", f"{r['probabilite_defaut']:.1%}")
                    couleur = "🟢" if r['risque']=="Faible" else "🟡" if r['risque']=="Modere" else "🔴"
                    k3.metric("Decision", f"{couleur} {r['decision']}")
                    k4.metric("Version", r['model_version'])

                    # Jauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=r['score_credit'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Score Credit"},
                        gauge={'axis': {'range': [0, 1000]},
                               'bar': {'color': "darkblue"},
                               'steps': [
                                   {'range': [0, 400], 'color': "red"},
                                   {'range': [400, 600], 'color': "orange"},
                                   {'range': [600, 800], 'color': "yellow"},
                                   {'range': [800, 1000], 'color': "green"}],
                               'threshold': {'line': {'color': "black", 'width': 4},
                                            'thickness': 0.75, 'value': r['score_credit']}}
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                    # SHAP
                    st.subheader("📊 Explication SHAP — Pourquoi cette decision ?")
                    shap_vals = r['shap_explanation']
                    if shap_vals:
                        shap_df = pd.DataFrame({
                            'Feature': list(shap_vals.keys()),
                            'Impact': list(shap_vals.values())
                        }).sort_values('Impact', key=abs, ascending=False).head(10)

                        colors = ['green' if x < 0 else 'red' for x in shap_df['Impact']]
                        fig2 = go.Figure(go.Bar(
                            x=shap_df['Impact'],
                            y=shap_df['Feature'],
                            orientation='h',
                            marker_color=colors
                        ))
                        fig2.update_layout(
                            title="Impact de chaque variable sur la probabilite de defaut",
                            xaxis_title="SHAP value (rouge = augmente le risque)"
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                    with st.expander("Details techniques"):
                        st.write("**Modele :**", r['model_type'])
                        st.write("**Features :**", r['features_utilisees'])
                        st.write("**Seuil :**", r['seuil_utilise'])
                else:
                    st.error(f"Erreur {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Erreur: {e}")

# ─── TAB 2 : BATCH ───────────────────────────────────────
with tab[2]:
    if not st.session_state.token:
        st.warning("Veuillez vous connecter")
        st.stop()

    st.header("Evaluation par lot (CSV)")
    uploaded = st.file_uploader("Fichier CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"{len(df)} lignes")
        st.dataframe(df.head())

        if st.button("🚀 Lancer le scoring batch"):
            records = df.to_dict(orient='records')
            try:
                resp = requests.post(f"{API_URL}/predict/batch", json=records, headers=headers, timeout=120)
                if resp.status_code == 200:
                    results = resp.json()['predictions']
                    df['score'] = [r['score_credit'] for r in results]
                    df['proba'] = [r['probabilite_defaut'] for r in results]
                    df['decision'] = [r['decision'] for r in results]
                    df['risque'] = [r['risque'] for r in results]
                    st.dataframe(df)
                    st.bar_chart(df['risque'].value_counts())
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Telecharger", csv, "results.csv", "text/csv")
                else:
                    st.error(f"Erreur {resp.status_code}")
            except Exception as e:
                st.error(str(e))

# ─── TAB 3 : AUDIT ──────────────────────────────────────
with tab[3]:
    if not st.session_state.token:
        st.warning("Veuillez vous connecter")
        st.stop()

    st.header("📋 Audit des predictions")

    if st.session_state.role != "superviseur":
        st.info("Reserve aux superviseurs")
        st.stop()

    try:
        resp = requests.get(f"{API_URL}/audit/stats", headers=headers, timeout=10)
        if resp.status_code == 200:
            stats = resp.json()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total predictions", stats['total_predictions'])
            c2.metric("Acceptes", stats['acceptes'])
            c3.metric("Refuses", stats['refuses'])
            c4.metric("Taux acceptation", f"{stats['taux_acceptation']:.1%}")

        resp2 = requests.get(f"{API_URL}/audit/predictions?limit=50", headers=headers, timeout=10)
        if resp2.status_code == 200:
            logs = resp2.json()
            df_logs = pd.DataFrame(logs)
            st.dataframe(df_logs)
    except Exception as e:
        st.error(str(e))

# ─── TAB 4 : ADMIN ───────────────────────────────────────
if len(tab) > 4:
    with tab[4]:
        st.header("🔧 Administration")
        st.markdown("""
        ### Informations systeme
        - **Modele actif** : Gradient Boosting v2.0
        - **Base de donnees** : SQLite (dev) / PostgreSQL (prod)
        - **Authentification** : JWT Bearer
        - **Logging** : Toutes les predictions sont tracees avec SHAP values
        - **Copyright** : (c) 2026 Banque Centrale
        """)