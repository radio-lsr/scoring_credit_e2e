"""
Application Web de Scoring de Credit — Banque Centrale v2.0
Interface Streamlit pour consommer l'API de prediction
"""
import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Scoring Credit — Banque Centrale",
    page_icon="🏦",
    layout="wide"
)

API_URL = st.sidebar.text_input("URL de l'API", value="http://localhost:8000")

st.title("🏦 Scoring de Credit — Banque Centrale")
st.markdown("*Evaluation du risque de defaut en temps reel*")

st.sidebar.header("📊 Informations")
st.sidebar.info("""
Cette application consomme l'API de scoring de credit.
Assurez-vous que l'API est demarree :
```bash
uvicorn api.main:app --reload --port 8000
```
""")

try:
    resp = requests.get(f"{API_URL}/", timeout=5)
    if resp.status_code == 200:
        data = resp.json()
        st.sidebar.metric("AUC", f"{data.get('model_auc', 'N/A'):.3f}")
        st.sidebar.metric("Gini", f"{data.get('model_gini', 'N/A'):.3f}")
        st.sidebar.caption(f"Modele : {data.get('model_type', 'N/A')}")
except Exception:
    st.sidebar.warning("API non accessible")

tab1, tab2, tab3 = st.tabs(["📝 Nouvelle Evaluation", "📁 Evaluation par Fichier", "📚 Documentation"])

# ==============================================================================
# UTILITAIRES D'AFFICHAGE
# ==============================================================================
def badge_risque(risque: str):
    couleurs = {
        "Faible": ("🟢", "#d4edda", "#155724"),
        "Modere": ("🟡", "#fff3cd", "#856404"),
        "Eleve": ("🟠", "#f8d7da", "#721c24"),
        "Tres eleve": ("🔴", "#f5c6cb", "#721c24")
    }
    emoji, bg, fg = couleurs.get(risque, ("⚪", "#e2e3e5", "#383d41"))
    return f'<span style="background-color:{bg};color:{fg};padding:4px 12px;border-radius:12px;font-weight:600;font-size:0.9em;">{emoji} {risque}</span>'

def badge_decision(decision: str):
    if "ACCEPTE" in decision and "CAUTION" not in decision:
        return f'<span style="background-color:#d4edda;color:#155724;padding:4px 12px;border-radius:12px;font-weight:600;">✅ {decision}</span>'
    elif "CAUTION" in decision:
        return f'<span style="background-color:#fff3cd;color:#856404;padding:4px 12px;border-radius:12px;font-weight:600;">⚠️ {decision}</span>'
    else:
        return f'<span style="background-color:#f8d7da;color:#721c24;padding:4px 12px;border-radius:12px;font-weight:600;">❌ {decision}</span>'

def jauge_score(score: int):
    if score >= 800:
        couleur = "linear-gradient(90deg, #28a745 0%, #28a745 100%)"
    elif score >= 600:
        couleur = "linear-gradient(90deg, #ffc107 0%, #ffc107 100%)"
    elif score >= 400:
        couleur = "linear-gradient(90deg, #fd7e14 0%, #fd7e14 100%)"
    else:
        couleur = "linear-gradient(90deg, #dc3545 0%, #dc3545 100%)"
    pct = score / 10
    return f"""
    <div style="width:100%; background-color:#2d2d2d; border-radius:10px; height:24px; margin:8px 0; position:relative;">
        <div style="width:{pct}%; background:{couleur}; height:24px; border-radius:10px; transition: width 0.5s ease;"></div>
        <div style="position:absolute; top:0; left:0; width:100%; text-align:center; line-height:24px; color:white; font-weight:bold; font-size:0.85em;">
            {score} / 1000
        </div>
    </div>
    """

def feature_tag(feature: str):
    return f'<span style="background-color:#1f2937;color:#e5e7eb;padding:3px 8px;border-radius:6px;font-size:0.75em;margin:2px;display:inline-block;border:1px solid #374151;">{feature}</span>'

# ==============================================================================
# TAB 1 — EVALUATION INDIVIDUELLE
# ==============================================================================
with tab1:
    st.header("Evaluation individuelle d'un emprunteur")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Identite")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        situation_matrimoniale = st.selectbox("Situation matrimoniale", 
                                               ["Celibataire", "Marie", "Divorce", "Veuf"])
        nombre_enfants = st.number_input("Nombre d'enfants", min_value=0, max_value=20, value=0)
        niveau_etude = st.selectbox("Niveau d'etudes", 
                                     ["Primaire", "Secondaire", "Superieur", "Aucun"])

    with col2:
        st.subheader("💼 Emploi & Revenus")
        revenu_mensuel = st.number_input("Revenu mensuel (CDF)", min_value=1000, value=450000, step=1000)
        type_emploi = st.selectbox("Type d'emploi", 
                                    ["CDI", "CDD", "Independant", "Sans_emploi"])
        anciennete_emploi_mois = st.number_input("Anciennete emploi (mois)", min_value=0, max_value=600, value=48)
        secteur_activite = st.selectbox("Secteur d'activite", 
                                         ["Agriculture", "Commerce", "Services", "Industrie", "Public"])

    with col3:
        st.subheader("💳 Credit & Banque")
        montant_credit = st.number_input("Montant du credit (CDF)", min_value=1000, value=2500000, step=1000)
        duree_credit_mois = st.selectbox("Duree (mois)", [12, 24, 36, 48, 60, 72], index=2)
        taux_endettement = st.slider("Taux d'endettement (%)", 0.0, 100.0, 25.0)
        nb_credits_en_cours = st.number_input("Credits en cours", min_value=0, max_value=20, value=0)
        historique_defaut = st.selectbox("Historique de defaut", [0, 1], format_func=lambda x: "Non" if x==0 else "Oui")
        compte_bancaire_anciennete_mois = st.number_input("Anciennete compte (mois)", min_value=0, value=36)
        nb_incidents_paiement_12m = st.number_input("Incidents paiement 12m", min_value=0, max_value=50, value=0)
        utilisation_credit_revolving = st.slider("Utilisation credit revolving (%)", 0.0, 100.0, 15.0)

    if st.button("🔍 Evaluer le risque", type="primary", use_container_width=True):
        payload = {
            "age": age,
            "revenu_mensuel": revenu_mensuel,
            "montant_credit": montant_credit,
            "duree_credit_mois": duree_credit_mois,
            "anciennete_emploi_mois": anciennete_emploi_mois,
            "nb_credits_en_cours": nb_credits_en_cours,
            "taux_endettement": taux_endettement,
            "historique_defaut": historique_defaut,
            "type_emploi": type_emploi,
            "niveau_etude": niveau_etude,
            "secteur_activite": secteur_activite,
            "compte_bancaire_anciennete_mois": compte_bancaire_anciennete_mois,
            "nb_incidents_paiement_12m": nb_incidents_paiement_12m,
            "utilisation_credit_revolving": utilisation_credit_revolving,
            "nombre_enfants": nombre_enfants,
            "situation_matrimoniale": situation_matrimoniale
        }

        with st.spinner("Analyse en cours..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                if response.status_code == 200:
                    result = response.json()

                    # ------------------------------------------------------------------
                    # RESULTATS — DASHBOARD
                    # ------------------------------------------------------------------
                    st.markdown("---")
                    st.subheader("📊 Resultat de l'evaluation")

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Score Credit", f"{result['score_credit']}/1000")
                    with c2:
                        st.metric("Probabilite de defaut", f"{result['probabilite_defaut']:.1%}")
                    with c3:
                        st.markdown("**Decision**<br>" + badge_decision(result['decision']), unsafe_allow_html=True)
                    with c4:
                        st.markdown("**Niveau de risque**<br>" + badge_risque(result['risque']), unsafe_allow_html=True)

                    st.markdown(jauge_score(result['score_credit']), unsafe_allow_html=True)

                    # ------------------------------------------------------------------
                    # DETAILS TECHNIQUES
                    # ------------------------------------------------------------------
                    with st.expander("📋 Details techniques", expanded=True):
                        dt1, dt2 = st.columns(2)

                        with dt1:
                            st.markdown("**🧠 Modele utilise**")
                            info_text = f"**{result['model_type']}**  \nSeuil de decision : `{result['seuil_utilise']}`"
                            st.info(info_text)

                            st.markdown("**📐 Features transmises au modele**")
                            tags = " ".join([feature_tag(f) for f in result['features_utilisees']])
                            st.markdown(f'<div style="line-height:1.8;">{tags}</div>', unsafe_allow_html=True)

                        with dt2:
                            st.markdown("**📖 Interpretation du score**")
                            score = result['score_credit']
                            if score >= 800:
                                st.success("**800–1000** : Risque faible, excellent profil")
                            elif score >= 600:
                                st.warning("**600–799** : Risque modere, acceptable avec conditions")
                            elif score >= 400:
                                st.error("**400–599** : Risque eleve, refus recommande")
                            else:
                                st.error("**0–399** : Risque tres eleve, refus systematique")

                            st.markdown("**📉 Repartition du risque**")
                            p = result['probabilite_defaut']
                            st.progress(1 - p, text=f"Confiance non-defaut : {(1-p)*100:.1f}%")

                        # ------------------------------------------------------------------
                        # SHAP
                        # ------------------------------------------------------------------
                        try:
                            exp_resp = requests.post(f"{API_URL}/explain", json=payload, timeout=10)
                            if exp_resp.status_code == 200:
                                exp = exp_resp.json()
                                st.markdown("---")
                                st.markdown("**🔍 Interpretabilite SHAP — Top features influencant la decision**")

                                shap_col1, shap_col2 = st.columns(2)
                                with shap_col1:
                                    st.markdown("🟢 **Facteurs reduisant le risque**")
                                    for item in exp.get('top_negative', []):
                                        st.markdown(
                                            f'<div style="display:flex;align-items:center;margin:4px 0;">'
                                            f'<div style="width:8px;height:8px;background:#28a745;border-radius:50%;margin-right:8px;"></div>'
                                            f'<span style="font-size:0.9em;"><b>{item["feature"]}</b> : {item["impact"]:.4f}</span>'
                                            f'</div>',
                                            unsafe_allow_html=True
                                        )

                                with shap_col2:
                                    st.markdown("🔴 **Facteurs augmentant le risque**")
                                    for item in exp.get('top_positive', []):
                                        st.markdown(
                                            f'<div style="display:flex;align-items:center;margin:4px 0;">'
                                            f'<div style="width:8px;height:8px;background:#dc3545;border-radius:50%;margin-right:8px;"></div>'
                                            f'<span style="font-size:0.9em;"><b>{item["feature"]}</b> : +{item["impact"]:.4f}</span>'
                                            f'</div>',
                                            unsafe_allow_html=True
                                        )
                        except Exception:
                            pass

                    st.markdown("---")

                else:
                    st.error(f"Erreur API : {response.status_code} — {response.text}")
            except Exception as e:
                st.error(f"Erreur de connexion : {e}")

# ==============================================================================
# TAB 2 — BATCH
# ==============================================================================
with tab2:
    st.header("Evaluation par lot (fichier CSV)")
    uploaded_file = st.file_uploader("Deposez un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"📄 {len(df)} lignes detectees")
        st.dataframe(df.head())

        if st.button("🚀 Lancer l'evaluation batch", type="primary"):
            with st.spinner("Traitement en cours..."):
                records = df.to_dict(orient='records')
                try:
                    response = requests.post(f"{API_URL}/predict/batch", json=records, timeout=60)
                    if response.status_code == 200:
                        results = response.json()['predictions']
                        df['score_credit'] = [r['score_credit'] for r in results]
                        df['probabilite_defaut'] = [r['probabilite_defaut'] for r in results]
                        df['decision'] = [r['decision'] for r in results]
                        df['risque'] = [r['risque'] for r in results]
                        st.success("✅ Evaluation terminee !")
                        st.dataframe(df)
                        fig_col1, fig_col2 = st.columns(2)
                        with fig_col1:
                            st.bar_chart(df['risque'].value_counts())
                        with fig_col2:
                            st.bar_chart(df['decision'].value_counts())
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Telecharger les resultats", csv, "scoring_results.csv", "text/csv")
                    else:
                        st.error(f"Erreur API : {response.status_code}")
                except Exception as e:
                    st.error(f"Erreur : {e}")

# ==============================================================================
# TAB 3 — DOCUMENTATION
# ==============================================================================
with tab3:
    st.header("📚 Documentation de l'API")
    st.markdown(f"""
    ### Endpoints disponibles

    | Endpoint | Methode | Description |
    |----------|---------|-------------|
    | `/` | GET | Info API |
    | `/health` | GET | Health check |
    | `/predict` | POST | Prediction individuelle |
    | `/predict/batch` | POST | Prediction batch |
    | `/explain` | POST | Explication SHAP (si disponible) |

    ### Exemple de requete cURL
    ```bash
    curl -X POST "{API_URL}/predict" \
      -H "Content-Type: application/json" \
      -d '{{
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
      }}'
    ```
    """)