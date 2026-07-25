import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Immobilier Inde — Prédiction Vente & Location",
    page_icon="🏠",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    models_dir = 'models'
    model_vente    = joblib.load(f'{models_dir}/modele_vente.joblib')
    model_location = joblib.load(f'{models_dir}/modele_location.joblib')
    meta           = joblib.load(f'{models_dir}/meta.joblib')
    return model_vente, model_location, meta

try:
    model_vente, model_location, meta = load_artifacts()
except Exception as e:
    st.error(f"Erreur lors du chargement des modèles : {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# RÈGLES FEATURE ENGINEERING (conformes au sujet)
#   Nbre_chambre    = floor(surface / 100) + 1
#   Nbre_salle_bain = 2 + floor(surface / 200)
#   Nbre_etage      ∈ [1, 3]
# ─────────────────────────────────────────────────────────────────────────────
def chambres_auto(m2: int) -> int:
    return int(m2 // 100) + 1

def sdb_auto(m2: int) -> int:
    return 2 + int(m2 // 200)

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMANDATIONS PAR VILLE (issues de l'analyse top 25 % du notebook)
# ─────────────────────────────────────────────────────────────────────────────
RECOMMANDATIONS = {
    "Bangalore": {
        "surface":   120,
        "finition":  "Semi-Furnished",
        "chambres":  2,
        "sdb":       2,
        "rendement": 4.0,
    },
    "New Delhi": {
        "surface":   150,
        "finition":  "Semi-Furnished",
        "chambres":  2,
        "sdb":       2,
        "rendement": 4.0,
    },
    "Thane": {
        "surface":   100,
        "finition":  "Semi-Furnished",
        "chambres":  2,
        "sdb":       2,
        "rendement": 4.0,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# MISE EN PAGE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏠 Estimation Immobilière en Inde — Vente & Location")
st.markdown(
    "Prédiction du prix de vente et du loyer mensuel "
    "(**New Delhi · Bangalore · Thane**) — modèle Random Forest (200 arbres)."
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — SAISIE DES CARACTÉRISTIQUES
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🎯 Caractéristiques du bien")

city     = st.sidebar.selectbox("Ville", meta['cities'])
area_m2  = st.sidebar.slider("Surface (m²)", min_value=30, max_value=500, value=100, step=5)

# CORRECTION 1 : Nbre_chambre et Nbre_salle_bain calculés automatiquement
#   selon les règles imposées par le sujet, non modifiables manuellement
nbre_chambre    = chambres_auto(area_m2)
nbre_salle_bain = sdb_auto(area_m2)

st.sidebar.info(
    f"**Chambres calculées** : {nbre_chambre}  \n"
    f"**Salles de bain calculées** : {nbre_salle_bain}  \n"
    f"*(règles sujet : 1 ch/100 m², 1 SDB/200 m²)*"
)

# CORRECTION 2 : max_value=3 (le sujet impose Nbre_etage ∈ [1, 3])
nbre_etage = st.sidebar.slider("Nombre d'étages", min_value=1, max_value=3, value=2)

furnishing = st.sidebar.selectbox("Finition", meta['furnishings'])

# ─────────────────────────────────────────────────────────────────────────────
# PRÉDICTION
# ─────────────────────────────────────────────────────────────────────────────
input_data = pd.DataFrame([{
    'City'           : city,
    'Area_m2'        : area_m2,
    'Nbre_chambre'   : nbre_chambre,
    'Nbre_salle_bain': nbre_salle_bain,
    'Nbre_etage'     : nbre_etage,
    'Furnishing'     : furnishing,
}])

if st.sidebar.button("Lancer l'estimation", type="primary"):

    pred_vente    = model_vente.predict(input_data)[0]
    pred_location = model_location.predict(input_data)[0]
    rendement     = pred_location * 12 / pred_vente * 100 if pred_vente > 0 else 0
    amort_ans     = pred_vente / (pred_location * 12)   if pred_location > 0 else 0

    INR_TO_CFA = 7.55

    st.subheader("📊 Résultats de l'estimation")

    # CORRECTION 3 : labels en français
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Prix de vente estimé (CFA)",
            value=f"{pred_vente:,.0f} CFA",
            delta=f"{pred_vente / INR_TO_CFA:,.0f} INR"
        )
    with col2:
        st.metric(
            label="Loyer mensuel estimé (CFA)",
            value=f"{pred_location:,.0f} CFA",
            delta=f"{pred_location / INR_TO_CFA:,.0f} INR"
        )
    with col3:
        st.metric(
            label="Rendement locatif brut",
            value=f"{rendement:.2f} % / an",
            delta=f"Amortissement en {amort_ans:.1f} ans"
        )

    st.success("Estimation effectuée par le modèle Random Forest (R² ≈ 0,73).")

    with st.expander("🔍 Détails de la configuration saisie"):
        st.dataframe(
            input_data.rename(columns={
                'City': 'Ville', 'Area_m2': 'Surface (m²)',
                'Nbre_chambre': 'Chambres', 'Nbre_salle_bain': 'Salles de bain',
                'Nbre_etage': 'Étages', 'Furnishing': 'Finition'
            }),
            use_container_width=True, hide_index=True
        )

else:
    st.info("👈 Ajustez les paramètres dans la barre latérale et cliquez sur **Lancer l'estimation**.")

# ─────────────────────────────────────────────────────────────────────────────
# CORRECTION 4 : SECTION RECOMMANDATIONS (demandée par le sujet)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("💡 Recommandations pour le porteur de projet")
st.markdown(
    "Configurations optimales identifiées à partir des biens du **quartile supérieur** "
    "(top 25 % en prix de vente) de chaque ville :"
)

cols = st.columns(len(RECOMMANDATIONS))
for col, (ville, rec) in zip(cols, RECOMMANDATIONS.items()):
    with col:
        st.markdown(f"**{ville}**")
        st.markdown(
            f"- Surface cible : **{rec['surface']} m²**\n"
            f"- Finition recommandée : **{rec['finition']}**\n"
            f"- Chambres : **{rec['chambres']}**\n"
            f"- Salles de bain : **{rec['sdb']}**\n"
            f"- Rendement locatif : **{rec['rendement']:.1f} % / an**"
        )
        # Prédiction express pour la config recommandée
        try:
            ex = pd.DataFrame([{
                'City'           : ville,
                'Area_m2'        : rec['surface'],
                'Nbre_chambre'   : rec['chambres'],
                'Nbre_salle_bain': rec['sdb'],
                'Nbre_etage'     : 2,
                'Furnishing'     : rec['finition'],
            }])
            pv = model_vente.predict(ex)[0]
            pl = model_location.predict(ex)[0]
            st.caption(
                f"Prix estimé : {pv:,.0f} CFA  \n"
                f"Loyer estimé : {pl:,.0f} CFA/mois"
            )
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# PIED DE PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Groupe 3 — M1 IABD 2026 · ESP-UCAD · Dr. Mamadou Camara  \n"
    "Modèle : Random Forest (200 arbres) · Données : 187 000 annonces immobilières indiennes"
)
