import datetime as dt

import joblib as jb
import numpy as np
import pandas as pd
import streamlit as st


# ================================================================
# 0. CONFIGURATION DE LA PAGE
# ================================================================

st.set_page_config(
    page_title="CoinAfrique AI - Intelligence Prédictive",
    page_icon="🚗",
    layout="wide",
)


# ================================================================
# 1. CHARGEMENT DES ARTEFACTS (mis en cache pour ne charger qu'une fois)
# ================================================================

@st.cache_resource
def load_artifacts():
    encoders = jb.load("encoders.joblib")
    var_cat_vid = jb.load("var_cat_vid.joblib")
    scaler = jb.load("scaler.joblib")
    rf = jb.load("rf_model.joblib")
    return encoders, var_cat_vid, scaler, rf


try:
    encoders, var_cat_vid, scaler, rf = load_artifacts()
except Exception as e:
    st.error(
        "❌ Impossible de charger les fichiers du modèle "
        "(encoders.joblib, var_cat_vid.joblib, scaler.joblib, rf_model.joblib). "
        "Assure-toi qu'ils sont bien dans le même dossier que ce script.\n\n"
        f"Détail : {e}"
    )
    st.stop()

clasnames = var_cat_vid[3]

MARQUES = list(var_cat_vid[0])
TRANSMISSIONS = list(var_cat_vid[1])
QUARTIERS = list(var_cat_vid[2])


# ================================================================
# 2. HISTORIQUE DES PRÉDICTIONS (état de session Streamlit)
# ================================================================

HISTORY_COLUMNS = [
    "Véhicule",
    "Année",
    "Prix",       # texte formaté pour l'affichage
    "PrixNum",    # valeur numérique, utilisée pour les moyennes
    "Quartier",
    "État Prédit",
    "Confiance",  # en %
    "Horodatage",
]


def _format_prix(prix):
    try:
        return f"{float(prix):,.0f} FCFA".replace(",", " ")
    except (TypeError, ValueError):
        return str(prix)


def _add_to_history(marque, annee, prix, quartier, etat, confiance):
    """Ajoute une prédiction à l'historique global du dashboard
    (stocké dans st.session_state, propre à chaque session)."""

    try:
        annee_val = int(float(annee))
    except (TypeError, ValueError):
        annee_val = None

    try:
        prix_val = float(prix)
    except (TypeError, ValueError):
        prix_val = np.nan

    new_row = {
        "Véhicule": str(marque),
        "Année": annee_val,
        "Prix": _format_prix(prix),
        "PrixNum": prix_val,
        "Quartier": str(quartier),
        "État Prédit": str(etat),
        "Confiance": round(float(confiance) * 100, 1),
        "Horodatage": dt.datetime.now(),
    }

    st.session_state.prediction_history = pd.concat(
        [st.session_state.prediction_history, pd.DataFrame([new_row])],
        ignore_index=True,
    )


def _seed_demo_history():
    """Pré-remplit le dashboard avec quelques exemples au démarrage."""

    now = dt.datetime.now()

    demo_rows = [
        ("Toyota Corolla", 2019, 5500000, "Almadies", "Très Bon", 96, 5),
        ("Hyundai Tucson", 2016, 6800000, "Mermoz", "Bon", 89, 4),
        ("Peugeot 308", 2012, 2300000, "Pikine", "Moyen", 72, 3),
        ("Mercedes C-Class", 2021, 14000000, "Sacré-Cœur", "Excellent", 98, 2),
        ("Toyota RAV4", 2020, 9200000, "Ouakam", "Très Bon", 94, 1),
        ("Kia Sportage", 2018, 7100000, "Point E", "Bon", 87, 0),
    ]

    rows = []
    for marque, annee, prix, quartier, etat, conf, jours in demo_rows:
        rows.append({
            "Véhicule": marque,
            "Année": annee,
            "Prix": _format_prix(prix),
            "PrixNum": float(prix),
            "Quartier": quartier,
            "État Prédit": etat,
            "Confiance": float(conf),
            "Horodatage": now - dt.timedelta(days=jours),
        })

    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


# Initialisation de l'historique une seule fois par session
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = _seed_demo_history()


# ================================================================
# 3. PRÉDICTION D'UN VÉHICULE
# ================================================================

def predict_single(marque, transmission, quartier, annee, prix):

    try:
        marque_enc = encoders[0].transform([marque])[0]
        trans_enc = encoders[1].transform([transmission])[0]
        quartier_enc = encoders[2].transform([quartier])[0]
    except ValueError as e:
        raise ValueError(f"Valeur catégorielle inconnue du modèle : {e}")

    # IMPORTANT : l'ordre doit correspondre exactement à celui utilisé
    # lors de l'entraînement du scaler/modèle :
    # [Marque, Année, Transmission, Prix, Quartier]
    x_new = np.array([
        marque_enc,
        float(annee),
        trans_enc,
        float(prix),
        quartier_enc,
    ]).reshape(1, -1)

    x_scaled = scaler.transform(x_new)
    y_pred = rf.predict(x_scaled)[0]
    probas = rf.predict_proba(x_scaled)[0]
    confidence = float(np.max(probas))

    return str(clasnames[y_pred]), confidence


# ================================================================
# 4. RECHERCHE DES COLONNES CSV
# ================================================================

def _find_column(df, candidates):
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# ================================================================
# 5. PRÉDICTION CSV (batch)
# ================================================================

def predict_csv(df):
    """Prend un DataFrame déjà lu, prédit chaque ligne, met à jour
    l'historique de session, renvoie (df_resultat, message)."""

    col_marque = _find_column(df, ["marque", "Marque"])
    col_trans = _find_column(df, ["transmission", "Transmission"])
    col_quartier = _find_column(df, ["quartier", "Quartier"])
    col_annee = _find_column(df, ["annee", "année", "Annee", "Année"])
    col_prix = _find_column(df, ["prix", "Prix"])

    cols = list(df.columns)
    col_marque = col_marque or (cols[0] if len(cols) > 0 else None)
    col_trans = col_trans or (cols[1] if len(cols) > 1 else None)
    col_quartier = col_quartier or (cols[2] if len(cols) > 2 else None)
    col_annee = col_annee or (cols[3] if len(cols) > 3 else None)
    col_prix = col_prix or (cols[4] if len(cols) > 4 else None)

    if None in (col_marque, col_trans, col_quartier, col_annee, col_prix):
        return None, (
            "❌ Le CSV doit contenir 5 colonnes : "
            "Marque, Transmission, Quartier, Annee, Prix."
        )

    predictions = []
    confidences = []

    for _, row in df.iterrows():
        try:
            pred, conf = predict_single(
                row[col_marque], row[col_trans], row[col_quartier],
                row[col_annee], row[col_prix],
            )
            _add_to_history(
                row[col_marque], row[col_annee], row[col_prix],
                row[col_quartier], pred, conf,
            )
        except Exception:
            pred, conf = "Inconnu", 0.0

        predictions.append(pred)
        confidences.append(float(conf * 100))

    df = df.copy()
    df["Etat_Predit"] = predictions
    df["Score_Confiance"] = [f"{x:.1f}%" for x in confidences]

    nb_total = len(df)
    nb_fiable = sum(x >= 80 for x in confidences)
    score_moyen = np.mean(confidences) if confidences else 0

    message = (
        f"✅ {nb_total} lignes traitées | "
        f"Score moyen : {score_moyen:.1f}% | "
        f"Prédictions fiables : {nb_fiable}"
    )

    return df, message


# ================================================================
# 6. AFFICHAGE DES SCORES
# ================================================================

def score_badge(score):
    if score >= 90:
        return f"🟢 {score}%"
    elif score >= 80:
        return f"🔵 {score}%"
    elif score >= 70:
        return f"🟠 {score}%"
    else:
        return f"🔴 {score}%"


# ================================================================
# 7. CONSTRUCTION DYNAMIQUE DU TABLEAU DE BORD
# ================================================================

def _build_dashboard_table(history_df, max_rows=50):
    if len(history_df) == 0:
        return pd.DataFrame(
            columns=["Véhicule", "Année", "Prix", "Quartier", "État Prédit", "Confiance"]
        )

    display_df = history_df.sort_values("Horodatage", ascending=False).head(max_rows).copy()
    display_df["Confiance"] = display_df["Confiance"].apply(score_badge)

    return display_df[["Véhicule", "Année", "Prix", "Quartier", "État Prédit", "Confiance"]]


def _build_kpi_html(history_df):
    total = len(history_df)

    if total == 0:
        avg_conf = 0.0
        avg_prix_m = 0.0
        today_count = 0
    else:
        avg_conf = history_df["Confiance"].mean()

        prix_valides = history_df["PrixNum"].dropna()
        avg_prix_m = (
            prix_valides.mean() / 1_000_000
            if len(prix_valides)
            else 0.0
        )

        # ============================================================
        # CORRECTION : conversion explicite de Horodatage en datetime
        # ============================================================
        historique_dates = pd.to_datetime(
            history_df["Horodatage"],
            errors="coerce"
        )

        today = dt.datetime.now().date()

        today_count = int(
            (historique_dates.dt.date == today).sum()
        )

    conf_status = (
        "✓ Performance validée"
        if avg_conf >= 80
        else "⚠️ À surveiller"
    )

    return f"""
<div class="kpi-container">
    <div class="kpi-card kpi-blue">
        <div class="kpi-icon">📊</div>
        <div class="kpi-title">PRÉDICTIONS RÉALISÉES</div>
        <div class="kpi-value">{total}</div>
        <div class="kpi-positive">Total cumulé</div>
    </div>

    <div class="kpi-card kpi-green">
        <div class="kpi-icon">🎯</div>
        <div class="kpi-title">CONFIANCE MOYENNE</div>
        <div class="kpi-value">{avg_conf:.1f} %</div>
        <div class="kpi-positive">{conf_status}</div>
    </div>

    <div class="kpi-card kpi-orange">
        <div class="kpi-icon">💰</div>
        <div class="kpi-title">PRIX MOYEN ÉVALUÉ</div>
        <div class="kpi-value">{avg_prix_m:.1f} M</div>
        <div class="kpi-positive">FCFA</div>
    </div>

    <div class="kpi-card kpi-purple">
        <div class="kpi-icon">⚡</div>
        <div class="kpi-title">PRÉDICTIONS AUJOURD'HUI</div>
        <div class="kpi-value">{today_count}</div>
        <div class="kpi-positive">Depuis minuit</div>
    </div>
</div>
"""


# ================================================================
# 8. CSS PROFESSIONNEL
# ================================================================

CUSTOM_CSS = """
<style>
.stApp { background: #f4f7fb; }

.dashboard-header {
    background: #ffffff;
    border-radius: 18px;
    padding: 30px 35px;
    margin-bottom: 25px;
    border: 1px solid #dce3eb;
    border-left: 7px solid #0b3d91;
    box-shadow: 0 6px 20px rgba(16, 42, 67, 0.08);
}
.dashboard-title {
    color: #0b2d5c;
    font-size: 34px;
    font-weight: 800;
    line-height: 1.2;
    letter-spacing: -0.5px;
}
.dashboard-subtitle {
    color: #536579;
    font-size: 16px;
    margin-top: 10px;
    font-weight: 500;
}
.kpi-container {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 18px;
    margin-bottom: 28px;
}
.kpi-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 23px;
    min-height: 145px;
    border: 1px solid #dce3eb;
    box-shadow: 0 5px 18px rgba(16, 42, 67, 0.07);
}
.kpi-blue { border-top: 5px solid #1769aa; }
.kpi-green { border-top: 5px solid #159447; }
.kpi-orange { border-top: 5px solid #e28a00; }
.kpi-purple { border-top: 5px solid #7048a8; }
.kpi-icon { font-size: 28px; margin-bottom: 8px; }
.kpi-title { color: #586779; font-size: 13px; font-weight: 700; letter-spacing: 0.4px; }
.kpi-value { color: #102a43; font-size: 30px; font-weight: 800; margin-top: 5px; }
.kpi-positive { color: #159447; font-size: 13px; font-weight: 700; margin-top: 7px; }
.section-title {
    background: #ffffff;
    color: #102a43;
    font-size: 21px;
    font-weight: 800;
    padding: 17px 20px;
    border-radius: 13px;
    border-left: 5px solid #1769aa;
    margin-top: 25px;
    margin-bottom: 17px;
    box-shadow: 0 4px 14px rgba(16,42,67,0.05);
}
.info-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 22px;
    border: 1px solid #dce3eb;
    box-shadow: 0 4px 15px rgba(16,42,67,0.05);
    min-height: 180px;
}
.info-card h3 { color: #102a43; font-size: 18px; font-weight: 800; margin-top: 0; }
.info-card p { color: #536579; font-size: 14px; line-height: 1.7; }
.legend-box {
    background: #ffffff;
    padding: 18px 20px;
    border-radius: 13px;
    border: 1px solid #dce3eb;
    box-shadow: 0 4px 14px rgba(16,42,67,0.05);
    margin-top: 15px;
}
.legend-title { color: #102a43; font-weight: 800; font-size: 15px; margin-bottom: 12px; }
.legend-item { display: inline-block; color: #536579; font-size: 14px; margin-right: 25px; margin-bottom: 8px; }

@media (max-width: 1000px) { .kpi-container { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 600px) {
    .kpi-container { grid-template-columns: 1fr; }
    .dashboard-title { font-size: 26px; }
    .dashboard-header { padding: 23px; }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ================================================================
# 9. HEADER
# ================================================================

HEADER_HTML = """
<div class="dashboard-header">
    <div class="dashboard-title">🚗 CoinAfrique AI</div>
    <div class="dashboard-subtitle">
        Tableau de bord intelligent pour la prédiction
        automatique de l'état des véhicules
    </div>
</div>
"""

st.markdown(HEADER_HTML, unsafe_allow_html=True)


# ================================================================
# 10. APPLICATION (onglets)
# ================================================================

tab_dashboard, tab_single, tab_csv = st.tabs(
    ["📊 Tableau de bord", "🔍 Prédiction simple", "📁 Importation CSV"]
)


# ------------------------------------------------------------------
# ONGLET 1 : DASHBOARD
# ------------------------------------------------------------------

with tab_dashboard:

    st.markdown(_build_kpi_html(st.session_state.prediction_history), unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Actualiser le tableau de bord", use_container_width=True):
            st.rerun()
    with col_b:
        if st.button("🗑️ Réinitialiser l'historique", use_container_width=True):
            st.session_state.prediction_history = pd.DataFrame(columns=HISTORY_COLUMNS)
            st.success("🗑️ Historique réinitialisé. Le tableau de bord repart à zéro.")
            st.rerun()

    st.markdown('<div class="section-title">🎯 Performance et qualité des prédictions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>🤖 Modèle de Machine Learning</h3>
                <p>Le système utilise un modèle <b>Random Forest</b> pour prédire
                automatiquement l'état probable d'un véhicule.</p>
                <p><b>Variables utilisées :</b><br>
                Marque • Transmission • Quartier • Année • Prix</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>🛡️ Niveau de fiabilité</h3>
                <p>Le score de confiance représente le niveau de certitude
                du modèle pour chaque prédiction.</p>
                <p><b>Seuil recommandé :</b> 80 %</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="legend-box">
            <div class="legend-title">📌 Interprétation du score de confiance</div>
            <span class="legend-item">🟢 <b>≥ 90 %</b> — Très fiable</span>
            <span class="legend-item">🔵 <b>80–89 %</b> — Fiable</span>
            <span class="legend-item">🟠 <b>70–79 %</b> — À vérifier</span>
            <span class="legend-item">🔴 <b>&lt; 70 %</b> — Faible confiance</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">🚗 Historique des dernières prédictions</div>', unsafe_allow_html=True)

    st.dataframe(
        _build_dashboard_table(st.session_state.prediction_history),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<div class="section-title">📈 Synthèse décisionnelle</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-card">
            <h3>💡 Interprétation du tableau de bord</h3>
            <p>L'application permet d'automatiser l'évaluation de l'état des
            véhicules à partir de leurs caractéristiques. Les chiffres ci-dessus
            reflètent les prédictions réellement effectuées via les onglets
            "Prédiction simple" et "Importation CSV".</p>
            <p>Un score de confiance élevé indique que le modèle est fortement
            certain de sa prédiction. Les résultats présentant un score faible
            doivent être vérifiés avant toute décision.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# ONGLET 2 : PRÉDICTION SIMPLE
# ------------------------------------------------------------------

with tab_single:

    st.markdown("## 🔍 Évaluation individuelle d'un véhicule")
    st.write("Renseignez les caractéristiques du véhicule pour obtenir une prédiction automatique.")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        marque = st.selectbox("🚗 Marque", MARQUES, key="marque_in")

        c1, c2 = st.columns(2)
        with c1:
            transmission = st.selectbox("⚙️ Transmission", TRANSMISSIONS, key="trans_in")
        with c2:
            quartier = st.selectbox("📍 Quartier", QUARTIERS, key="quartier_in")

        c3, c4 = st.columns(2)
        with c3:
            annee = st.number_input("📅 Année", value=2018, step=1, format="%d", key="annee_in")
        with c4:
            prix = st.number_input("💰 Prix (FCFA)", value=4500000, step=100000, key="prix_in")

        launch = st.button("🚀 Lancer la prédiction", type="primary", use_container_width=True)

    with col_right:
        result_placeholder = st.empty()
        confidence_placeholder = st.empty()

        if launch:
            if not marque or not transmission or not quartier:
                st.warning("⚠️ Veuillez sélectionner toutes les options.")
            else:
                try:
                    prediction, confidence = predict_single(marque, transmission, quartier, annee, prix)
                    _add_to_history(marque, annee, prix, quartier, prediction, confidence)
                    result_placeholder.success(f"🎯 État évalué : **{prediction}**")
                    confidence_placeholder.info(f"📊 Score de confiance : **{confidence * 100:.1f}%**")
                except ValueError as e:
                    st.error(f"❌ Erreur : {e}")


# ------------------------------------------------------------------
# ONGLET 3 : IMPORTATION CSV
# ------------------------------------------------------------------

with tab_csv:

    st.markdown("## 📁 Analyse prédictive en masse")
    st.write("Importez un fichier CSV contenant les colonnes :")
    st.markdown("**Marque • Transmission • Quartier • Annee • Prix**")

    uploaded_file = st.file_uploader("📤 Importer votre fichier CSV", type=["csv"])

    if st.button("🚀 Traiter le fichier", type="primary"):
        if uploaded_file is None:
            st.warning("⚠️ Veuillez importer un fichier CSV.")
        else:
            try:
                df_in = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"❌ Impossible de lire le CSV : {e}")
                df_in = None

            if df_in is not None:
                df_out, message = predict_csv(df_in)

                if df_out is None:
                    st.error(message)
                else:
                    st.success(message)
                    st.dataframe(df_out, use_container_width=True, hide_index=True)

                    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Télécharger les résultats",
                        data=csv_bytes,
                        file_name="predictions_resultats.csv",
                        mime="text/csv",
                    )
