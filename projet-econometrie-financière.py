import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import io

# Configuration de la page
st.set_page_config(
    page_title="Backtesting VaR & CVaR",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé amélioré
st.markdown("""
<style>
    /* Styles globaux */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header principal */
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Section headers */
    .section-header {
        font-size: 2rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
        font-weight: 600;
    }
    
    /* Sous-section headers */
    .subsection-header {
        font-size: 1.6rem;
        color: #34495e;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        font-weight: 500;
    }
    
    /* Cartes d'information */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Boutons stylés */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    /* Onglets stylés */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #e9ecef;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #dee2e6;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
    }
    
    /* Sidebar améliorée */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Formules mathématiques */
    .formula-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
        font-family: "Courier New", monospace;
        font-size: 1.1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Résultats de tests */
    .test-result {
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .accept {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #155724;
        color: #155724;
    }
    
    .reject {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #721c24;
        color: #721c24;
    }
    
    /* Cartes de métriques */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #3498db;
    }
    
    /* Animation pour les sections */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .section-animation {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# FONCTIONS UTILITAIRES
# ============================================

# ============================================
# SECTION THÉORIQUE - AJOUT DES COURS
# ============================================

def display_theoretical_content():
    """Affiche le contenu théorique sur le backtesting"""
    
    st.markdown("""
    # 📚 COURS THÉORIQUE : PRINCIPE DU BACKTESTING

    ## Introduction
    Le backtesting est une méthode de validation des modèles de risque qui consiste à comparer les prévisions de risque établies à l'avance avec les pertes effectivement observées sur une période donnée. 
    
    **Objectif principal** : Vérifier si un modèle de risque est capable de décrire correctement le comportement réel des pertes financières.

    ## 1. Principe Fondamental
    
    ### Définition
    Le backtesting répond à une question essentielle : **le modèle de risque est-il fiable?**
    
    ### Approche en deux périodes
    1. **Période d'estimation** : Calibrage du modèle avec données historiques
    2. **Période de test** : Comparaison prévisions vs réalisations
    
    Cette séparation est **indispensable** pour éviter le surapprentissage (évaluer sur les mêmes données que l'estimation).

    ### Concept de Violation
    - **Violation/Exception** : Quand la perte dépasse la VaR estimée
    - Un bon modèle doit produire des violations **rares** (selon niveau confiance) et **aléatoires**
    
    ## 2. Importance de la Structure Temporelle
    
    Le backtesting ne se limite pas au comptage des violations :
    
    - **Violations groupées** → Modèle réagit mal aux changements de volatilité
    - **Analyse dynamique** : Évaluation de la capacité à s'adapter aux conditions de marché
    
    ## 3. Backtesting CVaR : Complexité Accrue
    
    ### Particularités
    - Se concentre sur **situations extrêmes** (dépassements de VaR)
    - Vérifie l'**ampleur moyenne** des pertes vs CVaR estimée
    - Évalue le **risque de queue** (pertes les plus sévères)
    
    ## 4. Interprétation et Limites
    
    ### Outil de diagnostic
    Le backtesting est un **outil de diagnostic**, pas un jugement définitif.
    
    ### Nécessité de compléments
    - **Stress tests** supplémentaires
    - **Analyses de scénarios**
    - **Recalibration régulière**
    
    ## 5. Méthodologie Statistique
    
    ### Séquence de Violation
    Pour un portefeuille avec rendements $r_{p,t}$ et VaR estimée $VaR_t$, on définit :
    
    $$
    I_{t+1} = 
    \\begin{cases}
    1 & \\text{si } r_{p,t+1} < -VaR_{t+1} \\\\
    0 & \\text{si } r_{p,t+1} \\geq -VaR_{t+1}
    \\end{cases}
    $$
    
    Cette séquence $\{I_{t+1}\}_{t=1}^T$ constitue la base des tests statistiques.
    
    ### Propriétés Requises pour un Modèle Valide
    
    1. **Couverture Non Conditionnelle** :
       $$
       P(I_{t+1} = 1) = E(I_{t+1}) = p
       $$
       - Fréquence dépassements = probabilité théorique
       - Si fréquence > $p$ → sous-estimation risque
       - Si fréquence < $p$ → surestimation risque
    
    2. **Indépendance** :
       $$
       P(I_{t+1} = 1 | F_t) = P(I_{t+1} = 1)
       $$
       - Pas d'information dans l'historique des violations
       - Violations doivent être i.i.d. Bernoulli($p$)
    
    ## 6. Tests Statistiques de Backtesting
    
    ### 6.1 Test de Kupiec (1995) - Couverture Inconditionnelle
    
    **Hypothèses** :
    - $H_0$ : Proportion violations = $1 - \\alpha$
    - $H_1$ : Proportion violations ≠ $1 - \\alpha$
    
    **Statistique de test** :
    $$
    LR_{uc} = -2 \\ln\\left[\\frac{(1-\\alpha)^x \\alpha^{T-x}}{(1-\\hat{p})^x \\hat{p}^{T-x}}\\right]
    $$
    où :
    - $T$ = nombre total observations
    - $x$ = nombre violations observées
    - $\\hat{p} = x/T$ = fréquence empirique
    
    **Distribution** : $LR_{uc} \\sim \\chi^2(1)$ sous $H_0$
    
    **Décision** : Rejet $H_0$ si $LR_{uc} > \\chi^2_{1,1-\\gamma}$
    
    ### 6.2 Test d'Indépendance (Christoffersen 1998)
    
    **Objectif** : Vérifier l'absence de clustering des violations
    
    **Hypothèses** :
    - $H_0$ : Violations indépendantes
    - $H_1$ : Violations dépendantes
    
    **Statistique** : $LR_{ind} \\sim \\chi^2(1)$ sous $H_0$
    
    ### 6.3 Test de Couverture Conditionnelle (Christoffersen)
    
    **Combinaison** des deux tests précédents :
    $$
    LR_{cc} = LR_{uc} + LR_{ind} \\sim \\chi^2(2)
    $$
    
    **Test global** de validité du modèle VaR
    
    ## 7. Backtesting de la CVaR
    
    ### Définition CVaR
    $$
    CVaR_\\alpha = E[L_t | L_t > VaR_\\alpha]
    $$
    
    ### Approche par Fonction de Score
    Fonction de score couramment utilisée :
    
    $$
    S_t = (\\mathbb{1}_{\\{L_t > VaR_t\\}} - (1-\\alpha))VaR_t + \\frac{1}{1-\\alpha}\\mathbb{1}_{\\{L_t > VaR_t\\}}(L_t - CVaR_t)
    $$
    
    **Hypothèses** :
    - $H_0$ : VaR et CVaR correctement estimées
    - $H_1$ : CVaR mal estimée
    
    ## 8. Guide Pratique d'Interprétation
    
    ### Signaux d'Alerte
    1. **Nombre de violations** :
       - Trop élevé → Sous-estimation risque
       - Trop faible → Surestimation risque → Coût opportunité
    
    2. **Distribution temporelle** :
       - Clustering → Modèle non adaptatif
       - Régularité → Anomalie statistique
    
    3. **Ampleur des violations** (CVaR) :
       - Pertes moyennes > CVaR → Sous-estimation risque extrême
       - Pertes moyennes < CVaR → Prudence excessive
    
    ### Bonnes Pratiques
    - **Périodicité** : Backtesting régulier (mensuel/trimestriel)
    - **Robustesse** : Tester plusieurs méthodes et fenêtres
    - **Conservatisme** : En cas de doute, privilégier les modèles prudents
    - **Documentation** : Traçabilité complète des tests
    
    ## 9. Conclusion
    
    Le backtesting est un **processus essentiel** mais **non suffisant** :
    
    ✅ **Points forts** :
    - Validation quantitative objective
    - Détection précoce des dérives modèles
    - Conformité réglementaire (Bâle)
    
    ⚠️ **Limitations** :
    - Dépendance aux données historiques
    - Pas de garantie pour le futur
    - Nécessite compléments (stress tests)
    
    **Recommandation finale** : Utiliser le backtesting comme **composante d'un système intégré** de gestion des risques, combiné avec l'expertise métier et une surveillance continue des marchés.
    """)

# ============================================
# SECTION RAPPORT - STRUCTURE DÉTAILLÉE
# ============================================

def display_report_structure():
    """Affiche la structure détaillée du rapport"""
    
    st.markdown("""
    # 📝 STRUCTURE DU RAPPORT DE BACKTESTING
    
    ## Rapport Professionnel - Analyse de Risque Financier
    
    ### **Page de Garde**
    - Titre : Rapport de Backtesting VaR/CVaR
    - Organisation/Équipe
    - Date de production
    - Période analysée
    - Classification : Interne/Confidentiel
    
    ### **Table des Matières**
    
    ### **Résumé Exécutif** (1 page maximum)
    
    #### 1. Objectifs de l'Analyse
    - Contexte et justification du backtesting
    - Périmètre de l'étude
    - Cadre réglementaire applicable
    
    #### 2. Principaux Résultats
    - Synthèse des performances du modèle
    - Décisions clés issues du backtesting
    - Recommandations principales
    
    #### 3. Conclusions Opérationnelles
    - Validité du modèle actuel
    - Actions correctives requises
    - Calendrier de mise en œuvre
    
    ---
    
    ### **Chapitre 1 : Méthodologie et Cadre d'Analyse**
    
    #### 1.1 Définitions et Concepts Clés
    - Value at Risk (VaR) : définitions et interprétations
    - Conditional VaR (CVaR) : compléments et avantages
    - Principes généraux du backtesting
    
    #### 1.2 Modèles de Risque Évalués
    - Description détaillée des modèles testés
    - Paramètres d'estimation (fenêtres, méthodes)
    - Hypothèses sous-jacentes
    
    #### 1.3 Tests Statistiques Implémentés
    - Test de Kupiec : couverture inconditionnelle
    - Test d'indépendance : détection du clustering
    - Test de Christoffersen : couverture conditionnelle
    - Tests spécifiques CVaR
    
    #### 1.4 Données Utilisées
    - Sources et qualité des données
    - Période d'observation
    - Traitements appliqués (nettoyage, ajustements)
    
    ---
    
    ### **Chapitre 2 : Résultats du Backtesting VaR**
    
    #### 2.1 Analyse Descriptive des Violations
    - Nombre total de violations observées
    - Fréquence vs fréquence attendue
    - Statistiques descriptives par sous-période
    
    #### 2.2 Tests de Couverture Inconditionnelle
    - Résultats détaillés test Kupiec
    - Interprétation statistique
    - Analyse par niveau de confiance
    
    #### 2.3 Tests d'Indépendance
    - Détection de clustering temporel
    - Analyse autocorrélation des violations
    - Tests de persistance
    
    #### 2.4 Tests de Couverture Conditionnelle
    - Résultats test Christoffersen
    - Validité globale du modèle
    - Forces et faiblesses identifiées
    
    #### 2.5 Analyse par Sous-Périodes
    - Performance en période calme vs volatile
    - Stabilité temporelle des résultats
    - Points de rupture identifiés
    
    ---
    
    ### **Chapitre 3 : Backtesting de la CVaR**
    
    #### 3.1 Méthodologie Spécifique CVaR
    - Approches de backtesting retenues
    - Mesures de performance adaptées
    - Difficultés méthodologiques
    
    #### 3.2 Analyse des Pertes Extrêmes
    - Distribution des pertes au-delà de la VaR
    - Comparaison CVaR estimée vs réalisée
    - Évaluation du risque de queue
    
    #### 3.3 Tests Statistiques CVaR
    - Résultats des tests spécifiques
    - Validité des estimations CVaR
    - Complémentarité avec analyse VaR
    
    ---
    
    ### **Chapitre 4 : Analyse Comparative et Robustesse**
    
    #### 4.1 Comparaison des Modèles
    - Performance relative des différentes approches
    - Trade-off précision vs complexité
    - Consistances/inconsistances observées
    
    #### 4.2 Tests de Robustesse
    - Sensibilité aux paramètres d'estimation
    - Stabilité sur différentes fenêtres
    - Résistance aux chocs de marché
    
    #### 4.3 Benchmarking
    - Comparaison avec modèles de référence
    - Performance vs standards du secteur
    - Analyse des écarts
    
    ---
    
    ### **Chapitre 5 : Implications et Recommandations**
    
    #### 5.1 Évaluation Globale du Modèle
    - Score de performance synthétique
    - Points forts à conserver
    - Faiblesses à corriger
    
    #### 5.2 Recommandations Techniques
    - Ajustements paramétriques recommandés
    - Améliorations méthodologiques
    - Modifications algorithmiques
    
    #### 5.3 Implications Opérationnelles
    - Impact sur le capital réglementaire
    - Modifications processus de gestion des risques
    - Formation nécessaire pour les équipes
    
    #### 5.4 Plan d'Action
    - Actions prioritaires (court terme)
    - Améliorations à moyen terme
    - Feuille de route stratégique
    
    ---
    
    ### **Chapitre 6 : Annexes Techniques**
    
    #### Annexe A : Détails des Données
    - Description complète des séries utilisées
    - Métadonnées et dictionnaire de données
    - Journal des traitements appliqués
    
    #### Annexe B : Détails des Calculs
    - Formules mathématiques complètes
    - Algorithmes implémentés
    - Codes et scripts utilisés
    
    #### Annexe C : Résultats Détaillés
    - Tableaux complets de résultats
    - Sorties brutes des tests statistiques
    - Graphiques supplémentaires
    
    #### Annexe D : Références Bibliographiques
    - Articles académiques cités
    - Documentation réglementaire
    - Ouvrages de référence
    
    ---
    
    ### **Glossaire**
    - Définitions des termes techniques
    - Acronymes et abréviations
    - Notations mathématiques
    
    ---
    
    ### **Historique des Versions**
    - Version 1.0 : Date, Auteur, Modifications
    - Révisions ultérieures
    
    ## 🔧 Guide de Rédaction
    
    ### Style Rédactionnel
    1. **Clarté** : Langage accessible même pour non-spécialistes
    2. **Précision** : Chiffres exacts, sources citées
    3. **Objectivité** : Présentation neutre des résultats
    4. **Concision** : Aller à l'essentiel
    
    ### Présentation des Résultats
    - **Tableaux** : Structurés, titrés, avec légendes
    - **Graphiques** : Couleurs standards, échelles adaptées
    - **Commentaires** : Interprétation systématique des résultats
    
    ### Validation du Rapport
    - Vérification croisée des calculs
    - Relecture par pairs
    - Validation hiérarchique
    - Archivage version finale
    
    ## 📊 Indicateurs de Qualité du Rapport
    
    ### Obligatoires
    ✓ Couverture exhaustive du périmètre  
    ✓ Cohérence interne des résultats  
    ✓ Traçabilité complète des calculs  
    ✓ Conformité réglementaire  
    
    ### Recommandés
    ✓ Comparaisons benchmarks sectoriels  
    ✓ Analyses sensibilité approfondies  
    ✓ Recommandations actionnables  
    ✓ Plan de mise en œuvre détaillé  
    
    ## ⚠️ Avertissements Standards
    
    ### Limitations Méthodologiques
    - Résultats basés sur données historiques
    - Performances passées non garanties pour le futur
    - Hypothèses modélisation susceptibles d'évoluer
    
    ### Utilisation Responsable
    - Rapport à usage interne uniquement
    - Prise de décision complémentaire nécessaire
    - Surveillance continue requise
    
    ---
    
    *Document produit par le système automatisé de backtesting - [Nom de l'Organisation]*
    *Date de génération : {date_du_jour}*
    """)


def detect_date_column(df):
    """Détecte automatiquement la colonne de dates"""
    date_columns = ["Date"]
    for col in df.columns:
        # Essayer de convertir en datetime
        try:
            sample = df[col].dropna().iloc[0]
            if isinstance(sample, str) and len(sample) > 5:
                # Vérifier si ça ressemble à une date
                if any(sep in sample for sep in ['-', '/', '.']):
                    date_columns.append(col)
        except:
            continue
    
    return date_columns

def detect_numeric_columns(df):
    """Détecte les colonnes numériques (prix des actifs)"""
    numeric_cols = []
    for col in df.columns:
        try:
            # Essayer de convertir en numérique
            pd.to_numeric(df[col], errors='raise')
            numeric_cols.append(col)
        except:
            continue
    return numeric_cols

def calculate_returns(prices):
    """Calcule les rendements logarithmiques à partir des prix"""
    if isinstance(prices, pd.DataFrame):
        returns = pd.DataFrame()
        for col in prices.columns:
            returns[col] = np.log(prices[col] / prices[col].shift(1))
        return returns.dropna()
    else:
        return np.log(prices / prices.shift(1)).dropna()

def create_sample_data():
    """Crée des données d'exemple pour le template"""
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='B')
    
    # Générer des séries de prix réalistes avec drift et volatilité
    np.random.seed(42)
    n_dates = len(dates)
    
    # Actif 1: Action avec tendance haussière
    drift1 = 0.0002
    volatility1 = 0.015
    prices1 = 100 * np.exp(np.cumsum(np.random.normal(drift1, volatility1, n_dates)))
    
    # Actif 2: Action volatile
    drift2 = 0.0001
    volatility2 = 0.025
    prices2 = 50 * np.exp(np.cumsum(np.random.normal(drift2, volatility2, n_dates)))
    
    # Actif 3: Action stable
    drift3 = 0.0003
    volatility3 = 0.01
    prices3 = 75 * np.exp(np.cumsum(np.random.normal(drift3, volatility3, n_dates)))
    
    df = pd.DataFrame({
        'Date': dates,
        'Action_1': np.round(prices1, 2),
        'Action_2': np.round(prices2, 2),
        'Action_3': np.round(prices3, 2),
        'Indice_Market': np.round(1000 + 100 * np.sin(np.linspace(0, 10, n_dates)) + 
                                  np.random.normal(0, 10, n_dates), 2)
    })
    
    return df

# ============================================
# SIDEBAR - Navigation améliorée
# ============================================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='color: white; font-size: 1.8rem; margin-bottom: 2rem;'>📊 BACKTESTING</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Menu de navigation avec boutons stylés
    st.markdown("### 🎯 NAVIGATION")
    
    # Sections principales
    sections = {
        "📤 CHARGER DONNÉES": "upload",
        "📊 EXPLORER DONNÉES": "explore",
        "⚖️ PORTERFEUILLE": "portfolio",
        "📈 CALCUL RENDEMENTS": "returns",
        "🎯 BACKTESTING VaR": "var",
        "📊 BACKTESTING CVaR": "cvar",
        "📈 VISUALISATIONS": "visualize",
        "📝 RAPPORT": "report"
    }
    
    # Créer les boutons de navigation
    for section_name, section_id in sections.items():
        if st.button(section_name, key=f"nav_{section_id}", use_container_width=True):
            st.session_state['current_section'] = section_id
    
    # Initialiser la section courante
    if 'current_section' not in st.session_state:
        st.session_state['current_section'] = 'upload'
    
    st.markdown("---")
    
    # Paramètres globaux
    st.markdown("### ⚙️ PARAMÈTRES")
    
    # Valeur du portefeuille avec style
    valeur_portefeuille = st.number_input(
        "💼 Valeur du portefeuille (€)", 
        min_value=1000.0, 
        max_value=1000000000.0, 
        value=1000000.0,
        step=10000.0,
        help="Capital total à investir"
    )
    
    # Niveau de confiance avec slider amélioré
    confiance = st.select_slider(
        "🎯 Niveau de confiance",
        options=[90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
        value=95,
        help="Niveau de confiance pour le calcul de la VaR/CVaR"
    )
    
    st.session_state['portfolio_value'] = valeur_portefeuille
    st.session_state['confidence_level'] = confiance
    
    st.markdown("---")
    
    # Information
    with st.expander("ℹ️ À PROPOS"):
        st.info("""
        **Application de Backtesting VaR/CVaR**
        
        Cette application permet d'analyser et de valider
        vos modèles de risque financier.
        
        **Fonctionnalités :**
        - Import flexible de données Excel
        - Analyse descriptive avancée
        - Backtesting VaR (Kupiec, Christoffersen)
        - Backtesting CVaR
        - Visualisations interactives
        
        **Méthodologies :**
        - Kupiec (1995)
        - Christoffersen (1998)
        """)

# ============================================
# SECTION 1: CHARGEMENT DES DONNÉES
# ============================================
# ============================================
# MODIFICATION DE LA SECTION UPLOAD POUR INCLURE LES COURS
# ============================================

if st.session_state['current_section'] == 'upload':
    st.markdown('<h1 class="main-header">📤 CHARGEMENT DES DONNÉES</h1>', unsafe_allow_html=True)
    
    # Ajout d'un onglet pour les cours théoriques
    tab1, tab2, tab3 = st.tabs(["📤 Charger Données", "📚 Cours Théorique", "📝 Structure Rapport"])
    
    with tab1:
        # Le code existant de la section upload reste ici
        # Introduction
        with st.container():
            st.markdown("""
            <div class='info-card'>
            <h3>📋 Comment utiliser cette application ?</h3>
            <p>1. <strong>Téléchargez le template</strong> pour voir le format attendu</p>
            <p>2. <strong>Importez vos données</strong> Excel contenant les prix des actifs</p>
            <p>3. <strong>Configurez votre portefeuille</strong> en définissant les poids</p>
            <p>4. <strong>Exécutez les analyses</strong> et visualisez les résultats</p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📝 Format des données")
            st.markdown("""
            Votre fichier Excel doit contenir :
            
            **Colonnes obligatoires :**
            1. **Une colonne de dates** (format: JJ/MM/AAAA, AAAA-MM-JJ, etc.)
            2. **Une ou plusieurs colonnes de prix** (valeurs numériques)
            
            **Exemple de structure :**
            | Date | Action_A | Action_B | Indice_X |
            |------|----------|----------|----------|
            | 2023-01-01 | 100.50 | 45.30 | 1250.00 |
            | 2023-01-02 | 102.30 | 44.80 | 1245.50 |
            | ... | ... | ... | ... |
            
            **Format accepté :** .xlsx, .xls
            """)
        
        with col2:
            st.markdown("### 📥 Télécharger un template")
            
            # Créer des données d'exemple
            sample_df = create_sample_data()
            
            # Convertir en Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                sample_df.to_excel(writer, sheet_name='Donnees', index=False)
            
            template_data = output.getvalue()
            
            st.download_button(
                label="📥 TÉLÉCHARGER TEMPLATE",
                data=template_data,
                file_name="template_donnees_financieres.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Zone de téléversement
        st.markdown("### 🚀 IMPORTEZ VOS DONNÉES")
        
        uploaded_file = st.file_uploader(
            "Glissez-déposez votre fichier Excel ici",
            type=['xlsx', 'xls'],
            help="Sélectionnez un fichier Excel contenant vos données financières"
        )
        
        if uploaded_file is not None:
            try:
                # Lire le fichier Excel
                xls = pd.ExcelFile(uploaded_file)
                
                # Afficher les feuilles disponibles
                feuilles = xls.sheet_names
                st.success(f"✅ Fichier chargé avec succès !")
                st.info(f"**Feuilles détectées :** {', '.join(feuilles)}")
                
                # Sélectionner la feuille à utiliser
                selected_sheet = st.selectbox(
                    "Sélectionnez la feuille contenant vos données :",
                    feuilles
                )
                
                # Lire la feuille sélectionnée
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                
                # Afficher un aperçu
                st.markdown("#### 👁️ APERÇU DES DONNÉES")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Détection automatique des colonnes
                st.markdown("#### 🔍 DÉTECTION AUTOMATIQUE")
                
                # Détecter la colonne de dates
                date_cols = detect_date_column(df)
                if date_cols:
                    selected_date_col = st.selectbox(
                        "Sélectionnez la colonne de dates :",
                        date_cols,
                        index=0
                    )
                    
                    # Convertir en datetime
                    df[selected_date_col] = pd.to_datetime(df[selected_date_col], errors='coerce')
                    df = df.set_index(selected_date_col)
                    df = df.sort_index()
                    
                    st.success(f"✅ Dates configurées : {df.index[0].date()} → {df.index[-1].date()}")
                else:
                    st.warning("⚠️ Aucune colonne de dates détectée. Utilisation de l'index.")
                
                # Détecter les colonnes numériques (prix)
                numeric_cols = detect_numeric_columns(df)
                
                if numeric_cols:
                    st.success(f"✅ {len(numeric_cols)} colonnes numériques détectées")
                    
                    # Afficher les colonnes détectées
                    cols_per_row = 4
                    rows = [numeric_cols[i:i+cols_per_row] for i in range(0, len(numeric_cols), cols_per_row)]
                    
                    for row in rows:
                        cols = st.columns(len(row))
                        for idx, col_name in enumerate(row):
                            with cols[idx]:
                                st.metric(
                                    label=col_name,
                                    value=f"{len(df[col_name].dropna()):,} obs",
                                    delta=f"Min: {df[col_name].min():.2f} | Max: {df[col_name].max():.2f}"
                                )
                    
                    # Stocker les données dans la session
                    st.session_state['raw_data'] = df[numeric_cols]
                    st.session_state['data_loaded'] = True
                    st.session_state['available_assets'] = numeric_cols
                    st.session_state['date_col'] = selected_date_col if date_cols else None
                    
                    # Bouton pour passer à l'exploration
                    if st.button("🚀 EXPLORER LES DONNÉES", use_container_width=True):
                        st.session_state['current_section'] = 'explore'
                        st.rerun()
                    
                else:
                    st.error("❌ Aucune colonne numérique détectée. Vérifiez votre fichier.")
                    
            except Exception as e:
                st.error(f"❌ Erreur lors de la lecture du fichier : {str(e)}")
    
    with tab2:
        # Afficher le cours théorique
        display_theoretical_content()
    
    with tab3:
        # Afficher la structure du rapport
        display_report_structure()

# ============================================
# SECTION 2: EXPLORATION DES DONNÉES
# ============================================
elif st.session_state['current_section'] == 'explore':
    st.markdown('<h1 class="main-header">📊 EXPLORATION DES DONNÉES</h1>', unsafe_allow_html=True)
    
    if 'data_loaded' not in st.session_state or not st.session_state['data_loaded']:
        st.warning("⚠️ Veuillez d'abord charger des données.")
        if st.button("⬅️ RETOUR AU CHARGEMENT", use_container_width=True):
            st.session_state['current_section'] = 'upload'
            st.rerun()
        st.stop()
    
    df = st.session_state.get('raw_data')
    available_assets = st.session_state.get('available_assets', [])
    
    # Sélection des actifs à analyser
    st.markdown("### 🎯 SÉLECTION DES ACTIFS")
    
    selected_assets = st.multiselect(
        "Choisissez les actifs à analyser :",
        options=available_assets,
        default=available_assets[:min(3, len(available_assets))],
        help="Sélectionnez au moins un actif pour l'analyse"
    )
    
    if not selected_assets:
        st.warning("⚠️ Veuillez sélectionner au moins un actif.")
        st.stop()
    
    df_selected = df[selected_assets].dropna()
    
    # Statistiques descriptives
    st.markdown("### 📈 STATISTIQUES DESCRIPTIVES")
    
    tabs = st.tabs(["📊 Vue d'ensemble", "📈 Évolution", "📊 Distribution"])
    
    with tabs[0]:
        # Aperçu des données
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Période", f"{df_selected.index[0].date()} au {df_selected.index[-1].date()}")
        
        with col2:
            st.metric("Jours de trading", f"{len(df_selected):,}")
        
        with col3:
            st.metric("Actifs sélectionnés", f"{len(selected_assets)}")
        
        # Statistiques détaillées
        st.dataframe(df_selected.describe().style.format("{:.2f}"), use_container_width=True)
    
    with tabs[1]:
        # Évolution des prix
        fig = go.Figure()
        
        for asset in selected_assets:
            fig.add_trace(go.Scatter(
                x=df_selected.index,
                y=df_selected[asset],
                name=asset,
                mode='lines',
                hovertemplate='Date: %{x}<br>Prix: %{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Évolution des prix',
            xaxis_title='Date',
            yaxis_title='Prix',
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        # Distribution des prix
        fig = make_subplots(
            rows=len(selected_assets),
            cols=1,
            subplot_titles=selected_assets,
            vertical_spacing=0.05
        )
        
        for i, asset in enumerate(selected_assets):
            fig.add_trace(
                go.Histogram(
                    x=df_selected[asset],
                    name=asset,
                    nbinsx=50,
                    marker_color=f'rgb({(i+1)*60}, {(i+2)*40}, {(i+3)*80})'
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            height=300 * len(selected_assets),
            showlegend=False,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Corrélations
    st.markdown("### 🔗 MATRICE DE CORRÉLATION")
    
    if len(selected_assets) > 1:
        corr_matrix = df_selected.corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 12}
        ))
        
        fig_corr.update_layout(
            title='Corrélation entre les actifs',
            height=500
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Boutons de navigation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬅️ RETOUR", use_container_width=True):
            st.session_state['current_section'] = 'upload'
            st.rerun()
    
    with col2:
        if st.button("📊 SAUVEGARDER LES DONNÉES", use_container_width=True):
            # Sauvegarder les données sélectionnées
            st.session_state['selected_assets'] = selected_assets
            st.session_state['price_data'] = df_selected
            st.success("✅ Données sauvegardées !")
    
    with col3:
        if st.button("⚖️ CONFIGURER PORTEFEUILLE ➡️", use_container_width=True):
            st.session_state['selected_assets'] = selected_assets
            st.session_state['price_data'] = df_selected
            st.session_state['current_section'] = 'portfolio'
            st.rerun()

# ============================================
# SECTION 3: CONFIGURATION DU PORTEFEUILLE
# ============================================
elif st.session_state['current_section'] == 'portfolio':
    st.markdown('<h1 class="main-header">⚖️ CONFIGURATION DU PORTEFEUILLE</h1>', unsafe_allow_html=True)
    
    if 'selected_assets' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord sélectionner des actifs.")
        if st.button("⬅️ RETOUR À L'EXPLORATION", use_container_width=True):
            st.session_state['current_section'] = 'explore'
            st.rerun()
        st.stop()
    
    selected_assets = st.session_state.get('selected_assets', [])
    portfolio_value = st.session_state.get('portfolio_value', 1000000)
    
    st.markdown("""
    <div class='info-card'>
    <h3>⚖️ Définissez la composition de votre portefeuille</h3>
    <p>Les poids doivent être exprimés en pourcentage et leur somme doit être égale à 100%.</p>
    <p><strong>Valeur totale du portefeuille :</strong> {:,} €</p>
    </div>
    """.format(int(portfolio_value)), unsafe_allow_html=True)
    
    # Interface de configuration des poids
    weights = {}
    total_weight = 0
    
    # Créer 2 colonnes pour les actifs
    cols = st.columns(2)
    
    for idx, asset in enumerate(selected_assets):
        with cols[idx % 2]:
            st.markdown(f"**{asset}**")
            
            # Slider pour le poids
            weight = st.slider(
                f"Poids de {asset} (%)",
                min_value=0.0,
                max_value=100.0,
                value=100.0/len(selected_assets) if len(selected_assets) > 0 else 100.0,
                step=1.0,
                key=f"weight_{asset}"
            )
            
            weights[asset] = weight / 100.0
            total_weight += weight
            
            # Calcul de la valeur investie
            investment = portfolio_value * (weight / 100.0)
            st.info(f"**Valeur investie :** {investment:,.2f} €")
    
    # Afficher le total
    st.markdown(f"### 📊 TOTAL DES POIDS : {total_weight:.1f}%")
    
    if abs(total_weight - 100.0) > 0.1:
        st.error(f"❌ La somme des poids doit être égale à 100%. Actuellement : {total_weight:.1f}%")
    else:
        st.success("✅ Portefeuille correctement configuré !")
        
        # Visualisation avec pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(weights.keys()),
            values=[w * 100 for w in weights.values()],
            hole=0.3,
            textinfo='label+percent',
            marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        )])
        
        fig.update_layout(
            title="Répartition du portefeuille",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau récapitulatif
        st.markdown("### 📋 RÉCAPITULATIF DES INVESTISSEMENTS")
        
        summary_data = []
        for asset, weight in weights.items():
            investment = portfolio_value * weight
            summary_data.append({
                'Actif': asset,
                'Poids (%)': f"{weight*100:.1f}",
                'Valeur investie (€)': f"{investment:,.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Sauvegarder la configuration
        st.session_state['portfolio_weights'] = weights
        
        # Boutons de navigation
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("⬅️ MODIFIER LES ACTIFS", use_container_width=True):
                st.session_state['current_section'] = 'explore'
                st.rerun()
        
        with col2:
            if st.button("📈 CALCULER LES RENDEMENTS ➡️", use_container_width=True, 
                        disabled=abs(total_weight - 100.0) > 0.1):
                st.session_state['current_section'] = 'returns'
                st.rerun()

# ============================================
# SECTION 4: CALCUL DES RENDEMENTS
# ============================================
elif st.session_state['current_section'] == 'returns':
    st.markdown('<h1 class="main-header">📈 CALCUL DES RENDEMENTS</h1>', unsafe_allow_html=True)
    
    if 'price_data' not in st.session_state or 'portfolio_weights' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord configurer le portefeuille.")
        if st.button("⬅️ RETOUR AU PORTEFEUILLE", use_container_width=True):
            st.session_state['current_section'] = 'portfolio'
            st.rerun()
        st.stop()
    
    price_data = st.session_state.get('price_data')
    weights = st.session_state.get('portfolio_weights')
    
    # Calcul des rendements
    st.markdown("### 📊 CALCUL DES RENDEMENTS LOGARITHMIQUES")
    
    with st.expander("📚 Théorie des rendements logarithmiques", expanded=True):
        st.markdown("""
        #### Définition
        Les rendements logarithmiques sont calculés comme :
        
        $$ r_t = \\ln\\left(\\frac{P_t}{P_{t-1}}\\right) $$
        
        où $P_t$ est le prix à la date $t$.
        
        #### Avantages
        1. **Additivité dans le temps** : $r_{0→T} = \\sum_{t=1}^T r_t$
        2. **Distribution plus proche de la normale**
        3. **Symétrie entre gains et pertes**
        4. **Cohérence avec la capitalisation continue**
        """)
    
    # Calculer les rendements pour chaque actif
    returns_data = calculate_returns(price_data)
    
    # Calculer le rendement du portefeuille
    portfolio_return = pd.Series(0.0, index=returns_data.index)
    for asset, weight in weights.items():
        if asset in returns_data.columns:
            portfolio_return += weight * returns_data[asset]
    
    returns_data['PORTERFEUILLE'] = portfolio_return
    
    # Afficher les résultats
    tabs = st.tabs(["📈 Visualisation", "📊 Statistiques", "📋 Données"])
    
    with tabs[0]:
        # Graphique des rendements
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Rendements des actifs', 'Rendement du portefeuille'),
            vertical_spacing=0.15
        )
        
        # Actifs individuels
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        for i, asset in enumerate(returns_data.columns[:-1]):
            fig.add_trace(
                go.Scatter(
                    x=returns_data.index,
                    y=returns_data[asset],
                    name=asset,
                    mode='lines',
                    line=dict(color=colors[i % len(colors)], width=1)
                ),
                row=1, col=1
            )
        
        # Portefeuille
        fig.add_trace(
            go.Scatter(
                x=returns_data.index,
                y=returns_data['PORTERFEUILLE'],
                name='Portefeuille',
                mode='lines',
                line=dict(color='#2C3E50', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=700,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Rendement", row=1, col=1)
        fig.update_yaxes(title_text="Rendement", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        # Statistiques des rendements
        stats_df = returns_data.describe().T
        stats_df['Skewness'] = returns_data.skew()
        stats_df['Kurtosis'] = returns_data.kurtosis()
        stats_df['VaR 95%'] = returns_data.apply(lambda x: -np.percentile(x, 5))
        stats_df['CVaR 95%'] = returns_data.apply(
            lambda x: -x[x <= np.percentile(x, 5)].mean()
        )
        
        st.dataframe(stats_df.style.format("{:.6f}"), use_container_width=True)
    
    with tabs[2]:
        # Données brutes
        st.dataframe(returns_data.style.format("{:.6f}"), use_container_width=True)
    
    # Sauvegarder les rendements
    st.session_state['returns_data'] = returns_data
    
    # Boutons de navigation
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬅️ MODIFIER PORTEFEUILLE", use_container_width=True):
            st.session_state['current_section'] = 'portfolio'
            st.rerun()
    
    with col2:
        if st.button("🎯 BACKTESTING VaR ➡️", use_container_width=True):
            st.session_state['current_section'] = 'var'
            st.rerun()

# ============================================
# SECTION 5: BACKTESTING VaR
# ============================================
elif st.session_state['current_section'] == 'var':
    st.markdown('<h1 class="main-header">🎯 BACKTESTING VALUE AT RISK</h1>', unsafe_allow_html=True)
    
    if 'returns_data' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord calculer les rendements.")
        if st.button("⬅️ RETOUR AUX RENDEMENTS", use_container_width=True):
            st.session_state['current_section'] = 'returns'
            st.rerun()
        st.stop()
    
    returns_data = st.session_state.get('returns_data')
    portfolio_returns = returns_data['PORTERFEUILLE']
    confiance = st.session_state.get('confidence_level', 95)
    
    # Paramètres du backtesting
    st.markdown("### ⚙️ PARAMÈTRES DU BACKTESTING")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        var_method = st.selectbox(
            "Méthode de calcul",
            ["Historique", "Paramétrique (Normale)", "Cornish-Fisher", "Monte Carlo"]
        )
    
    with col2:
        estimation_window = st.slider(
            "Fenêtre d'estimation (jours)",
            min_value=100,
            max_value=500,
            value=252,
            help="Nombre de jours utilisés pour estimer la VaR"
        )
    
    with col3:
        alpha = 1 - confiance/100
        st.metric("Seuil α", f"{alpha:.3f}", f"Confiance: {confiance}%")
    
    # Fonctions de calcul de VaR
    def calculate_var_historical(returns, alpha):
        return -np.percentile(returns, alpha * 100)
    
    def calculate_var_parametric(returns, alpha):
        mean = returns.mean()
        std = returns.std()
        return -(mean + std * stats.norm.ppf(alpha))
    
    def calculate_var_cornish_fisher(returns, alpha):
        mean = returns.mean()
        std = returns.std()
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        z = stats.norm.ppf(alpha)
        z_cf = z + (z**2 - 1) * skew/6 + (z**3 - 3*z) * kurt/24 - (2*z**3 - 5*z) * skew**2/36
        return -(mean + std * z_cf)
    
    # Calcul de la VaR mobile
    var_series = []
    violations = []
    
    for i in range(estimation_window, len(portfolio_returns)):
        train_data = portfolio_returns.iloc[i-estimation_window:i]
        
        if var_method == "Historique":
            var = calculate_var_historical(train_data, alpha)
        elif var_method == "Paramétrique (Normale)":
            var = calculate_var_parametric(train_data, alpha)
        elif var_method == "Cornish-Fisher":
            var = calculate_var_cornish_fisher(train_data, alpha)
        else:  # Monte Carlo simplifié
            mean = train_data.mean()
            std = train_data.std()
            simulations = np.random.normal(mean, std, 10000)
            var = -np.percentile(simulations, alpha * 100)
        
        var_series.append(var)
        
        # Vérifier la violation
        actual_return = portfolio_returns.iloc[i]
        violation = 1 if actual_return < -var else 0
        violations.append(violation)
    
    # Créer les séries
    var_series = pd.Series(var_series, index=portfolio_returns.index[estimation_window:])
    violations_series = pd.Series(violations, index=portfolio_returns.index[estimation_window:])
    
    # Visualisation
    st.markdown("### 📈 VaR vs RENDEMENTS RÉELS")
    
    fig = go.Figure()
    
    # Rendements
    fig.add_trace(go.Scatter(
        x=portfolio_returns.index[estimation_window:],
        y=portfolio_returns.iloc[estimation_window:],
        name='Rendements',
        mode='lines',
        line=dict(color='blue', width=1)
    ))
    
    # VaR
    fig.add_trace(go.Scatter(
        x=var_series.index,
        y=-var_series,
        name=f'VaR ({confiance}%)',
        mode='lines',
        line=dict(color='red', width=2)
    ))
    
    # Violations
    violation_dates = violations_series[violations_series == 1].index
    violation_returns = portfolio_returns.loc[violation_dates]
    
    fig.add_trace(go.Scatter(
        x=violation_dates,
        y=violation_returns,
        name='Violations',
        mode='markers',
        marker=dict(color='black', size=8, symbol='x')
    ))
    
    fig.update_layout(
        title=f'VaR {confiance}% vs rendements du portefeuille',
        xaxis_title='Date',
        yaxis_title='Rendement / VaR',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques des violations
    st.markdown("### 📊 STATISTIQUES DES VIOLATIONS")
    
    n_observations = len(violations_series)
    n_violations = violations_series.sum()
    expected_violations = n_observations * alpha
    violation_rate = n_violations / n_observations
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Observations", f"{n_observations:,}")
    
    with col2:
        st.metric("Violations", f"{n_violations:,}", 
                 f"Attendues: {expected_violations:.1f}")
    
    with col3:
        st.metric("Taux observé", f"{violation_rate:.2%}")
    
    with col4:
        st.metric("Taux attendu", f"{alpha:.2%}")
    
    # Tests statistiques
    st.markdown("### 🧪 TESTS STATISTIQUES")
    
    # 1. Test de Kupiec (Couverture inconditionnelle)
    p_theorique = alpha
    p_empirique = violation_rate
    
    if p_empirique > 0 and p_empirique < 1:
        LR_uc = -2 * np.log(
            ((1-p_theorique)**(n_observations-n_violations) * p_theorique**n_violations) /
            ((1-p_empirique)**(n_observations-n_violations) * p_empirique**n_violations)
        )
    else:
        LR_uc = np.inf
    
    chi2_critique_1 = stats.chi2.ppf(0.95, df=1)
    kupiec_pvalue = 1 - stats.chi2.cdf(LR_uc, df=1) if LR_uc < np.inf else 0
    
    # 2. Test d'Indépendance (Christoffersen)
    # Créer une matrice de transition
    violations_list = violations_series.tolist()
    
    # Compter les transitions
    n00 = n01 = n10 = n11 = 0
    
    for i in range(1, len(violations_list)):
        if violations_list[i-1] == 0 and violations_list[i] == 0:
            n00 += 1
        elif violations_list[i-1] == 0 and violations_list[i] == 1:
            n01 += 1
        elif violations_list[i-1] == 1 and violations_list[i] == 0:
            n10 += 1
        elif violations_list[i-1] == 1 and violations_list[i] == 1:
            n11 += 1
    
    # Probabilités conditionnelles
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    
    # Vraisemblance sous H0 (indépendance)
    L0 = ((1-pi)**(n00+n10) * pi**(n01+n11))
    
    # Vraisemblance sous H1 (dépendance)
    L1 = ((1-pi0)**n00 * pi0**n01) * ((1-pi1)**n10 * pi1**n11)
    
    # Statistique du test d'indépendance
    if L0 > 0 and L1 > 0:
        LR_ind = -2 * np.log(L0 / L1)
    else:
        LR_ind = np.inf
    
    chi2_critique_ind = stats.chi2.ppf(0.95, df=1)
    ind_pvalue = 1 - stats.chi2.cdf(LR_ind, df=1) if LR_ind < np.inf else 0
    
    # 3. Test de Couverture Conditionnelle (Christoffersen)
    LR_cc = LR_uc + LR_ind
    chi2_critique_2 = stats.chi2.ppf(0.95, df=2)
    cc_pvalue = 1 - stats.chi2.cdf(LR_cc, df=2) if LR_cc < np.inf else 0
    
    # Afficher les résultats dans des onglets
    tabs = st.tabs(["📊 Test Kupiec", "📈 Test Indépendance", "🎯 Test Christoffersen", "📋 Synthèse"])
    
    with tabs[0]:
        st.markdown("#### Test de Kupiec (1995)")
        st.markdown("**Couverture inconditionnelle**")
        st.markdown(f"""
        - **Hypothèse H₀** : Proportion violations = {p_theorique:.2%}
        - **Hypothèse H₁** : Proportion violations ≠ {p_theorique:.2%}
        
        **Statistiques :**
        - Nombre observations : {n_observations:,}
        - Violations observées : {n_violations:,}
        - Violations attendues : {expected_violations:.1f}
        - Taux observé : {p_empirique:.2%}
        - Taux attendu : {p_theorique:.2%}
        
        **Test :**
        - Statistique LR : {LR_uc:.4f}
        - Valeur critique (χ²₁,₀.₉₅) : {chi2_critique_1:.4f}
        - p-value : {kupiec_pvalue:.4f}
        """)
        
        if kupiec_pvalue < 0.05:
            st.markdown('<div class="test-result reject">❌ REJET H₀: Le modèle ne passe pas le test de couverture</div>', unsafe_allow_html=True)
            st.info("**Interprétation** : La fréquence des violations est significativement différente de la fréquence attendue.")
        else:
            st.markdown('<div class="test-result accept">✅ ACCEPTÉ H₀: Le modèle passe le test de couverture</div>', unsafe_allow_html=True)
            st.success("**Interprétation** : La fréquence des violations est cohérente avec le niveau de confiance.")
    
    with tabs[1]:
        st.markdown("#### Test d'Indépendance (Christoffersen 1998)")
        st.markdown("**Vérification de l'absence de clustering**")
        st.markdown(f"""
        **Matrice de transition :**
        
        | État t-1 → État t | 0 → 0 | 0 → 1 | 1 → 0 | 1 → 1 |
        |-------------------|-------|-------|-------|-------|
        | Nombre            | {n00} | {n01} | {n10} | {n11} |
        
        **Probabilités conditionnelles :**
        - P(1|0) = {pi0:.4f}
        - P(1|1) = {pi1:.4f}
        - P(1) = {pi:.4f}
        
        **Test :**
        - Statistique LR : {LR_ind:.4f}
        - Valeur critique (χ²₁,₀.₉₅) : {chi2_critique_ind:.4f}
        - p-value : {ind_pvalue:.4f}
        """)
        
        # Analyse du clustering
        clustering_detected = pi1 > pi0 * 1.5  # Seuil arbitraire pour détecter clustering
        
        if ind_pvalue < 0.05:
            st.markdown('<div class="test-result reject">❌ REJET H₀: Les violations ne sont pas indépendantes</div>', unsafe_allow_html=True)
            if clustering_detected:
                st.warning("**Clustering détecté** : Les violations ont tendance à se regrouper dans le temps.")
            else:
                st.info("**Pattern non aléatoire** : Les violations suivent un pattern particulier.")
        else:
            st.markdown('<div class="test-result accept">✅ ACCEPTÉ H₀: Les violations sont indépendantes</div>', unsafe_allow_html=True)
            st.success("**Interprétation** : Aucune évidence de clustering temporel.")
    
    with tabs[2]:
        st.markdown("#### Test de Couverture Conditionnelle (Christoffersen)")
        st.markdown("**Test global de validité du modèle**")
        st.markdown(f"""
        **Combinaison des deux tests précédents :**
        - LR_cc = LR_uc + LR_ind
        - LR_cc = {LR_uc:.4f} + {LR_ind:.4f} = {LR_cc:.4f}
        
        **Distribution sous H₀ :** χ²(2)
        
        **Test :**
        - Statistique LR : {LR_cc:.4f}
        - Valeur critique (χ²₂,₀.₉₅) : {chi2_critique_2:.4f}
        - p-value : {cc_pvalue:.4f}
        """)
        
        if cc_pvalue < 0.05:
            st.markdown('<div class="test-result reject">❌ REJET H₀: Le modèle n\'est pas valide</div>', unsafe_allow_html=True)
            st.error("**Conclusion** : Le modèle de VaR ne respecte pas la propriété de couverture conditionnelle.")
        else:
            st.markdown('<div class="test-result accept">✅ ACCEPTÉ H₀: Le modèle est valide</div>', unsafe_allow_html=True)
            st.success("**Conclusion** : Le modèle de VaR est statistiquement valide.")
    
    with tabs[3]:
        st.markdown("#### 📋 SYNTHÈSE DES TESTS")
        
        # Créer un tableau de synthèse
        synthèse_data = {
            'Test': ['Kupiec (Couverture)', 'Indépendance', 'Christoffersen (Global)'],
            'Statistique': [f"{LR_uc:.4f}", f"{LR_ind:.4f}", f"{LR_cc:.4f}"],
            'Valeur critique': [f"{chi2_critique_1:.4f}", f"{chi2_critique_ind:.4f}", f"{chi2_critique_2:.4f}"],
            'p-value': [f"{kupiec_pvalue:.4f}", f"{ind_pvalue:.4f}", f"{cc_pvalue:.4f}"],
            'Décision': [
                '✅ Accepté' if kupiec_pvalue >= 0.05 else '❌ Rejeté',
                '✅ Accepté' if ind_pvalue >= 0.05 else '❌ Rejeté',
                '✅ Accepté' if cc_pvalue >= 0.05 else '❌ Rejeté'
            ]
        }
        
        df_synthèse = pd.DataFrame(synthèse_data)
        st.dataframe(df_synthèse, use_container_width=True)
        
        # Conclusion globale
        st.markdown("#### 🎯 CONCLUSION GLOBALE")
        
        if cc_pvalue >= 0.05:
            st.success("""
            ✅ **MODÈLE VALIDE**
            
            Le modèle de VaR passe tous les tests statistiques :
            1. Fréquence des violations conforme au niveau de confiance
            2. Aucune évidence de clustering temporel
            3. Propriété de couverture conditionnelle respectée
            """)
        else:
            st.error("""
            ❌ **MODÈLE NON VALIDE**
            
            Le modèle de VaR ne passe pas tous les tests :
            """)
            
            if kupiec_pvalue < 0.05:
                st.warning("• **Problème de couverture** : Fréquence des violations incorrecte")
            if ind_pvalue < 0.05:
                st.warning("• **Problème d'indépendance** : Violations groupées dans le temps")
            
            st.info("""
            **Recommandations :**
            1. Recalibrer le modèle avec plus de données
            2. Essayer une autre méthode de calcul de VaR
            3. Ajuster les paramètres d'estimation
            4. Considérer des modèles GARCH pour mieux capturer la volatilité
            """)
    
    # Visualisation des transitions
    st.markdown("### 🔄 ANALYSE DES TRANSITIONS")
    
    fig_transitions = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Pas de violation (t-1)", "Violation (t-1)", "Pas de violation (t)", "Violation (t)"],
            color=["#2E86AB", "#A23B72", "#2E86AB", "#A23B72"]
        ),
        link=dict(
            source=[0, 0, 1, 1],  # indices correspondant aux labels
            target=[2, 3, 2, 3],
            value=[n00, n01, n10, n11],
            label=[f"{n00} transitions", f"{n01} transitions", f"{n10} transitions", f"{n11} transitions"]
        )
    )])
    
    fig_transitions.update_layout(
        title="Diagramme de Sankey - Transitions entre états",
        font_size=12,
        height=400
    )
    
    st.plotly_chart(fig_transitions, use_container_width=True)
    
    # Évaluation complémentaire
    st.markdown("### 📊 ÉVALUATION COMPLÉMENTAIRE")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Ratio violations
        ratio = violation_rate / alpha
        st.metric("Ratio Violations/Observé", f"{ratio:.2f}")
        
        if ratio > 1.5:
            st.error("Sous-estimation sévère du risque")
        elif ratio > 1.2:
            st.warning("Légère sous-estimation")
        elif ratio < 0.8:
            st.warning("Surestimation du risque")
        elif ratio < 0.5:
            st.error("Surestimation sévère")
        else:
            st.success("Calibration adéquate")
    
    with col2:
        # Test de séries
        from statsmodels.tsa.stattools import acf
        
        acf_values = acf(violations_series, nlags=5, fft=False)
        autocorr_max = np.max(np.abs(acf_values[1:]))  # Exclure le lag 0
        
        st.metric("Autocorrélation max (lag 1-5)", f"{autocorr_max:.3f}")
        
        if autocorr_max > 0.2:
            st.warning("Autocorrélation détectée")
        else:
            st.success("Pas d'autocorrélation significative")
    
    # Sauvegarder tous les résultats
    st.session_state['var_results'] = {
        'var_series': var_series,
        'violations': violations_series,
        'test_results': {
            'kupiec': {
                'LR': LR_uc,
                'critical_value': chi2_critique_1,
                'pvalue': kupiec_pvalue,
                'passed': kupiec_pvalue >= 0.05
            },
            'independence': {
                'LR': LR_ind,
                'critical_value': chi2_critique_ind,
                'pvalue': ind_pvalue,
                'passed': ind_pvalue >= 0.05,
                'transitions': {'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11},
                'probabilities': {'pi0': pi0, 'pi1': pi1, 'pi': pi}
            },
            'christoffersen': {
                'LR': LR_cc,
                'critical_value': chi2_critique_2,
                'pvalue': cc_pvalue,
                'passed': cc_pvalue >= 0.05
            }
        }
    }
    
    # Boutons de navigation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬅️ MODIFIER RENDEMENTS", use_container_width=True):
            st.session_state['current_section'] = 'returns'
            st.rerun()
    
    with col2:
        if st.button("📈 ANALYSER DÉTAILS", use_container_width=True):
            # Option pour plus d'analyses
            pass
    
    with col3:
        if st.button("📊 BACKTESTING CVaR ➡️", use_container_width=True):
            st.session_state['current_section'] = 'cvar'
            st.rerun()
# ============================================
# SECTION 6: BACKTESTING CVaR
# ============================================
elif st.session_state['current_section'] == 'cvar':
    st.markdown('<h1 class="main-header">📊 BACKTESTING CONDITIONAL VaR</h1>', unsafe_allow_html=True)
    
    if 'var_results' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord effectuer le backtesting VaR.")
        if st.button("⬅️ RETOUR À VaR", use_container_width=True):
            st.session_state['current_section'] = 'var'
            st.rerun()
        st.stop()
    
    returns_data = st.session_state.get('returns_data')
    portfolio_returns = returns_data['PORTERFEUILLE']
    var_results = st.session_state.get('var_results')
    
    st.markdown("""
    <div class='info-card'>
    <h3>📊 Conditional Value at Risk (CVaR)</h3>
    <p>La CVaR (Expected Shortfall) mesure la perte moyenne dans les pires α% des cas.</p>
    <p><strong>Définition :</strong> CVaR_α = E[L | L > VaR_α]</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Calcul de la CVaR
    st.markdown("### 🧮 CALCUL DE LA CVaR")
    
    var_series = var_results['var_series']
    violations = var_results['violations']
    
    # Calculer la CVaR historique
    cvar_series = []
    cvar_dates = []
    
    for i, date in enumerate(var_series.index):
        if i >= 250:  # Fenêtre de 250 jours
            window_returns = portfolio_returns.loc[:date].iloc[-250:]
            window_var = var_series.loc[date]
            
            # Rendements dans la queue (pires α%)
            tail_returns = window_returns[window_returns < -window_var]
            
            if len(tail_returns) > 0:
                cvar = -tail_returns.mean()
                cvar_series.append(cvar)
                cvar_dates.append(date)
    
    if len(cvar_series) == 0:
        st.error("❌ Pas assez d'observations pour calculer la CVaR")
        st.stop()
    
    cvar_series = pd.Series(cvar_series, index=cvar_dates)
    
    # Visualisation
    st.markdown("### 📈 COMPARAISON VaR/CVaR")
    
    fig = go.Figure()
    
    # Dates communes
    common_dates = cvar_series.index.intersection(var_series.index)
    
    # VaR
    fig.add_trace(go.Scatter(
        x=common_dates,
        y=-var_series.loc[common_dates],
        name=f'VaR ({confiance}%)',
        mode='lines',
        line=dict(color='red', width=2)
    ))
    
    # CVaR
    fig.add_trace(go.Scatter(
        x=common_dates,
        y=-cvar_series.loc[common_dates],
        name=f'CVaR ({confiance}%)',
        mode='lines',
        line=dict(color='orange', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'Comparaison VaR et CVaR ({confiance}%)',
        xaxis_title='Date',
        yaxis_title='Mesure de risque',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Test de la CVaR
    st.markdown("### 🧪 TEST DE LA CVaR")
    
    # Identifier les violations de la VaR
    violation_dates = violations[violations == 1].index
    common_violation_dates = violation_dates.intersection(cvar_series.index)
    
    if len(common_violation_dates) < 10:
        st.warning(f"⚠️ Insuffisant de violations ({len(common_violation_dates)}) pour tester la CVaR")
    else:
        # Calculer les écarts
        gaps = []
        for date in common_violation_dates:
            actual_loss = -portfolio_returns.loc[date]
            cvar_loss = cvar_series.loc[date]
            gap = actual_loss - cvar_loss
            gaps.append(gap)
        
        gaps = np.array(gaps)
        
        # Test t
        t_stat, p_value = stats.ttest_1samp(gaps, 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Résultats du test")
            st.markdown(f"""
            - Violations testées: {len(gaps)}
            - Écart moyen: {gaps.mean():.6f}
            - Écart-type: {gaps.std():.6f}
            - Statistique t: {t_stat:.4f}
            - p-value: {p_value:.4f}
            """)
        
        with col2:
            st.markdown("#### Décision")
            if p_value < 0.05:
                st.markdown('<div class="test-result reject">❌ REJET: La CVaR n\'est pas correctement estimée</div>', unsafe_allow_html=True)
                if gaps.mean() > 0:
                    st.error("Les pertes observées dépassent la CVaR estimée")
                else:
                    st.warning("Les pertes observées sont inférieures à la CVaR estimée")
            else:
                st.markdown('<div class="test-result accept">✅ ACCEPTÉ: La CVaR est correctement estimée</div>', unsafe_allow_html=True)
    
    # Ratio CVaR/VaR
    st.markdown("### 🔄 RATIO CVaR / VaR")
    
    ratio_series = cvar_series / var_series.loc[cvar_series.index]
    
    fig_ratio = go.Figure()
    
    fig_ratio.add_trace(go.Scatter(
        x=ratio_series.index,
        y=ratio_series,
        name='Ratio CVaR/VaR',
        mode='lines',
        line=dict(color='green', width=2)
    ))
    
    fig_ratio.add_hline(
        y=ratio_series.mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Moyenne: {ratio_series.mean():.3f}"
    )
    
    fig_ratio.update_layout(
        title='Évolution du ratio CVaR / VaR',
        xaxis_title='Date',
        yaxis_title='Ratio',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_ratio, use_container_width=True)
    
    # Boutons de navigation
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬅️ RETOUR À VaR", use_container_width=True):
            st.session_state['current_section'] = 'var'
            st.rerun()
    
    with col2:
        if st.button("📈 VISUALISATIONS ➡️", use_container_width=True):
            st.session_state['current_section'] = 'visualize'
            st.rerun()

# ============================================
# SECTION 7: VISUALISATIONS
# ============================================
elif st.session_state['current_section'] == 'visualize':
    st.markdown('<h1 class="main-header">📈 VISUALISATIONS AVANCÉES</h1>', unsafe_allow_html=True)
    
    if 'returns_data' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord effectuer les analyses.")
        if st.button("⬅️ RETOUR AU DÉBUT", use_container_width=True):
            st.session_state['current_section'] = 'upload'
            st.rerun()
        st.stop()
    
    returns_data = st.session_state.get('returns_data')
    portfolio_returns = returns_data['PORTERFEUILLE']
    
    # Sélection des visualisations
    st.markdown("### 🎨 CHOISISSEZ VOS VISUALISATIONS")
    
    viz_options = st.multiselect(
        "Sélectionnez les graphiques à afficher :",
        [
            "Distribution des rendements",
            "QQ-Plot (normalité)",
            "Fonction d'autocorrélation",
            "Volatilité mobile",
            "Heatmap des rendements",
            "Analyse des queues de distribution"
        ],
        default=["Distribution des rendements", "QQ-Plot (normalité)"]
    )
    
    if "Distribution des rendements" in viz_options:
        st.markdown("#### 📊 DISTRIBUTION DES RENDEMENTS")
        
        fig = go.Figure()
        
        # Histogramme
        fig.add_trace(go.Histogram(
            x=portfolio_returns,
            nbinsx=50,
            name='Rendements',
            opacity=0.7,
            marker_color='#4ECDC4'
        ))
        
        # Courbe normale
        x_norm = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100)
        y_norm = stats.norm.pdf(x_norm, portfolio_returns.mean(), portfolio_returns.std())
        
        fig.add_trace(go.Scatter(
            x=x_norm,
            y=y_norm,
            name='Distribution normale',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Distribution des rendements vs normale',
            xaxis_title='Rendement',
            yaxis_title='Densité',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    if "QQ-Plot (normalité)" in viz_options:
        st.markdown("#### 📈 QQ-PLOT (TEST DE NORMALITÉ)")
        
        # Calcul du QQ-Plot
        qq = stats.probplot(portfolio_returns, dist="norm", fit=True)
        x_theoretical = qq[0][0]
        y_observed = qq[0][1]
        
        fig = go.Figure()
        
        # Points
        fig.add_trace(go.Scatter(
            x=x_theoretical,
            y=y_observed,
            mode='markers',
            name='Données',
            marker=dict(size=6, color='#FF6B6B')
        ))
        
        # Droite de référence
        x_line = np.array([x_theoretical.min(), x_theoretical.max()])
        y_line = qq[1][0] + qq[1][1] * x_line
        
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='Normale théorique',
            line=dict(color='#2C3E50', width=2)
        ))
        
        fig.update_layout(
            title='QQ-Plot des rendements',
            xaxis_title='Quantiles théoriques',
            yaxis_title='Quantiles observés',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    if "Fonction d'autocorrélation" in viz_options:
        st.markdown("#### 🔄 FONCTION D'AUTOCORRÉLATION")
        
        # Calcul de l'ACF
        n_lags = st.slider("Nombre de décalages", 10, 100, 40)
        acf_values = np.correlate(portfolio_returns - portfolio_returns.mean(), 
                                 portfolio_returns - portfolio_returns.mean(), 
                                 mode='full')
        acf_values = acf_values[len(acf_values)//2:len(acf_values)//2 + n_lags + 1] / acf_values[len(acf_values)//2]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(range(n_lags + 1)),
            y=acf_values,
            name='ACF',
            marker_color='#45B7D1'
        ))
        
        # Bande de confiance
        conf_band = 1.96 / np.sqrt(len(portfolio_returns))
        fig.add_hline(y=conf_band, line_dash="dash", line_color="red")
        fig.add_hline(y=-conf_band, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title=f'Fonction d\'autocorrélation ({n_lags} décalages)',
            xaxis_title='Décalage',
            yaxis_title='Autocorrélation',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Boutons de navigation
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬅️ RETOUR À CVaR", use_container_width=True):
            st.session_state['current_section'] = 'cvar'
            st.rerun()
    
    with col2:
        if st.button("📝 GÉNÉRER RAPPORT ➡️", use_container_width=True):
            st.session_state['current_section'] = 'report'
            st.rerun()

# ============================================
# SECTION 8: RAPPORT
# ============================================
elif st.session_state['current_section'] == 'report':
    st.markdown('<h1 class="main-header">📝 RAPPORT COMPLET</h1>', unsafe_allow_html=True)
    
    # Vérifier que les analyses sont complètes
    required_data = ['returns_data', 'portfolio_weights', 'var_results']
    missing_data = [d for d in required_data if d not in st.session_state]
    
    if missing_data:
        st.error(f"❌ Données manquantes : {', '.join(missing_data)}")
        if st.button("⬅️ RETOUR AU DÉBUT", use_container_width=True):
            st.session_state['current_section'] = 'upload'
            st.rerun()
        st.stop()
    
    # Générer le rapport
    st.markdown("### 📋 SYNTHÈSE DE L'ANALYSE")
    
    # Informations générales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💼 Portefeuille", f"{st.session_state.get('portfolio_value', 0):,.0f} €")
    
    with col2:
        st.metric("🎯 Confiance", f"{st.session_state.get('confidence_level', 95)}%")
    
    with col3:
        assets = st.session_state.get('selected_assets', [])
        st.metric("📊 Actifs", f"{len(assets)}")
    
    with col4:
        returns_data = st.session_state.get('returns_data')
        if returns_data is not None:
            st.metric("📈 Observations", f"{len(returns_data):,}")
    
    # Résultats VaR
    st.markdown("### 🎯 RÉSULTATS BACKTESTING VaR")
    
    var_results = st.session_state.get('var_results', {})
    violations = var_results.get('violations', pd.Series([]))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        violation_rate = violations.mean() if len(violations) > 0 else 0
        st.metric("Taux de violation", f"{violation_rate:.2%}")
    
    with col2:
        expected_rate = 1 - st.session_state.get('confidence_level', 95)/100
        st.metric("Taux attendu", f"{expected_rate:.2%}")
    
    with col3:
        test_passed = var_results.get('test_results', {}).get('passed', False)
        status = "✅ PASSÉ" if test_passed else "❌ ÉCHEC"
        st.metric("Test Kupiec", status)
    
    # Recommandations
    st.markdown("### 💡 RECOMMANDATIONS")
    
    if test_passed:
        st.success("""
        ✅ **Le modèle de risque est valide**
        
        **Actions recommandées :**
        - Continuer à utiliser le modèle actuel
        - Surveiller régulièrement les violations
        - Recalibrer le modèle chaque trimestre
        """)
    else:
        st.warning("""
        ⚠️ **Le modèle nécessite des ajustements**
        
        **Actions recommandées :**
        - Recalibrer le modèle avec plus de données
        - Considérer d'autres méthodes de calcul
        - Augmenter les tests de robustesse
        - Implémenter des stress tests supplémentaires
        """)
    
    # Exporter le rapport
    st.markdown("### 📥 EXPORTATION")
    
    # Créer un DataFrame de synthèse
    report_data = {
        'Paramètre': [
            'Date du rapport',
            'Valeur du portefeuille (€)',
            'Niveau de confiance (%)',
            'Nombre d\'actifs',
            'Période analysée',
            'Observations',
            'Taux de violation (%)',
            'Taux attendu (%)',
            'Test Kupiec (p-value)',
            'Décision du test'
        ],
        'Valeur': [
            datetime.now().strftime('%Y-%m-%d %H:%M'),
            f"{st.session_state.get('portfolio_value', 0):,.0f}",
            str(st.session_state.get('confidence_level', 95)),
            str(len(st.session_state.get('selected_assets', []))),
            f"{returns_data.index[0].date()} → {returns_data.index[-1].date()}" if returns_data is not None else "N/A",
            f"{len(returns_data):,}" if returns_data is not None else "N/A",
            f"{violation_rate:.2%}" if len(violations) > 0 else "N/A",
            f"{expected_rate:.2%}",
            f"{var_results.get('test_results', {}).get('pvalue', 'N/A')}",
            'Accepté' if test_passed else 'Rejeté'
        ]
    }
    
    df_report = pd.DataFrame(report_data)
    
    # Afficher le rapport
    st.dataframe(df_report, use_container_width=True, hide_index=True)
    
    # Bouton d'export
    if st.button("📥 TÉLÉCHARGER LE RAPPORT (Excel)", use_container_width=True):
        # Créer le fichier Excel
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_report.to_excel(writer, sheet_name='Synthèse', index=False)
            
            # Ajouter les données détaillées
            if 'returns_data' in st.session_state:
                st.session_state['returns_data'].to_excel(writer, sheet_name='Rendements')
            
            if 'var_results' in st.session_state:
                pd.DataFrame({
                    'Date': var_results.get('var_series', pd.Series()).index,
                    'VaR': var_results.get('var_series', pd.Series()).values,
                    'Violation': var_results.get('violations', pd.Series()).values
                }).to_excel(writer, sheet_name='Backtesting_VaR', index=False)
        
        report_bytes = output.getvalue()
        
        # Téléchargement
        st.download_button(
            label="✅ CLIQUEZ POUR TÉLÉCHARGER",
            data=report_bytes,
            file_name=f"rapport_backtesting_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # Bouton pour recommencer
    if st.button("🔄 NOUVELLE ANALYSE", use_container_width=True):
        # Réinitialiser la session
        for key in list(st.session_state.keys()):
            if key != 'current_section':
                del st.session_state[key]
        st.session_state['current_section'] = 'upload'
        st.rerun()

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 2rem;">
    <p style="font-size: 1.1rem; font-weight: 600; color: #2C3E50;">📊 APPLICATION DE BACKTESTING FINANCIER</p>
    <p>Développé avec Streamlit • Méthodologies académiques • Outil pédagogique</p>
    <p>© 2024 - Analyse de risque financier</p>
</div>
""", unsafe_allow_html=True)




