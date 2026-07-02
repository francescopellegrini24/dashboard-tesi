import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 0. GESTIONE LIBRERIE
# ==============================================================================
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ==============================================================================
# 1. CONFIGURAZIONE PAGINA
# ==============================================================================
st.set_page_config(page_title="V.E.R.A. - Vendor Rating AI", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetric"] { 
        background-color: #ffffff; padding: 15px; border-radius: 8px; 
        border: 1px solid #e9ecef; box-shadow: 1px 1px 4px rgba(0,0,0,0.05);
    }
    .conclusion-box { 
        background-color: #f0f7f4; border-left: 6px solid #2e7d32; 
        padding: 20px; border-radius: 6px; margin-top: 15px; 
    }
    .switch-box { 
        background-color: #e3f2fd; border-left: 5px solid #1976d2; 
        padding: 15px; border-radius: 8px; margin-bottom: 20px; 
    }
    .comparison-title { color: #1976d2; font-weight: bold; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ V.E.R.A. — Vendor Evaluation & Rating AI")
st.write("Questa applicazione utilizza algoritmi di Machine Learning per analizzare le prestazioni dei fornitori e calcolare un punteggio di affidabilità sintetico, integrando metriche operative tradizionali e parametri ESG.")

# ==============================================================================
# 2. CARICAMENTO DATI
# ==============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv('Dataset_Fornitori_1000_Realistico_Dashboard.csv', header=4)
    df.columns = df.columns.str.strip()
    
    features_raw = ['Costo Unitario (€)', 'Lead Time (Giorni)', 'Difettosità (%)', 
                    'Distanza (Km)', 'Emissioni CO2 (Kg)', 'ISO 14001', 'Energia Rinnov. (%)']
    
    features_norm = ['Norm. (Costo Unitario)', 'Norm. Lead Time', 'Norm. Difettosità', 
                     'Norm. Distanza', 'Norm. CO2', 'Norm. ISO', 'Norm. Rinnovabili']
    
    target = 'AI-Ready Score (1-100)'
    df = df[['FORNITORI'] + features_raw + features_norm + [target]].dropna()
    return df, features_raw, features_norm, target

data, features_raw, features_norm, target = load_data()

etichette_grafico = ['Costo', 'Lead Time', 'Difettosità', 'Distanza', 'CO2', 'ISO 14001', 'Rinnovabili']
pesi_excel_teorici = [35.0, 10.0, 10.0, 10.0, 20.0, 5.0, 10.0]

# ==============================================================================
# 3. MOTORE DI ADDESTRAMENTO V.E.R.A.
# ==============================================================================
def allena_modelli(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelli = {
        'Regressione Lineare': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=5),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    metriche = {}
    pesi_modelli = {'Modello Teorico Standard': pesi_excel_teorici}
    
    for nome, modello in modelli.items():
        modello.fit(X_train, y_train)
        y_pred = modello.predict(X_test)
        
        metriche[nome] = {
            'Precisione (R²)': r2_score(y_test, y_pred),
            'Errore (MSE)': mean_squared_error(y_test, y_pred)
        }
        
        if nome == 'Regressione Lineare':
            coef_abs = np.abs(modello.coef_)
            pesi_modelli[nome] = (coef_abs / coef_abs.sum()) * 100
        else:
            pesi_modelli[nome] = modello.feature_importances_ * 100
            
    best_name = max(metriche, key=lambda k: metriche[k]['Precisione (R²)'])
    return modelli, metriche, pesi_modelli, best_name, X_train

# Allenamento dei modelli sulle due configurazioni di dati
mod_raw, met_raw, pesi_raw, best_raw, X_tr_raw = allena_modelli(data[features_raw], data[target])
mod_norm, met_norm, pesi_norm, best_norm, X_tr_norm = allena_modelli(data[features_norm], data[target])

# ==============================================================================
# 4. CONFIGURAZIONE INTERFACCIA V.E.R.A.
# ==============================================================================
tab_score, tab_modelli, tab_simulatore = st.tabs([
    "🎯 1. Panoramica Dati", 
    "🔬 2. Analisi e Confronto Modelli", 
    "🔮 3. Simulatore Predittivo"
])

# ------------------------------------------------------------------------------
# TAB 1: PANORAMICA DATI
# ------------------------------------------------------------------------------
with tab_score:
    st.subheader("🎯 Distribuzione dei Punteggi dei Fornitori")
    st.write(f"Riepilogo statistico delle valutazioni calcolate dal sistema V.E.R.A. su un campione di **{len(data)} fornitori**.")
    
    st.metric(label="📊 Punteggio Medio Complessivo", value=f"{data[target].mean():.2f} / 100")
    
    st.markdown("---")
    st.write("**Database completo e classificazione ordinata delle prestazioni:**")
    st.dataframe(data[['FORNITORI'] + features_raw + [target]].sort_values(by=target, ascending=False), hide_index=True)

# ------------------------------------------------------------------------------
# TAB 2: VALUTAZIONE MODELLI AI
# ------------------------------------------------------------------------------
with tab_modelli:
    st.markdown("""
    <div class="switch-box">
        <h4 style="margin-top: 0px; color: #0d47a1;">🎛️ Configurazione dei Dati di Input</h4>
        Selezionare la modalità di elaborazione dei dati per osservare come la scala dei valori influenzi l'importanza assegnata dalle varianti algoritmiche alle singole variabili.
    </div>
    """, unsafe_allow_html=True)
    
    tipo_dati = st.radio(
        label="Seleziona la modalità di elaborazione dei dati:",
        options=[
            "📉 1. DATI GREZZI: I modelli elaborano i valori espressi nelle unità di misura originali (Euro, Giorni, Kilometri, Percentuali).", 
            "📏 2. DATI NORMALIZZATI: I modelli elaborano i valori precedentemente convertiti su una scala uniforme compresa tra 0 e 1."
        ],
        label_visibility="collapsed"
    )

    if "GREZZI" in tipo_dati:
        metriche_attive, pesi_attivi, best_name_attivo = met_raw, pesi_raw, best_raw
        titolo_grafico = "Importanza delle variabili calcolata dai modelli su Dati Grezzi"
        spiegazione_testo = "Gli algoritmi elaborano i dati mantenendo le unità di misura originali. Poiché le variabili presentano ordini di grandezza molto differenti, i modelli adattano matematicamente i coefficienti interni per compensare gli squilibri di scala numerica, raggiungendo un'elevata precisione predittiva."
    else:
        metriche_attive, pesi_attivi, best_name_attivo = met_norm, pesi_norm, best_norm
        titolo_grafico = "Importanza delle variabili calcolata dai modelli su Dati Normalizzati"
        spiegazione_testo = "Gli algoritmi elaborano i dati dopo che sono stati ricondotti a una scala comune (0-1). L'azzeramento delle differenze di grandezza numerica permette ai modelli di identificare la relazione lineare e di riflettere l'esatta importanza percentuale stabilita nel modello teorico di partenza."

    st.markdown("---")
    
    cols = st.columns(len(metriche_attive))
    for idx, (nome_modello, metriche_modello) in enumerate(metriche_attive.items()):
        with cols[idx]:
            if nome_modello == best_name_attivo:
                st.success(f"🏆 **{nome_modello} (Migliore)**")
            else:
                st.info(f"🔹 **{nome_modello}**")
            st.metric(label="Coefficiente di Determinazione (R²)", value=f"{metriche_modello['Precisione (R²)']:.4f}")

    st.markdown("---")
    st.write(f"💡 **Analisi dello scenario:** {spiegazione_testo}")

    # GENERAZIONE GRAFICO DELLE IMPORTANZE
    df_pesi = pd.DataFrame(pesi_attivi, index=etichette_grafico)
    fig_pesi, ax_pesi = plt.subplots(figsize=(15, 7), dpi=150)
    x = np.arange(len(etichette_grafico))
    width = 0.15
    
    colors = ['#7f8c8d', '#2980b9', '#f39c12', '#27ae60', '#d35400']
    for i, (col, color) in enumerate(zip(df_pesi.columns, colors)):
        bars = ax_pesi.bar(x + (i - 2) * width, df_pesi[col], width, label=col, color=color, edgecolor='white', alpha=0.9)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0.3:
                ax_pesi.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 4), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7, rotation=90)

    ax_pesi.set_ylabel('Importanza Percentuale (%)', fontsize=11)
    ax_pesi.set_title(titolo_grafico, fontweight='bold', fontsize=13)
    ax_pesi.set_xticks(x)
    ax_pesi.set_xticklabels(etichette_grafico, rotation=0, ha='center', fontsize=11)
    ax_pesi.legend(frameon=True, facecolor='white', edgecolor='none')
    ax_pesi.grid(axis='y', linestyle='--', alpha=0.3)
    ax_pesi.spines['top'].set_visible(False)
    ax_pesi.spines['right'].set_visible(False)
    ax_pesi.set_ylim(0, df_pesi.max().max() * 1.25)
    st.pyplot(fig_pesi)

    # CONCLUSIONE METODOLOGICA DIRETTA V.E.R.A.
    st.markdown(f"""
    <div class="conclusion-box">
        <h4 style='color: #1b5e20; margin-top:0;'>📊 Nota Metodologica sull'Effetto Scala</h4>
        Il confronto evidenzia l'impatto dell'effetto scala sui modelli di Machine Learning. Nella prima configurazione (dati grezzi), lo scostamento dei pesi rispetto al modello teorico costituisce una correzione matematica necessaria per gestire variabili con scale numeriche eterogenee. Nella seconda configurazione (dati normalizzati), l'omogeneizzazione dell'input elimina l'effetto scala, consentendo agli algoritmi di replicare fedelmente la ponderazione lineare originaria e dimostrando la stabilità e la coerenza del processo di apprendimento.
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 3: SIMULATORE PREDITTIVO
# ------------------------------------------------------------------------------
with tab_simulatore:
    st.subheader("🔮 Simulazione Predittiva per Nuovi Fornitori")
    st.write("Inserire i parametri operativi e ambientali di un fornitore per calcolare la stima del punteggio complessivo e analizzare l'impatto di ciascun fattore sul risultato.")
    
    st.markdown("<h4 class='comparison-title'>📊 Valori Medi di Riferimento del Mercato</h4>", unsafe_allow_html=True)
    st.write("Parametri medi rilevati all'interno del dataset attuale per il confronto delle prestazioni:")
    
    col_p1, col_p2, col_p3, col_p4, col_p5, col_p6 = st.columns(6)
    with col_p1:
        st.metric(label="💰 Costo Medio", value=f"{data['Costo Unitario (€)'].mean():.2f} €")
    with col_p2:
        st.metric(label="⏱️ Lead Time Medio", value=f"{data['Lead Time (Giorni)'].mean():.1f} gg")
    with col_p3:
        st.metric(label="⚠️ Difettosità Media", value=f"{data['Difettosità (%)'].mean() * 100:.2f} %")
    with col_p4:
        st.metric(label="📍 Distanza Media", value=f"{data['Distanza (Km)'].mean():.0f} Km")
    with col_p5:
        st.metric(label="🌱 Emissioni CO2 Medie", value=f"{data['Emissioni CO2 (Kg)'].mean():.1f} Kg")
    with col_p6:
        st.metric(label="⚡ Rinnovabili Medie", value=f"{data['Energia Rinnov. (%)'].mean() * 100:.1f} %")
        
    st.markdown("---")
    
    st.markdown("#### 🔮 Inserimento dei Parametri del Fornitore:")
    col_sx, col_dx = st.columns(2)
    with col_sx:
        st.markdown("**📋 Variabili Operative ed Economiche**")
        costo = st.number_input("Costo Unitario (€)", value=11.50, min_value=0.0, step=0.10, format="%.2f")
        lead_time = st.number_input("Lead Time (Giorni)", value=25.0, min_value=0.0, step=1.0, format="%.1f")
        difettosita_pct = st.number_input("Tasso di Difettosità (%)", value=1.50, min_value=0.0, max_value=100.0, step=0.1, format="%.2f")
        
    with col_dx:
        st.markdown("**🌱 Variabili Ambientali e Logistiche**")
        distanza = st.number_input("Distanza di Trasporto (Km)", value=4000.0, min_value=0.0, step=50.0, format="%.1f")
        emissioni = st.number_input("Emissioni di CO2 (Kg)", value=55.0, min_value=0.0, step=1.0, format="%.1f")
        rinnovabili_pct = st.number_input("Quota Energia Rinnovabile (%)", value=25.0, min_value=0.0, max_value=100.0, step=1.0, format="%.1f")
        iso_str = st.selectbox("Certificazione ISO 14001", ["Presente", "Assente"])

    difettosita = difettosita_pct / 100.0
    rinnovabili = rinnovabili_pct / 100.0
    iso = 1.0 if "Presente" in iso_str else 0.0
    
    input_data = pd.DataFrame([[costo, lead_time, difettosita, distanza, emissioni, iso, rinnovabili]], columns=features_raw)

    st.markdown("---")
    if st.button("🚀 Calcola Valutazione Predittiva", type="primary"):
        modello_simulatore = mod_raw[best_raw]
        prediction = modello_simulatore.predict(input_data)[0]
        st.success(f"### Punteggio Predetto ({best_raw}): **{prediction:.2f} / 100**")
        
        st.markdown("#### 🔍 Analisi di Spiegabilità Locale tramite Valori SHAP")
        if SHAP_AVAILABLE:
            st.write("Il grafico descrive il contributo quantitativo, positivo o negativo, che ciascuna variabile apporta allo scostamento del punteggio finale rispetto al valore atteso medio del modello.")
            explainer = shap.Explainer(modello_simulatore, X_tr_raw)
            shap_values = explainer(input_data)
            fig_shap, ax_shap = plt.subplots(figsize=(10, 5), dpi=150)
            shap.plots.waterfall(shap_values[0], show=False)
            fig_shap = plt.gcf()
            fig_shap.set_facecolor('white')
            st.pyplot(fig_shap)
        else:
            st.info("ℹ️ Il punteggio è stato calcolato. La visualizzazione del grafico SHAP richiede l'installazione della relativa libreria nell'ambiente locale.")