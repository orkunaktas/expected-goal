import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Tema ayarı
st.set_page_config(page_title="Maç Simülasyonu", page_icon="⚽", layout="wide")

# Stil ayarları ve başlık
st.markdown("""
    <style>
    .title {
        font-size: 48px;
        font-family: 'Arial', sans-serif;
        text-align: center;
        color: #4A90E2;
    }
    .header {
        font-size: 24px;
        font-family: 'Arial', sans-serif;
        margin-bottom: 20px;
        color: #4A90E2;
    }
    .description {
        font-size: 16px;
        font-family: 'Arial', sans-serif;
        color: #333;
        margin-bottom: 30px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #777;
        margin-top: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# Başlık ve açıklama
st.markdown('<div class="title">Maç Sonucu ve Karşılıklı Gol Olasılıkları Hesaplama ⚽</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Bu araç, bir futbol maçında tahmini xG değerlerine göre maç sonucunu ve karşılıklı gol olasılıklarını hesaplar.</div>', unsafe_allow_html=True)

# Sidebar'a xG girişi ve tema seçimi ekleme
st.sidebar.header("Ayarlar")
xG_home = st.sidebar.number_input('Home Takım xG Değeri:', min_value=0.0, max_value=10.0, value=1.42, step=0.01)
xG_away = st.sidebar.number_input('Away Takım xG Değeri:', min_value=0.0, max_value=10.0, value=1.93, step=0.01)
num_simulations = st.sidebar.slider('Simülasyon Sayısı:', min_value=1000, max_value=20000, value=10000, step=1000)

# Bar grafiğine etiket ekleme fonksiyonu
def add_values(bars, ax):
    """Bar grafiklerine değerleri ekler."""
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}%", ha='center', va='bottom', fontsize=12)

# Simülasyon butonu
if st.button('Simülasyonu Çalıştır 🚀'):

    both_teams_score = 0
    neither_scores = 0
    results = defaultdict(lambda: defaultdict(int))

    for _ in range(num_simulations):
        home_goals = np.random.poisson(xG_home)
        away_goals = np.random.poisson(xG_away)

        if home_goals > 0 and away_goals > 0:
            both_teams_score += 1
            both_teams_score_result = 'Var'
        else:
            neither_scores += 1
            both_teams_score_result = 'Yok'

        if home_goals > away_goals:
            match_result = 'MS1'
        elif home_goals < away_goals:
            match_result = 'MS2'
        else:
            match_result = 'MSX'

        results[match_result][both_teams_score_result] += 1

    both_teams_score_rate = (both_teams_score / num_simulations) * 100
    neither_scores_rate = (neither_scores / num_simulations) * 100

    df_both_teams_score = pd.DataFrame({
        'Durum': ['Karşılıklı Gol Var', 'Karşılıklı Gol Yok'],
        'Oran (%)': [both_teams_score_rate, neither_scores_rate]
    })

    df_results = pd.DataFrame([
        {'Match Result': match_result, 'KG Var': (results[match_result]['Var'] / num_simulations) * 100, 'KG Yok': (results[match_result]['Yok'] / num_simulations) * 100}
        for match_result in ['MS1', 'MSX', 'MS2']
    ])

    # Grafikler
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    bars = axs[0].bar(df_both_teams_score['Durum'], df_both_teams_score['Oran (%)'], color=['#66b3ff', '#ff9999'], edgecolor='black')
    axs[0].set_xlabel('Durum', fontsize=14)
    axs[0].set_ylabel('Oran (%)', fontsize=14)
    axs[0].set_title('Karşılıklı Gol Olasılıkları', fontsize=16)

    for bar in bars:
        yval = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f}%", ha='center', va='bottom', fontsize=12)

    bar_width = 0.4
    bar_positions1 = np.arange(len(df_results['Match Result']))
    bar_positions2 = bar_positions1 + bar_width

    bars1 = axs[1].bar(bar_positions1, df_results['KG Var'], bar_width, label='Karşılıklı Gol Var', color='#66b3ff', edgecolor='black')
    bars2 = axs[1].bar(bar_positions2, df_results['KG Yok'], bar_width, label='Karşılıklı Gol Yok', color='#ff9999', edgecolor='black')

    axs[1].set_xlabel('Maç Sonucu', fontsize=14)
    axs[1].set_ylabel('Olasılık (%)', fontsize=14)
    axs[1].set_title('Maç Sonucu ve Karşılıklı Gol Olasılıkları', fontsize=16)
    axs[1].set_xticks(bar_positions1 + bar_width / 2)
    axs[1].set_xticklabels(df_results['Match Result'])

    add_values(bars1, axs[1])
    add_values(bars2, axs[1])
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()

    # Streamlit ile grafik gösterimi
    st.pyplot(fig)


