import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title='Spotify Genre Classifier',
    page_icon=':musical_note:',
    layout='centered',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
    .main { background-color: #121212; color: white; }
    .stApp { background-color: #121212; }
    .stMetric { background-color: #1DB954; color: black; border-radius: 10px; padding: 10px; }
    .stSlider > div > div > div > div { background-color: #1DB954 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    pipeline = joblib.load('models/genre_pipeline.pkl')
    le = joblib.load('models/label_encoder.pkl')
    return pipeline, le

pipeline, le = load_model()

st.title('Spotify Genre Classifier')
st.markdown('Настрой аудио-параметры трека, и ML-модель предскажет его жанр!')

st.sidebar.header('Параметры трека')
def randomize_params():
    return {
        'danceability': np.random.uniform(0.2, 0.9),
        'energy': np.random.uniform(0.2, 0.9),
        'loudness': np.random.uniform(-20.0, 0.0),
        'speechiness': np.random.uniform(0.0, 0.5),
        'acousticness': np.random.uniform(0.0, 0.9),
        'instrumentalness': np.random.uniform(0.0, 0.9),
        'liveness': np.random.uniform(0.0, 0.5),
        'valence': np.random.uniform(0.1, 0.9),
        'tempo': np.random.uniform(70.0, 180.0),
        'duration_ms': np.random.randint(120000, 360000),
        'popularity': np.random.randint(10, 100)
    }

if st.sidebar.button('Случайный трек'):
    st.session_state.params = randomize_params()

params = st.session_state.get('params', {
    'danceability': 0.7, 'energy': 0.7, 'loudness': -5.0, 'speechiness': 0.05,
    'acousticness': 0.1, 'instrumentalness': 0.0, 'liveness': 0.1, 'valence': 0.6,
    'tempo': 120.0, 'duration_ms': 200000, 'popularity': 70
})

feature_configs = {
    'danceability': ('Танцевальность', 0.0, 1.0, 0.01),
    'energy': ('Энергичность', 0.0, 1.0, 0.01),
    'loudness': ('Громкость', -60.0, 0.0, 0.1),
    'speechiness': ('Речевость', 0.0, 1.0, 0.01),
    'acousticness': ('Акустичность', 0.0, 1.0, 0.01),
    'instrumentalness': ('Инструментальность', 0.0, 1.0, 0.01),
    'liveness': ('Живое выступление', 0.0, 1.0, 0.01),
    'valence': ('Позитивность', 0.0, 1.0, 0.01),
    'tempo': ('Темп', 50.0, 200.0, 1.0),
    'duration_ms': ('Длительность (мс)', 0, 600000, 1000),
    'popularity': ('Популярность', 0, 100, 1)
}

input_data = {}
for feat, (label, min_val, max_val, step) in feature_configs.items():
    input_data[feat] = st.sidebar.slider(label, min_value=min_val, max_value=max_val, value=params[feat], step=step)


st.session_state.params = input_data

st.markdown("---")
if st.button("Предсказать жанр", type="primary", use_container_width=True):
    with st.spinner("Анализируем аудио-сигнал..."):
        expected_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                         'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'popularity']
        
        input_df = pd.DataFrame([input_data])[expected_cols]
        
        probas = pipeline.predict_proba(input_df)[0]
        
        top_3_idx = np.argsort(probas)[::-1][:3]
        top_3 = [(le.inverse_transform([i])[0], round(probas[i] * 100, 1)) for i in top_3_idx]
        
        st.success(f"Модель уверена, что это: **{top_3[0][0].upper()}**")
        st.markdown("### Топ-3 жанра:")
        for i, (genre, prob) in enumerate(top_3):
            st.progress(prob / 100)
            st.caption(f"**{i+1}.** {genre} — {prob}%")
            
        st.markdown("---")
        st.info("*Модель использует алгоритм машинного обучения (Pipeline: StandardScaler + Tuned Model), обученный на 100K+ треках Spotify.*")