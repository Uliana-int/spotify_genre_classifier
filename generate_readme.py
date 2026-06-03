import joblib
import os

def generate_readme():
    try:
        pipeline = joblib.load('models/genre_pipeline.pkl')
        le = joblib.load('models/label_encoder.pkl')
        model_name = type(pipeline.named_steps['model']).__name__
        n_classes = len(le.classes_)
    except FileNotFoundError:
        model_name = "Tuned Model (Pipeline)"
        n_classes = 10

    readme_content = f"""# Spotify Genre Classifier

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

Классификация музыкальных жанров на основе аудио-фич с использованием классического ML. Проект включает полный цикл разработки: от EDA и борьбы с дисбалансом до интерпретации (SHAP) и веб-деплоя.

## О данных
- **Источник**: [Spotify Tracks Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **Объем**: ~114 000 треков, отфильтровано до топ-10 самых частых жанров.
- **Признаки (11)**: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, popularity.
- **Целевая переменная**: track_genre ({n_classes} классов).

## Как запустить проект

### 1. Локальный запуск (Исследование и обучение)
1. Клонируй репозиторий: `git clone https://github.com/Uliana-int/spotify_genre_classifier.git`
2. Перейди в папку: `cd spotify_genre_classifier`
3. Создай окружение: `python -m venv venv` и активируй его (`source venv/bin/activate` или `venv\\Scripts\\activate` для Windows).
4. Установи зависимости: `pip install -r requirements.txt`
5. Запусти Jupyter: `jupyter notebook`

Выполняй ноутбуки строго по порядку (Run All):
- `01_eda.ipynb` — Исследование данных, очистка, визуализация.
- `02_modeling.ipynb` — Обучение, 5-Fold CV, GridSearch, анализ дисбаланса (SMOTE), сохранение sklearn.Pipeline.
- `03_shap_analysis.ipynb` — Глобальная и локальная интерпретация предсказаний (SHAP).

### 2. Запуск веб-интерфейса (Streamlit)
Выполни в терминале: `streamlit run app.py`
Откроется браузер с интерактивным предсказанием жанра по аудио-параметрам в реальном времени.

## Результаты и архитектура
- **Лучшая модель**: {model_name} (с подобранными гиперпараметрами через GridSearchCV).
- **Пайплайн**: Данные проходят через StandardScaler и модель в едином sklearn.Pipeline. Это исключает data leakage и позволяет деплоить модель одним файлом.
- **Дисбаланс**: Проведен анализ и сравнение подходов (Baseline vs SMOTE vs Class Weight).
- **Интерпретируемость**: Использован SHAP (TreeExplainer) для объяснения глобальной важности признаков и локальных предсказаний.

## Структура проекта
- `data/` — Исходные и очищенные данные.
- `models/` — Сохраненные артефакты (genre_pipeline.pkl, label_encoder.pkl).
- `01_eda.ipynb` — Разведочный анализ.
- `02_modeling.ipynb` — Обучение, CV и пайплайн.
- `03_shap_analysis.ipynb` — SHAP-интерпретация.
- `app.py` — Код веб-приложения Streamlit.
- `requirements.txt` — Зависимости проекта.

## Автор
**Uliana-int** | [GitHub](https://github.com/Uliana-int)

---
*Проект создан для демонстрации навыков End-to-End Machine Learning.*
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("README.md успешно сгенерирован!")

if __name__ == "__main__":
    generate_readme()