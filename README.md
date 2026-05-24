# Spotify Genre Classifier

Классификация музыкальных жанров на основе аудио-фич с использованием классического ML.

## Данные
- Источник: [Spotify Tracks Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- Фильтрация: топ-10 жанров для баланса классов
- Признаки: `danceability`, `energy`, `valence`, `tempo`, `loudness`, `acousticness` и др.

## Как запустить
1. Установи зависимости: `pip install -r requirements.txt`
2. Скачай `spotify_tracks.csv` с Kaggle и положи в `data/raw/`
3. Запусти ноутбуки по порядку:
   - `01_eda.ipynb` — исследование, очистка, сохранение `spotify_clean.csv`
   - `02_modeling.ipynb` — обучение, CV, GridSearch, визуализация

## Результаты
| Модель | Accuracy (5-Fold CV) | F1-Macro (CV)
LogReg: Accuracy = 0.6107 | F1 = 0.6062 
RF: Accuracy = 0.7733 | F1 = 0.7712 
KNN: Accuracy = 0.6838 | F1 = 0.6818
GB: Accuracy = 0.7642| F1 = 0.7637

## Стек
`Python 3.10+` | `scikit-learn` | `pandas` | `seaborn` | `matplotlib`

## Автор
Uliana-int | [GitHub](https://github.com/Uliana-int)