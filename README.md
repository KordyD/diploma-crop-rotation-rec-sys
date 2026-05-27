# Diploma Crop Rotation

## Структура Репозитория

- `notebooks/` - пайплайн ноутбуков и хелперы.
- `dataset/` - локальные входные данные.
- `artifacts/` - локальные сгенерированные датасеты, чекпоинты, модели, метрики, графики, демо.
- `docs/` - хранит дополнительные документационные источники.

## Notebook Pipeline

1. `01_data_analysis_and_prep.ipynb` - загрузка данных, mapping CDL-кодов, фильтрация классов.
2. `02_window_dataset_and_feature_table.ipynb` - построение оконного датасета `history_1..history_3 -> target`.
3. `03_baselines_and_split.ipynb` - сплит по `CSBID` и бейзлайны.
4. `04_catboost_baseline.ipynb` - catboost.
5. `05_catboost_improved.ipynb` - экспериментальная ветка с моделью.
6. `06_catboost_plus_graph.ipynb` - оценка CatBoost + transition graph hybrid.
7. `07_recommendation_confidence.ipynb` - top-k рекомендации, confidence, calibration.
8. `08_markov_recommendation_eval.ipynb` - Markov recommendation baseline и сравнение.
9. `09_knowledge_graph_explainability.ipynb` - граф знаний и объяснимость признаков.
10. `10_spatial_explainable_map.ipynb` - демо.
