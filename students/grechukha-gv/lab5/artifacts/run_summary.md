# Сводка запуска lab5

Датасет: 20 Newsgroups из `sklearn.datasets.fetch_20newsgroups`, subset=train.
Категории: comp.graphics, rec.sport.baseball, sci.med, talk.politics.misc.
Документы: 800, термины после TF-IDF: 320, ненулевых значений: 11922.
Плотность матрицы: 0.0466. Скрытых TF-IDF значений для теста: 2200.
Баланс категорий: comp.graphics: 207, rec.sport.baseball: 223, sci.med: 201, talk.politics.misc: 169.
Примеры терминов: 00, 000, 10, 100, 11, 12, 14, 15, 16, 17, 1992, 1993, 20, 24, 30.

## Результаты

| Модель | RMSE | NDCG@10 | Fit time, sec | Iterations/components |
|--------|------|---------|---------------|-----------------------|
| Собственный SLIM | 0.2592 | 0.1059 | 0.027 | 6 |
| sklearn ElasticNet SLIM | 0.2592 | 0.1057 | 0.536 | 245 |
| Собственная LSA | 0.2444 | 0.0811 | 0.065 | 45 |
| sklearn TruncatedSVD | 0.2444 | 0.0838 | 0.031 | 45 |
