# RFM Clustering API

Сервис для кластеризации клиентов интернет-магазина на основе RFM (Recency, Frequency, Monetary) признаков. Он развёрнут как API с использованием FastAPI и запускается через Docker.

---

## Состав

- `Model.py` — скрипт с минимальной предобработкой данных (удаляет записи с без id пользователя, с возвратом, нулевыми или отрицательными значениями), логикой RFM-анализа, кластеризации и построения графиков
- `main.py` — FastAPI-приложение, которое запускает API
- `requirements.txt` — список зависимостей
- `Dockerfile`, `docker-compose.yml` — всё необходимое для контейнеризации
- `example.csv` — пример входного файла
- `output/` — сюда сохраняются итоговые PNG-графики

---

## Как запустить

1. Откройте терминал в директории проекта.

2. Выполните команду:

```
docker-compose up --build
```

3. После запуска, API будет доступен по адресу:

```
http://localhost:8000/docs
```

В этом интерфейсе можно загрузить CSV-файл и получить результат кластеризации.

---

## Как пользоваться

- Эндпоинт: `POST /analyze`
- Необходимо загрузить CSV-файл со следующими колонками:
  - `CustomerID`
  - `InvoiceNo`
  - `InvoiceDate`
  - `Quantity`
  - `UnitPrice`

Пример файла уже приложен (`online_retail.csv`).

---

## Что вернётся

API выполнит RFM-анализ, кластеризацию и визуализацию. В ответе будет:

```json
{
  "silhouette_score": 0.6520,
  "plots": {
    "2d": "/output/cluster_plot_with_centroids.png",
    "3d": "/output/cluster_plot_3d_hull.png"
  }
}
```

Графики сохраняются в папке `output/`.

---

## Используемые технологии

- Python 3.10
- FastAPI
- Uvicorn
- Pandas, Numpy, Scikit-learn, Scipy
- Matplotlib
