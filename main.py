from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import os
from Model import load_data, preprocess_rfm, run_kmeans_clustering, plot_clusters
from io import StringIO

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"status": "ok", "message": "RFM Clustering API is working"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # Чтение CSV-файла
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        # Применение модели
        df_clean = load_data_from_df(df)
        rfm = preprocess_rfm(df_clean)
        rfm_combined, score = run_kmeans_clustering(rfm)

        start_date = df_clean["InvoiceDate"].min().date()
        end_date = df_clean["InvoiceDate"].max().date()
        plot_clusters(rfm_combined, OUTPUT_DIR, start_date, end_date)

        return JSONResponse(content={
            "silhouette_score": round(score, 4),
            "plots": {
                "2d": "/output/cluster_plot_with_centroids.png",
                "3d": "/output/cluster_plot_3d_hull.png"
            }
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


def load_data_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.dropna(subset=["CustomerID"])
    df_clean = df_clean[~df_clean["InvoiceNo"].astype(str).str.startswith("C")]
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
    df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["UnitPrice"]
    return df_clean
