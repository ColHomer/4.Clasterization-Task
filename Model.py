import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import ConvexHull


def load_data(file_path):
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    df_clean = df.dropna(subset=["CustomerID"])
    df_clean = df_clean[~df_clean["InvoiceNo"].astype(str).str.startswith("C")]
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
    df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["UnitPrice"]
    return df_clean


def preprocess_rfm(df_clean):
    reference_date = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df_clean.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (reference_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    }).reset_index()
    rfm.columns = ["CustomerID", "Давность (Recency)", "Частота (Frequency)", "Затраты (Monetary)"]
    return rfm


def run_kmeans_clustering(rfm, k=4):
    scaler = StandardScaler()
    rfm_scaled_values = scaler.fit_transform(rfm[["Давность (Recency)", "Частота (Frequency)", "Затраты (Monetary)"]])
    rfm_scaled = pd.DataFrame(rfm_scaled_values, columns=["Давность", "Частота", "Затраты"])
    rfm_scaled["CustomerID"] = rfm["CustomerID"].values

    model = KMeans(n_clusters=k, random_state=42)
    rfm_scaled["Кластер"] = model.fit_predict(rfm_scaled[["Давность", "Частота", "Затраты"]])
    silhouette = silhouette_score(rfm_scaled[["Давность", "Частота", "Затраты"]], rfm_scaled["Кластер"])
    rfm_combined = pd.concat([rfm, rfm_scaled["Кластер"]], axis=1)
    return rfm_combined, silhouette


def plot_clusters(rfm_combined, output_dir, start_date, end_date):
    os.makedirs(output_dir, exist_ok=True)
    cluster_labels = {
        0: "Лояльные клиенты: покупают часто, недавно, тратят много",
        1: "Неактивные клиенты: покупали редко и давно",
        2: "VIP-клиенты: высокая частота и траты",
        3: "Потенциальные клиенты: средняя активность"
    }
    centroids = rfm_combined.groupby("Кластер")[["Давность (Recency)", "Частота (Frequency)"]].mean().values
    plt.figure(figsize=(10, 6))
    for cluster in sorted(rfm_combined["Кластер"].unique()):
        data = rfm_combined[rfm_combined["Кластер"] == cluster]
        label = f"Кластер {cluster}: {cluster_labels.get(cluster, '')}"
        plt.scatter(data["Давность (Recency)"], data["Частота (Frequency)"], label=label)
    for idx, (x, y) in enumerate(centroids):
        plt.scatter(x, y, c='black', s=200, alpha=0.6, marker='X')
        plt.text(x, y + 1.5, f"Центроид {idx}", fontsize=9, ha='center', color='black')
    plt.xlabel("Давность (в днях)")
    plt.ylabel("Частота покупок")
    plt.title(f"Кластеры клиентов с центроидами\nПериод: {start_date} — {end_date}")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_plot_with_centroids.png"))
    plt.close()


    # 3D Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    colors = cm.get_cmap('tab10')
    markers = ['o', '^', 's', 'x']
    for cluster in sorted(rfm_combined["Кластер"].unique()):
        group = rfm_combined[rfm_combined["Кластер"] == cluster]
        points = group[["Давность (Recency)", "Частота (Frequency)", "Затраты (Monetary)"]].values
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            label=f"Кластер {cluster}: {cluster_labels.get(cluster, '')}",
            s=10, alpha=0.6,
            marker=markers[cluster % len(markers)],
            color=colors(cluster)
        )
        if len(points) >= 4:
            try:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    simp = points[simplex]
                    ax.plot_trisurf(simp[:, 0], simp[:, 1], simp[:, 2], color=colors(cluster), alpha=0.1)
            except Exception as e:
                print(f"Ошибка построения оболочки для кластера {cluster}: {e}")
    centroids_3d = rfm_combined.groupby("Кластер")[["Давность (Recency)", "Частота (Frequency)", "Затраты (Monetary)"]].mean().values
    for idx, (x, y, z) in enumerate(centroids_3d):
        ax.scatter(x, y, z, c='black', s=100, marker='X')
        ax.text(x, y, z + 1000, f"Центроид {idx}", color='black', fontsize=8, ha='center')
    ax.set_xlabel("Давность (в днях)")
    ax.set_ylabel("Частота покупок")
    ax.set_zlabel("Затраты")
    ax.set_title(f"3D Кластеры клиентов по признакам RFM\nПериод: {start_date} — {end_date}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_plot_3d_hull.png"))
    plt.show()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "data", "online_retail.csv")
    output_dir = os.path.join(base_dir, "output")
    df_clean = load_data(file_path)
    rfm = preprocess_rfm(df_clean)
    start_date = df_clean["InvoiceDate"].min().date()
    end_date = df_clean["InvoiceDate"].max().date()
    rfm_combined, score = run_kmeans_clustering(rfm)
    print(f"Silhouette Score: {score:.4f}")
    plot_clusters(rfm_combined, output_dir, start_date, end_date)
