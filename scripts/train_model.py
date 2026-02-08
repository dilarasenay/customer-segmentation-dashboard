# ======================================================
# CUSTOMER SEGMENTATION - MODEL TRAINING PIPELINE
# Bu scriptte:
# - scaled RFM verisi ile KMeans modeli eğitiyorum
# - oluşan cluster'ları original RFM verisine yazıyorum
# - modeli .pkl olarak kaydediyorum
# - dashboard için clustered veri export ediyorum
# ======================================================


# -------------------------------
# Gerekli kütüphaneleri ekledim
# -------------------------------
import pandas as pd
from sklearn.cluster import KMeans
import joblib


# --------------------------------------------------
# 1️⃣ Veri setlerini okudum
# scaled veri → model eğitimi için
# original veri → dashboard için gerçek değerler
# --------------------------------------------------
df_scaled = pd.read_csv("data/processed/rfm_scaled.csv")
df_original = pd.read_csv("data/processed/customers_rfm.csv")

print("Veriler başarıyla okundu")


# --------------------------------------------------
# 2️⃣ Model eğitiminde kullanacağım feature'ları seçtim
# sadece RFM kolonlarını kullanıyorum
# --------------------------------------------------
X = df_scaled[["recency", "frequency", "monetary"]]


# --------------------------------------------------
# 3️⃣ KMeans modelini oluşturdum
# 4 cluster → 4 müşteri segmenti
# random_state → aynı sonucu üretmesi için
# --------------------------------------------------
kmeans = KMeans(n_clusters=4, random_state=42)


# --------------------------------------------------
# 4️⃣ Modeli eğittim
# --------------------------------------------------
kmeans.fit(X)

print("KMeans modeli eğitildi")


# --------------------------------------------------
# 5️⃣ Cluster tahminlerini aldım
# scaled veri ile tahmin yapıyorum
# sonucu original dataframe'e yazıyorum
# çünkü dashboard gerçek değerleri gösterecek
# --------------------------------------------------
df_original["cluster"] = kmeans.predict(X)

print("Cluster kolonu original veriye eklendi")
print(df_original.head())


# --------------------------------------------------
# 6️⃣ Eğitilmiş modeli kaydettim
# dashboard veya API tarafında tekrar kullanılacak
# --------------------------------------------------
model_path = "models/kmeans_model.pkl"
joblib.dump(kmeans, model_path)

print("KMeans modeli kaydedildi")


# --------------------------------------------------
# 7️⃣ Cluster ortalamalarını analiz etmek için
# groupby ile ortalama RFM değerlerini aldım
# --------------------------------------------------
print("\nCluster Ortalamaları:")
print(
    df_original.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
)


# --------------------------------------------------
# 8️⃣ Cluster numaralarını iş anlamlı segmentlere çevirdim
# --------------------------------------------------
cluster_map = {
    0: "Lost Customers",
    1: "VIP Customers",
    2: "New Customers",
    3: "Loyal Customers"
}

df_original["segment"] = df_original["cluster"].map(cluster_map)

print("Segment kolonu eklendi")


# --------------------------------------------------
# 9️⃣ Dashboard için clustered veriyi export ettim
# --------------------------------------------------
output_path = "data/processed/rfm_clustered.csv"
df_original.to_csv(output_path, index=False)

print("Clustered veri kaydedildi")
print("Pipeline tamamlandı ")
