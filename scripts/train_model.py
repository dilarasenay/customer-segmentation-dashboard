# Veri işleme için pandas kütüphanesini ekledim
import pandas as pd

# KMeans algoritmasını kullanmak için sklearn içinden import ettim
from sklearn.cluster import KMeans

# Önceden feature scaling yapılmış RFM verisinin bulunduğu dosya yolu
data_path = "data/processed/rfm_scaled.csv"

# CSV dosyasını okuyarak dataframe'e aktardım
df = pd.read_csv(data_path)

# Kontrol amaçlı verinin başarıyla okunup okunmadığını terminale yazdırıyorum
print("Veri başarıyla okundu ")

# Model eğitiminde sadece RFM feature'larını kullanacağım için ilgili kolonları seçtim
X = df[["recency", "frequency", "monetary"]]

# 4 segment oluşturacak şekilde KMeans modelini oluşturdum
# random_state=42 verilebilirlik (reproducibility) için eklendi
kmeans = KMeans(n_clusters=4, random_state=42)

# Seçtiğim RFM feature'ları ile modeli eğittim
kmeans.fit(X)

# Modelin başarıyla eğitildiğini görmek için çıktı veriyorum
print("KMeans modeli eğitildi ")

# Eğitilmiş model ile her müşterinin ait olduğu cluster'ı tahmin ettim
df["cluster"] = kmeans.predict(X)

# Cluster kolonunun başarıyla eklendiğini kontrol amaçlı yazdırıyorum
print("Cluster kolonu eklendi ")
print(df.head())

# Modeli kaydetmek için joblib kütüphanesini ekledim
import joblib

# Eğitilmiş KMeans modelini daha sonra dashboard içinde kullanabilmek için kaydedeceğim dosya yolu
model_path = "models/kmeans_model.pkl"

# Eğitilmiş modeli .pkl formatında kaydettim
joblib.dump(kmeans, model_path)

# Modelin başarıyla kaydedildiğini belirten çıktı
print("KMeans modeli kaydedildi ")

# Cluster bilgisi eklenmiş son veriyi kaydedeceğim dosya yolu
output_path = "data/processed/rfm_clustered.csv"

# Cluster eklenmiş dataframe'i CSV olarak kaydettim
df.to_csv(output_path, index=False)

# Clustered verinin başarıyla kaydedildiğini belirten çıktı
print("Clustered veri kaydedildi ")


print("\nCluster Ortalamaları:")
print(df.groupby("cluster")[["recency", "frequency", "monetary"]].mean())

cluster_map = {
    0: "Lost Customers",
    1: "VIP Customers",
    2: "New Customers",
    3: "Loyal Customers"
}

df["segment"] = df["cluster"].map(cluster_map)

# clusterları iş anlamlı segmentlere dönüştürdüm
cluster_map = {
    0: "Lost Customers",
    1: "VIP Customers",
    2: "New Customers",
    3: "Loyal Customers"
}

df["segment"] = df["cluster"].map(cluster_map)

print("Segment kolonu eklendi ")
