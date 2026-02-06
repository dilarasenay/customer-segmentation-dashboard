import pandas as pd
import datetime as dt

# 1. Veriyi Oku
df = pd.read_csv("data/raw/customers.csv")

# 2. Ön Hazırlık: Online ve Offline verileri birleştir
df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_customer_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 3. Tarih Dönüşümleri
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

# 4. Analiz Tarihi (Referans Günü)
analysis_date = df["last_order_date"].max() + dt.timedelta(days=2)

# 5. RFM Metriklerinin Hesaplanması
rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).dt.days
rfm["frequency"] = df["total_order_num"]
rfm["monetary"] = df["total_customer_value"]

# 6. RFM Skorlaması (1-5 Puan Arası)
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# RFM Skorunu birleştirme
rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

# 7. Sonucu Kaydet (Feature Scaling yapacak arkadaşın için)
rfm.to_csv("data/processed/customers_rfm.csv", index=False)

print("İşlem Başarılı! RFM tablosu 'data/processed/customers_rfm.csv' olarak kaydedildi.")
print(rfm.head())