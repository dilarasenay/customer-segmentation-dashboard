import pandas as pd
import datetime as dt

# 1. Veriyi Yükleme
# Dosya yolunu senin klasör yapına göre ayarladık
df = pd.read_csv('data/raw/customers.csv')

# 2. Omnichannel (Online + Offline) Verilerin Birleştirilmesi
# Sıklık (Frequency) ve Parasal Değer (Monetary) için iki kanalı topluyoruz
# GH@07022026 Sıklık (Frequency) ve Parasal Değer (Monetary) için iki kanalı topluyoruz
df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_customer_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# 3. Tarih Verilerinin Dönüştürülmesi
# CSV'deki tarih sütunlarını pandas'ın anlayacağı 'datetime' formatına çeviriyoruz
date_columns = ["last_order_date", "first_order_date"]
for col in date_columns:
    df[col] = pd.to_datetime(df[col])

# 4. Analiz Tarihinin Belirlenmesi (Recency için)
# Veri setindeki en son alışveriş tarihinden 2 gün sonrasını "bugün" kabul ediyoruz
analysis_date = df["last_order_date"].max() + dt.timedelta(days=2)

# 5. RFM Metriklerinin Hesaplanması
rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).dt.days
rfm["frequency"] = df["total_order_num"]
rfm["monetary"] = df["total_customer_value"]

# 6. RFM Skorlarının Hesaplanması (1-5 Arası)
# qcut fonksiyonu veriyi küçükten büyüğe sıralayıp 5 eşit parçaya böler

# Recency: Yeni alışveriş yapan (düşük gün sayısı) daha değerlidir, bu yüzden etiketler ters (5,4,3,2,1)
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

# Frequency: Daha çok alışveriş yapan daha değerlidir (1,2,3,4,5)
# Not: Aynı değerden çok fazla varsa 'rank' metodu ile çakışmaları önlüyoruz
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

# Monetary: Daha çok bırakan daha değerlidir (1,2,3,4,5)
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# 7. RF Skorunun Oluşturulması
# Segmentasyon genellikle Recency ve Frequency üzerinden yapılır
rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

# 8. İşlenmiş Veriyi Kaydetme
rfm.to_csv('data/processed/customers_rfm.csv', index=False)

print("RFM Skorları başarıyla hesaplandı ve 'data/processed/customers_rfm.csv' adresine kaydedildi!")

# 9. Segmentlerin İsimlendirilmesi
# RF_SCORE üzerinden regex (düzenli ifadeler) kullanarak gruplandırma yapıyoruz
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

# Güncellenmiş veriyi tekrar kaydet
rfm.to_csv('data/processed/customers_rfm.csv', index=False)