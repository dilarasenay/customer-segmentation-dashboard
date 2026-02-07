import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Dosyayı oku
su_anki_dizin = os.path.dirname(os.path.abspath(__file__))
dosya_yolu = os.path.join(su_anki_dizin, '..', '..', 'data', 'processed', 'rfm_clustered.csv')
df = pd.read_csv(dosya_yolu)
print("Dosya başarıyla bulundu!")

# rakmaları isimlendirelim.
isimlendirme = {
    0: "Kayıp Müşteriler" ,    # Nerede 0 rakamı görse yanına "Kayıp Müşteriler" yazacaktır.
    1: "VIP/Şampiyonlar",
    2: "Yeni/Potansiyel",
    3: "Sadık Müşteriler" 
}
renk_sozlugu = {
    "Kayıp Müşteriler": "tomato",
    "VIP/Şampiyonlar": "gold",
    "Yeni/Potansiyel": "skyblue",
    "Sadık Müşteriler": "limegreen"
}
# --- 1. GRAFİK: PASTA --- (Segment Dağılımı)

# 'segment_ismi' adında yeni bir sütun ekleyelim
df['segment_ismi'] = df['cluster'].map(isimlendirme)   #.map() cluster sütunundaki rakamları isimlerle eşleştirmeye yarar.
hazir_veri = df.groupby(['cluster', 'segment_ismi']).size()  # örnek çıktı : 0 Kayıp Müşteriler 150  gruplandırdık yani 

plt.figure(figsize=(10, 7))    # bu sayılarla oynayarak grafiği daha geniş veya daha kare yapabiliriz.
pasta_renkleri = [renk_sozlugu[name] for clus, name in hazir_veri.index]
plt.pie(hazir_veri, labels=[name for clus, name in hazir_veri.index], autopct='%1.1f%%', startangle=90, colors=pasta_renkleri) 
#counts: Pastanın dilim boyutlarını belirleyen sayıdır.
#labels=counts.index: Her dilimin üzerine hangi segmentin isminin yazılacağını belirler.
#%1.1f demek, virgülden sonra sadece 1 basamak göster demektir.
#grafiğin daha dengeli durmasını sağlar.
plt.title('Müşteri Portföyümüzün Dağılımı', fontsize=15)

# --- 2.GRAFİK: GELİR (SÜTUN) --- (Toplam Gelir Payı)
gelir_verisi = df.groupby('segment_ismi')['monetary'].sum().sort_values(ascending=False)
#toplam gelirleri büyükten küçüğe doğru sıralar ve azalan sırada yapalım
plt.figure(figsize=(12, 6))
sns.barplot(x=gelir_verisi.index, y=gelir_verisi.values, palette=renk_sozlugu)
plt.title('Müşteri Segmentlerinin Finansal Ağırlık Dağılımı', fontsize=16, pad=20)
plt.xlabel('Müşteri Segmenti', fontsize=12)
plt.ylabel('Toplam Harcama Skoru (Standardize)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# --- 3. GRAFİK: RFM SAÇILIM (SCATTER ) --- (Müşteri Davranış Haritası)
plt.figure(figsize=(12, 8))
# sns.scatterplot ile noktaları çiziyoruz
sns.scatterplot(data=df, 
                x='recency', 
                y='frequency', 
                hue='segment_ismi',
                palette=renk_sozlugu, 
                s=100, # Nokta büyüklüğü
                alpha=0.6) # Üst üste binen noktaların görünmesi için hafif şeffaflık

plt.title('Müşteri Davranış Haritası', fontsize=16, pad=20)
plt.xlabel('Recency (Kaç Gün Önce Geldi?)', fontsize=12)
plt.ylabel('Frequency (Alışveriş Sıklığı)', fontsize=12)
plt.legend(title='Segmentler', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()

plt.grid(True, linestyle='--', alpha=0.5)

# ---4.GRAFİK: ORTALAMA METRİKLER --- (Her Grubun Puan Tablosu)
ortalama_metrikler = df.groupby('segment_ismi')[['recency', 'frequency', 'monetary']].mean()  ## 1. her segment için rfm değerlerinin ortalaması
# 2. Veriyi görselleştirmek için "uzun" (melt) formata çevirelim
ortalama_metrikler_melted = ortalama_metrikler.reset_index().melt(  # rfm sütunlarını tek bir sütunda toplayalım.
    id_vars='segment_ismi', 
    var_name='Metrik', 
    value_name='Ortalama Puan'
)
plt.figure(figsize=(14, 8))
sns.barplot(data=ortalama_metrikler_melted, 
            x='segment_ismi', 
            y='Ortalama Puan', 
            hue='Metrik',  # rfm yan yana sütunlar şeklinde olması için
            palette='coolwarm') # Metrikleri ayırmak için palet

plt.title('Segmentlerin Davranış Profili (Ortalama R-F-M Skorları)', fontsize=16, pad=20)
plt.xlabel('Müşteri Segmenti', fontsize=12)
plt.ylabel('Ortalama Standartlaştırılmış Puan', fontsize=12)
# 0 çizgisini (ortalamayı) belirgin bir siyah çizgiyle işaretleyelim
plt.axhline(0, color='black', linewidth=1.5, linestyle='-') 
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(title='RFM Metrikleri', loc='upper left')
plt.tight_layout()

# --- 5. GRAFİK: HARCAMA DAĞILIMI (BOXPLOT) ---  (Her segmentin içindeki harcama dağılımı)
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, 
            x='segment_ismi', 
            y='monetary', 
            palette=renk_sozlugu)

plt.title('Müşteri Segmentlerinin Harcama Dağılımı ve Aykırı Değerler', fontsize=16, pad=20)
plt.xlabel('Müşteri Segmenti', fontsize=12)
plt.ylabel('Harcama Skoru (Monetary)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)   #yatay çizgiler 
plt.tight_layout()  #taşmayı üst üste binmeyi engellemek içindir.
plt.show()

df.to_csv('dashboard_data.csv', index=False)