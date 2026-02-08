import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---VERİ YÜKLEME ---
try:
    # Colab veya yerel dizin için dosya okuma
    df = pd.read_csv('rfm_clustered.csv')
except:
    su_anki_dizin = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    dosya_yolu = os.path.join(su_anki_dizin, '..', '..', 'data', 'processed', 'rfm_clustered.csv')
    df = pd.read_csv(dosya_yolu)

print("Dosya başarıyla yüklendi!")

# --- İSİMLENDİRME VE RENKLER ---
isimlendirme = {
    0: "Kayıp Müşteriler",
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

df['segment_ismi'] = df['cluster'].map(isimlendirme)

# --- GRAFİK: PASTA (Segment Dağılımı) ---
hazir_veri = df.groupby(['cluster', 'segment_ismi']).size()
plt.figure(figsize=(10, 7))
pasta_renkleri = [renk_sozlugu[name] for clus, name in hazir_veri.index]
plt.pie(hazir_veri, labels=[name for clus, name in hazir_veri.index], autopct='%1.1f%%', startangle=90, colors=pasta_renkleri)
plt.title('Müşteri Portföyümüzün Dağılımı', fontsize=15)

# --- GRAFİK: TOPLAM GELİR (Sütun) ---
gelir_verisi = df.groupby('segment_ismi')['monetary'].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=gelir_verisi.index, y=gelir_verisi.values, palette=renk_sozlugu)
plt.title('Müşteri Segmentlerinin Finansal Ağırlık Dağılımı', fontsize=16, pad=20)
plt.xlabel('Müşteri Segmenti', fontsize=12)
plt.ylabel('Toplam Harcama (Monetary)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.ticklabel_format(style='plain', axis='y') 
plt.tight_layout()

# --- GRAFİK: SIKLIK vs HARCAMA (Saçılım) ---
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, 
                x='frequency', 
                y='monetary', 
                hue='segment_ismi', 
                palette=renk_sozlugu, 
                s=120, 
                alpha=0.7, 
                edgecolor='white')

plt.title('Müşteri Değer Haritası: Sıklık vs Harcama', fontsize=16, pad=20)
plt.xlabel('Alışveriş Sıklığı (Frequency)', fontsize=12)
plt.ylabel('Toplam Harcama (Monetary)', fontsize=12)
plt.legend(title='Segmentler', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# --- GRAFİK: ORTALAMA METRİKLER ---
ortalama_metrikler = df.groupby('segment_ismi')[['recency', 'frequency', 'monetary']].mean()
ortalama_metrikler_melted = ortalama_metrikler.reset_index().melt(id_vars='segment_ismi', var_name='Metrik', value_name='Ortalama')
plt.figure(figsize=(14, 8))
sns.barplot(data=ortalama_metrikler_melted, x='segment_ismi', y='Ortalama', hue='Metrik', palette='coolwarm')
plt.title('Segmentlerin Ortalama Davranış Profili', fontsize=16, pad=20)
plt.axhline(0, color='black', linewidth=1.5)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.yscale('log') # Y eksenini logaritmik yapar
plt.tight_layout()

# --- 7. GRAFİK: HARCAMA DAĞILIMI (Boxplot) ---
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='segment_ismi', y='monetary', palette=renk_sozlugu)
plt.title('Segmentlerin Harcama Dağılımı ve Aykırı Değerler', fontsize=16, pad=20)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.yscale('log') # Y eksenini logaritmik yapar
plt.tight_layout()

plt.show()