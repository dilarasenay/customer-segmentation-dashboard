import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib # Modelleri okumak için 
import numpy as np 

# --- SABİT AYARLAR ---
ISIMLER = { 0: "Kayıp Müşteriler", 1: "VIP / Şampiyonlar", 2: "Yeni / Potansiyel", 3: "Sadık Müşteriler" }

app = Flask(__name__, 
            template_folder='app/templates', 
            static_folder='app/static')
# ==========================================
# 1. HELPER FUNCTIONS (YARDIMCI FONKSİYONLAR)
# ==========================================

def get_data_path(filename):
    """
    Proje dizin yapısına göre dinamik dosya yolu oluşturur.
    
    Amaç: İşletim sistemi fark etmeksizin (Windows/Mac/Linux) 
    'data/processed' klasörü altındaki dosyalara hatasız erişim sağlamak.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'data', 'processed', filename)


# ==========================================
# 2. DATA MODULES (VERİ MODÜLLERİ)
# ==========================================

def get_rfm_data():
    """
    KPI Kartları ve Grafikler için veri hazırlar.
    GÜNCELLEME: Profil grafiği (Sağdaki) artık ölçek sorunu olmaması için
    4 ana kümeye (Cluster) göre hesaplanıyor.
    """
    rfm_path = get_data_path('customers_rfm.csv')
    cluster_path = get_data_path('rfm_clustered.csv') # Küme verisini de okuyacağız
    
    try:
        # 1. Ana Veriyi Oku (KPI ve Ciro için)
        df = pd.read_csv(rfm_path)
        df.columns = df.columns.str.lower()
        
        # --- KPI Hesaplamaları ---
        total_customers = len(df)
        total_revenue = df['monetary'].sum() if 'monetary' in df.columns else 0
        revenue_formatted = f"{total_revenue:,.0f} ₺"

        leading_segment = "Belirsiz"
        if 'segment' in df.columns:
            leading_segment = df['segment'].value_counts().idxmax().replace('_', ' ').title()
        
        # --- Segment Dağılımı (Mevcut) ---
        chart_labels = []
        chart_values = []
        if 'segment' in df.columns:
            counts = df['segment'].value_counts()
            chart_labels = [l.replace('_', ' ').title() for l in counts.index]
            chart_values = counts.values.tolist()

        # --- EKLENEN KISIM: Ciro Dağılımı ---
        ciro_etiketleri = []
        ciro_verileri = []
        
        if 'segment' in df.columns and 'monetary' in df.columns:
            # Segmentlere göre parayı topla ve sırala
            gelir_grubu = df.groupby('segment')['monetary'].sum().sort_values(ascending=False)
            
            # Etiketleri düzelt (loyal_customers -> Loyal Customers)
            ciro_etiketleri = [str(x).replace('_', ' ').title() for x in gelir_grubu.index]
            ciro_verileri = gelir_grubu.values.tolist()

        # --- 4. YENİ: Segment Profilleri (4 ANA KÜME İÇİN) ---
        # Burayı rfm_clustered.csv'den alıyoruz ki sadece 4 tane olsun.
        avg_data = {"categories": [], "recency": [], "frequency": []}
        
        try:
            df_cl = pd.read_csv(cluster_path)
            df_cl.columns = df_cl.columns.str.lower()
            
            # İsimlendirme Sözlüğü (Senin standardın)
            isimler = {
                0: "Kayıp Müşteriler",
                1: "VIP / Şampiyonlar",
                2: "Yeni / Potansiyel",
                3: "Sadık Müşteriler"
            }
            
            # Kümeleri isimlendir
            df_cl['grup_adi'] = df_cl['cluster'].map(isimler).fillna("Diğer")
            
            # Ortalamaları al
            means = df_cl.groupby('grup_adi')[['recency', 'frequency']].mean().round(1)
            
            avg_data = {
                "categories": means.index.tolist(),
                "recency": means['recency'].tolist(),    # Çizgi Grafik (Sağ Eksen)
                "frequency": means['frequency'].tolist() # Sütun Grafik (Sol Eksen)
            }
            
            

        except Exception as e:
            print(f"Cluster verisi okunamadı, eskiye dönülüyor: {e}")

        # --- EKLENEN KISIM: Box Plot (Harcama Dağılımı) ---
        boxplot_verisi = []
        if 'segment' in df.columns and 'monetary' in df.columns:
            for seg in df['segment'].unique():
                seg_data = df[df['segment'] == seg]['monetary']
                
                # İstatistikleri Çıkar (Aykırı değerleri biraz tıraşlıyoruz ki kutu görünsün)
                # Bıyıklar: %5 (Alt) ve %95 (Üst) sınırları
                # Kutu: %25 (Q1) ve %75 (Q3) sınırları
                boxplot_verisi.append({
                    'x': str(seg).replace('_', ' ').title(),
                    'y': [
                        seg_data.quantile(0.05), # Min (Alt Bıyık)
                        seg_data.quantile(0.25), # Q1 (Kutu Altı)
                        seg_data.median(),       # Medyan (Çizgi)
                        seg_data.quantile(0.75), # Q3 (Kutu Üstü)
                        seg_data.quantile(0.95)  # Max (Üst Bıyık)
                    ]
                })

        # --- YENİ: 4'lü Segment Pasta Grafiği İçin Veri ---
        # Burası senin dosyanın içinde olmayan kısım, bunu ekliyoruz.
        pie_labels = []
        pie_values = []
        
        # rfm_clustered.csv dosyasını okuyoruz (4 Küme burada var)
        cluster_path = get_data_path('rfm_clustered.csv')
        
        try:
            if os.path.exists(cluster_path):
                df_cl = pd.read_csv(cluster_path)
                
                # İsimlendirme Sözlüğü (Renklerin karışmaması için)
                isimler = {0: "Kayıp Müşteriler", 1: "VIP / Şampiyonlar", 2: "Yeni / Potansiyel", 3: "Sadık Müşteriler"}
                
                # Cluster numarasına göre (0, 1, 2, 3) gruplayıp sayıyoruz
                counts = df_cl.groupby('cluster').size()
                
                # Etiketleri ve sayıları listeye çeviriyoruz
                pie_labels = [isimler.get(i, f"Küme {i}") for i in counts.index]
                pie_values = counts.values.tolist()
        except Exception as e:
            print(f"Pasta Grafik Hatası: {e}")        

        return {
            "sayi": total_customers,
            "skor": revenue_formatted,
            "isim": leading_segment,
            "dagilim_etiketleri": chart_labels,
            "dagilim_verileri": chart_values,
            "ciro_etiketleri": ciro_etiketleri,
            "ciro_verileri": ciro_verileri,
            "profil_verileri": avg_data,
            "boxplot_verisi": boxplot_verisi,
            "pasta_etiketleri": pie_labels,
            "pasta_verileri": pie_values
        }

    except Exception as e:
        print(f"KRİTİK HATA (RFM Modülü): {e}")
        return None

def get_kmeans_data():
    """
    K-Means verisini hazırlar.
    1. Gerçek (Raw) verileri kullanır.
    2. VIP'leri (Outlier) görsel netlik için filtreler.
    3. Kümeleri isimlendirir ve standart renklerini atar.
    """
    cluster_path = get_data_path('rfm_clustered.csv')
    raw_path = get_data_path('customers_rfm.csv')
    
    # --- 1. TANIMLAMALAR (İSİM ve RENK) ---
    # Bu kısım visualization.py ile aynı olmalı ki tutarlılık sağlansın.
    
    # İsimlendirme Sözlüğü (Cluster ID -> Anlamlı İsim)
    isimlendirme = {
        0: "Kayıp Müşteriler",    # Riskli/Kötü durum
        1: "VIP / Şampiyonlar",   # En iyiler
        2: "Yeni / Potansiyel",   # Gelişime açık
        3: "Sadık Müşteriler"     # İstikrarlı
    }

    # Renk Sözlüğü (Cluster ID -> Hex Kodu veya Renk İsmi)
    # Renkleri segmentin ruhuna uygun seçtik.
    renk_sozlugu = {
        0: "#FF6347",  # Tomato (Kırmızımsı - Tehlike)
        1: "#FFD700",  # Gold (Altın - Şampiyon)
        2: "#87CEEB",  # SkyBlue (Mavi - Yeni/Umut)
        3: "#32CD32"   # LimeGreen (Yeşil - Güvenli/Sadık)
    }

    try:
        # --- 2. VERİ OKUMA VE BİRLEŞTİRME ---
        df_cluster = pd.read_csv(cluster_path)
        df_raw = pd.read_csv(raw_path)
        
        df_cluster.columns = df_cluster.columns.str.lower()
        df_raw.columns = df_raw.columns.str.lower()

        # customer_id üzerinden gerçek veri ile küme bilgisini birleştir
        cols_to_use = ['customer_id', 'cluster']
        df_merged = pd.merge(df_raw, df_cluster[cols_to_use], on='customer_id', how='inner')

        # --- 3. OUTLIER (BALİNA) TEMİZLİĞİ 🧹 ---
        # Grafiğin sıkışmasını önlemek için en çok harcayan %5'i gizle.
        esik_deger = df_merged['monetary'].quantile(0.98)
        df_filtered = df_merged[df_merged['monetary'] < esik_deger]
        
        print(f"📊 Scatter Data: {len(df_merged)} -> {len(df_filtered)} nokta (VIP'ler filtrelendi)")

        # --- 4. VERİYİ PAKETLEME (RENK DAHİL) ---
        series_data = []
        
        # Kümeler arasında döngü kur (0, 1, 2, 3)
        unique_clusters = df_filtered['cluster'].unique()
        
        for cluster_id in unique_clusters:
            # O kümeye ait veriyi çek
            grup_df = df_filtered[df_filtered['cluster'] == cluster_id]
            
            # Performans için nokta sayısını sınırla (Örn: 150)
            if len(grup_df) > 150:
                grup_df = grup_df.sample(150)
            
            # [Para, Sıklık] formatına getir
            data_points = grup_df[['monetary', 'frequency']].values.tolist()
            
            # İsim ve Renk bilgilerini sözlüklerden çek
            # .get() kullanıyoruz ki listede olmayan bir numara gelirse hata vermesin
            grup_adi = isimlendirme.get(cluster_id, f"Küme {cluster_id}")
            grup_rengi = renk_sozlugu.get(cluster_id, "#999999") # Bulamazsa gri yap

            # ApexCharts'ın istediği format:
            series_data.append({
                "name": grup_adi,
                "data": data_points,
                "color": grup_rengi  # <--- RENK BİLGİSİNİ BURAYA EKLEDİK!
            })
                
        return series_data

    except Exception as e:
        print(f"❌ HATA (K-Means Data): {e}")
        # Hata durumunda boş liste dön ki site çökmesin
        return []
                                    

# ==========================================
# 4. PREDICTION ENGINE (TAHMİN MOTORU) 🧠
# ==========================================

# Global değişkenler (Modelleri bir kere yükle, her defasında yorma)
kmeans_model = None
scaler_model = None

# Global değişkenler
kmeans_model = None
scaler_model = None

def load_models():
    """Uygulama başlarken Joblib ile eğitilmiş modelleri yükler."""
    global kmeans_model, scaler_model
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Arkadaşının kaydettiği dosya yolları
    model_path = os.path.join(base_dir, 'models', 'kmeans_model.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    
    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            kmeans_model = joblib.load(model_path)
            scaler_model = joblib.load(scaler_path)
            
            print(f"✅ Joblib Modelleri Yüklendi!\n📂 Model: {model_path}")
        else:
            print("⚠️ UYARI: Model dosyaları bulunamadı. Lütfen 'scripts/train_model.py' çalıştırın.")
    except Exception as e:
        print(f"❌ Model Yükleme Hatası (Joblib): {e}")

# Uygulama başlarken çalıştır
load_models()


# ==========================================
# 3. APP ROUTES (YÖNLENDİRMELER)
# ==========================================

@app.route('/')
def index():
    """
    Ana Sayfa Yönlendirmesi.
    Tüm analiz modüllerini çalıştırır ve sonuçları 'index.html' şablonuna gönderir.
    """
    
    # Veri modüllerinden sonuçları çek
    rfm_context = get_rfm_data()
    kmeans_series = get_kmeans_data()
    
    # Veri bütünlüğü kontrolü
    if rfm_context is None:
        return "<h1>Sistem Hatası</h1><p>Veri dosyaları okunamadı. Lütfen sunucu loglarını kontrol edin.</p>"

    # Frontend'e veri enjeksiyonu
    return render_template('index.html', 
                           **rfm_context,          # RFM verilerini unpack et
                           scatter_verisi=kmeans_series # K-Means verisini ekle
                           )

# API Endpoint: Tahmin Yap (POST /api/predict)
@app.route('/api/predict', methods=['POST'])
def predict():
    if not kmeans_model:
        return jsonify({'success': False, 'error': 'Modeller sunucuda yüklü değil!'}), 500

    try:
        data = request.json
        # Gelen verileri float'a çeviriyoruz
        recency = float(data.get('recency'))
        frequency = float(data.get('frequency'))
        monetary = float(data.get('monetary'))
        
        # Tahmin işlemi
        input_data = np.array([[recency, frequency, monetary]])
        input_scaled = scaler_model.transform(input_data)
        cluster_id = int(kmeans_model.predict(input_scaled)[0])
        
        # --- BURASI YENİ: Sözlükten ismi çekiyoruz ---
        # app.py'ın başında tanımladığın ISIMLER sözlüğünü kullanır
        segment_adi = ISIMLER.get(cluster_id, f"Segment {cluster_id}")
        
        return jsonify({
            'success': True,
            'cluster': cluster_id,
            'segment_name': segment_adi # JavaScript'e ismi gönderiyoruz
        })

    except Exception as e:
        print(f"Tahmin Hatası: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400
    
if __name__ == '__main__':
    app.run(debug=True)