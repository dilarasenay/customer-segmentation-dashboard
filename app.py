import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib # Modelleri okumak için 
import numpy as np 


app = Flask(__name__)

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
    RFM Analizi verilerini okur ve Dashboard KPI'ları için hazırlar.
    Kaynak: data/processed/customers_rfm.csv
    """
    file_path = get_data_path('customers_rfm.csv')
    
    try:
        # Veri okuma
        df = pd.read_csv(file_path)
        
        # Sütun isim standardizasyonu (Case-insensitive işlem için)
        df.columns = df.columns.str.lower()
        
        # --- KPI Hesaplamaları ---
        total_customers = len(df)
        
        # Ciro Hesaplama (Monetary)
        if 'monetary' in df.columns:
            total_revenue = df['monetary'].sum()
            revenue_formatted = f"{total_revenue:,.0f} ₺"
        else:
            revenue_formatted = "Veri Yok"

        # Lider Segment (En yüksek frekansa sahip grup)
        leading_segment = "Belirsiz"
        if 'segment' in df.columns:
            leading_segment = df['segment'].value_counts().idxmax()
        
        # --- Görselleştirme Hazırlığı (Frontend Formatı) ---
        
        # 1. Tablo Verisi: Ciroya göre top 100 müşteri
        table_data = []
        if 'monetary' in df.columns:
            table_data = df.sort_values(by='monetary', ascending=False).head(100).to_dict('records')

        # 2. Pasta Grafik Verisi: Segment dağılımı
        chart_labels = []
        chart_values = []
        if 'segment' in df.columns:
            segment_counts = df['segment'].value_counts()
            chart_labels = [label.replace('_', ' ').title() for label in segment_counts.index.tolist()]
            chart_values = segment_counts.values.tolist()

        return {
            "sayi": total_customers,
            "skor": revenue_formatted,
            "isim": leading_segment,
            "grafik_etiketleri": chart_labels,
            "grafik_verileri": chart_values,
            "tablo_verisi": table_data
        }

    except FileNotFoundError:
        print(f"HATA: 'customers_rfm.csv' dosyası belirtilen dizinde bulunamadı.")
        return None
    except Exception as e:
        print(f"KRİTİK HATA (RFM Modülü): {e}")
        return None


def get_kmeans_data():
    """
    K-Means verisini hazırlar ama grafiği bozan AYKIRI DEĞERLERİ (Outliers) temizler.
    """
    file_path = get_data_path('rfm_clustered.csv')
    
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower()
        
        series_data = []
        cluster_col = 'cluster' # veya 'segment'
        
        # --- OUTLIER TEMİZLİĞİ (GRAFİĞİ FERAHLATMAK İÇİN) ---
        # Harcamanın %95'inden fazlasını yapanları grafiğe almıyoruz.
        # Bu, grafiğin "zoom" yapmasını ve kümelerin ayrışmasını sağlar.
        esik_deger = df['monetary'].quantile(0.95)
        df_filtered = df[df['monetary'] < esik_deger] 
        
        if cluster_col in df_filtered.columns:
            unique_clusters = sorted(df_filtered[cluster_col].unique())
            
            for cluster_id in unique_clusters:
                cluster_df = df_filtered[df_filtered[cluster_col] == cluster_id]
                
                # Her kümeden rastgele 50 kişi al (sample), head(50) değil!
                # head() yaparsan sadece en tepedekileri alırsın, sample() karışık alır.
                if len(cluster_df) > 50:
                    sample_data = cluster_df[['monetary', 'frequency']].sample(50).values.tolist()
                else:
                    sample_data = cluster_df[['monetary', 'frequency']].values.tolist()
                
                series_data.append({
                    "name": f"Segment {cluster_id}", 
                    "data": sample_data
                })
                
        return series_data

    except Exception as e:
        print(f"HATA: {e}")
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
def predict_segment():
    """
    Frontend'den gelen veriyi alır, Joblib modelleriyle tahmin yapar.
    """
    # Global modelleri kontrol et
    if not kmeans_model or not scaler_model:
        return jsonify({'success': False, 'error': 'Modeller sunucuda yüklü değil!'}), 500

    try:
        # 1. Veriyi al
        data = request.json
        
        # 2. Değerleri hazırla
        recency = float(data.get('recency'))
        frequency = float(data.get('frequency'))
        monetary = float(data.get('monetary'))
        
        # 3. Model formatına çevir (2 Boyutlu array)
        # Scaler beklediği için önce ölçeklendiriyoruz
        input_data = np.array([[recency, frequency, monetary]])
        input_scaled = scaler_model.transform(input_data)
        
        # 4. Tahmin yap
        cluster_id = kmeans_model.predict(input_scaled)[0]
        
        # 5. Cevabı gönder
        return jsonify({
            'success': True,
            'cluster': int(cluster_id)
        })

    except Exception as e:
        print(f"Tahmin Hatası: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400
    
if __name__ == '__main__':
    app.run(debug=True)