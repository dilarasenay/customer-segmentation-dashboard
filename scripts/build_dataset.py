import pandas as pd
import sys
import os

# Üst dizindeki app modülüne erişebilmek için path ekliyoruz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.data_preprocessing import scale_rfm_features, save_scaler

def main():
    # 1. Arkadaşından gelen dosyayı okuyoruz
    input_path = 'data/raw/customers_rfm.csv'
    output_path = 'data/processed/rfm_scaled.csv'
    
    if not os.path.exists(input_path):
        print(f"Hata: {input_path} dosyası bulunamadı. Lütfen dosyayı data/raw/ altına ekleyin.")
        return

    df = pd.read_csv(input_path)
    
    # 2. Ölçeklendirme işlemini yapıyoruz
    # Arkadaşının hazırladığı 'recency', 'frequency', 'monetary' sütunlarını kullanıyoruz
    print("Ölçeklendirme işlemi başlatılıyor...")
    df_scaled, scaler = scale_rfm_features(df)
    
    # 3. Scaler'ı dashboard kullanımı için kaydet 
    save_scaler(scaler, 'models/scaler.pkl')
    
    # 4. Ölçeklendirilmiş veriyi kaydet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_scaled.to_csv(output_path, index=False)
    
    print(f"İşlem başarıyla tamamlandı!")
    print(f"Ölçeklendirilmiş veri: {output_path}")
    print(f"Kaydedilen Scaler: models/scaler.pkl")

if __name__ == "__main__":
    main()