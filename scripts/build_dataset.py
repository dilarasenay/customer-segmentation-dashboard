import sys
import os
import pandas as pd

# Proje yollarını ayarla
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.utils.data_preprocessing import scale_rfm_features

def main():
    input_path = 'data/processed/customers_rfm.csv'
    output_path = 'data/processed/rfm_scaled.csv'
    
    if os.path.exists(input_path):
        df_rfm = pd.read_csv(input_path)
        print("Scaling işlemi başlatılıyor...")
        
        # Fonksiyonu çağır
        df_scaled = scale_rfm_features(df_rfm)
        
        # Sonucu kaydet
        df_scaled.to_csv(output_path, index=False)
        print(f"İşlem bitti! Ölçeklenmiş veri burada: {output_path}")
    else:
        print(f"Hata: {input_path} bulunamadı!")

if __name__ == "__main__":
    main()