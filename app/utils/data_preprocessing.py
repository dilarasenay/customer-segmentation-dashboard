import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def scale_rfm_features(df, features=['recency', 'frequency', 'monetary']):
    """
    RFM sütunlarını StandardScaler kullanarak ölçeklendirir.
    """
    scaler = StandardScaler()
    
    # Verinin bozulmaması için kopyasını alıyoruz
    df_scaled = df.copy()
    
    # Sadece sayısal ham değerleri ölçeklendiriyoruz
    df_scaled[features] = scaler.fit_transform(df[features])
    
    return df_scaled, scaler

def save_scaler(scaler, path='models/scaler.pkl'):
    """Scaler nesnesini modeller klasörüne kaydeder."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)