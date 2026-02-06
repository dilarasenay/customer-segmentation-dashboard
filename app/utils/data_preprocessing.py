import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

def scale_rfm_features(df, save_scaler=True, scaler_path='models/scaler.pkl'):
    features = ['recency', 'frequency', 'monetary']
    scaler = StandardScaler()
    
    # Scaling işlemi
    x_scaled = scaler.fit_transform(df[features])
    
    # DataFrame oluşturma
    df_scaled = pd.DataFrame(x_scaled, columns=features, index=df.index)
    df_scaled['customer_id'] = df['customer_id']
    
    # Scaler'ı kaydet
    if save_scaler:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        
    return df_scaled