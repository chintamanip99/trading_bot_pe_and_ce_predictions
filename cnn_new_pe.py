"""
CNN PE trainer
Outputs: cnn_fixed_pe.h5, cnn_model_config_pe.csv
Shares cnn_feature_scaler.pkl with CE (same raw OHLCV+OI range).
"""
import numpy as np
import pandas as pd
import requests, datetime, joblib, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from pyts.image import GramianAngularField
from imblearn.over_sampling import RandomOverSampler

# ---------------- CONFIG ----------------
INSTRUMENT_KEY  = "MCX_FO|580936"    # ⬅ PE
ACCESS_TOKEN    = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiIzMjc3NTkiLCJqdGkiOiI2YWJhMTFmOWQ4MGE3OTc3MmY4NWMzZDIiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzkwNTc5MTkzLCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3OTA2MzI4MDB9.0ZX_KioQPLHW3xRTiD_4MyP27tOplMJYxdBaiWjW02s"
INTERVAL        = "1minute"
HISTORY_DAYS    = 10

WINDOW_SIZE     = 10
TARGET_POSITION = 1
BATCH_SIZE      = 8
EPOCHS          = 100
LEARNING_RATE   = 1e-3

MODEL_FILE   = "cnn_fixed_pe.h5"
SCALER_FILE  = "cnn_feature_scaler.pkl"    # shared with CE
CONFIG_FILE  = "cnn_model_config_pe.csv"

FEATURES      = ['Open','High','Low','Close','Volume','OI']
FEATURE_COUNT = len(FEATURES)


def clean_data(df):
    df.replace([np.inf,-np.inf], np.nan, inplace=True)
    df.ffill(inplace=True); df.bfill(inplace=True); df.fillna(0, inplace=True)
    return df


def fetch_and_preprocess_data():
    to_date   = datetime.datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.datetime.now() - datetime.timedelta(days=HISTORY_DAYS)).strftime('%Y-%m-%d')
    url = f"https://api.upstox.com/v2/historical-candle/{INSTRUMENT_KEY}/{INTERVAL}/{to_date}/{from_date}"
    headers = {'Accept':'application/json',
               'Authorization': f'Bearer {ACCESS_TOKEN}'}
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        print(f"API failed: {r.status_code}"); return None, None
    candles = r.json().get('data', {}).get('candles', [])
    if not candles:
        print("No candles"); return None, None

    df = pd.DataFrame(candles,
                      columns=['Datetime','Open','High','Low','Close','Volume','Extra'])
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    for c in ['Open','High','Low','Close','Volume']:
        df[c] = df[c].astype(float)
    if 'Extra' in df.columns:
        df['OI'] = pd.to_numeric(df['Extra'], errors='coerce')
        df['OI'].fillna(df['Volume'], inplace=True)
    else:
        df['OI'] = df['Volume'].rolling(10).mean().fillna(df['Volume'])

    df = df[FEATURES].copy()
    df = clean_data(df)

    df['future_price'] = df['Close'].shift(-TARGET_POSITION)
    df['price_change'] = df['future_price'] - df['Close']
    up_q  = df['price_change'].quantile(0.75)
    low_q = df['price_change'].quantile(0.25)
    df['label'] = df['price_change'].apply(
        lambda x: 1 if x > up_q else 2 if x < low_q else 0
    )
    df.dropna(inplace=True)

    print(f"Data shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")

    # ---- IMPORTANT: reuse CE scaler if it exists; otherwise fit on PE ----
    import os
    if os.path.exists(SCALER_FILE):
        scaler = joblib.load(SCALER_FILE)
        print(f"✅ Reusing existing {SCALER_FILE}")
    else:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(df[FEATURES].values)
        joblib.dump(scaler, SCALER_FILE)
        print(f"✅ Saved NEW {SCALER_FILE}")

    df[FEATURES] = scaler.transform(df[FEATURES])

    return df, FEATURES


def create_gaf_images(data, columns, window_size):
    gaf = GramianAngularField(method='summation', image_size=window_size)
    imgs = []
    for i in range(window_size, len(data)):
        win = data.iloc[i-window_size:i][columns].values.T.astype(np.float64)
        try:
            imgs.append(gaf.fit_transform(win))
        except Exception as e:
            print(f"GAF err @{i}: {e}")
    return np.array(imgs)


def build_model(input_shape):
    m = Sequential([
        Conv2D(32, (3,3), activation='swish', padding='same', input_shape=input_shape),
        BatchNormalization(), MaxPooling2D((2,2)), Dropout(0.2),
        Conv2D(64, (3,3), activation='swish', padding='same'),
        BatchNormalization(), MaxPooling2D((2,2)), Dropout(0.2),
        Conv2D(128, (3,3), activation='swish', padding='same'),
        BatchNormalization(), MaxPooling2D((2,2)), Dropout(0.2),
        Flatten(),
        Dense(128, activation='swish'), Dropout(0.3),
        Dense(64,  activation='swish'), Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    m.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return m


def main():
    print("=" * 70)
    print("CNN PE TRAINER")
    print("=" * 70)

    df, feats = fetch_and_preprocess_data()
    if df is None or len(df) < WINDOW_SIZE + 2:
        print("❌ Not enough data"); return

    print("\nCreating GAF images...")
    gaf = create_gaf_images(df, feats, WINDOW_SIZE)
    if len(gaf) == 0:
        print("❌ No GAF images"); return

    X = np.transpose(gaf, (0, 2, 3, 1))
    y = df['label'].values[WINDOW_SIZE:]
    L = min(len(X), len(y)); X, y = X[:L], y[:L]

    print(f"X: {X.shape}  y: {y.shape}")
    print(f"Class dist before: {np.bincount(y)}")

    ros = RandomOverSampler(random_state=42)
    X_flat = X.reshape(X.shape[0], -1)
    Xr, yr = ros.fit_resample(X_flat, y)
    Xr = Xr.reshape(-1, WINDOW_SIZE, WINDOW_SIZE, len(feats))
    print(f"Class dist after:  {np.bincount(yr)}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        Xr, yr, test_size=0.2, random_state=42, stratify=yr
    )
    print(f"Train: {len(X_tr)}  Val: {len(X_val)}")

    model = build_model((WINDOW_SIZE, WINDOW_SIZE, len(feats)))
    model.summary()

    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                            min_lr=1e-5, verbose=1)
    es  = EarlyStopping(monitor='val_loss', patience=20,
                        restore_best_weights=True, verbose=1)

    model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_val, y_val),
              callbacks=[rlr, es], verbose=1)

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    present = np.unique(y_val)
    names   = ['Hold','Buy','Sell']
    present_names = [names[i] for i in present if i < len(names)]
    print("\n" + classification_report(y_val, y_pred, target_names=present_names))

    model.save(MODEL_FILE)
    print(f"✅ Saved {MODEL_FILE}")

    cfg = {
        'model_file': MODEL_FILE,
        'features': FEATURES,
        'feature_count': len(FEATURES),
        'window_size': WINDOW_SIZE,
        'target_position': TARGET_POSITION,
        'classes': ['Hold','Buy','Sell'],
        'class_mapping': {0:'Hold', 1:'Buy (Bullish)', 2:'Sell (Bearish)'},
        'scaler_file': SCALER_FILE,
        'feature_engineering': 'raw_OHLCV_OI -> MinMax(-1,1) -> GAF(summation)',
        'saved_date': datetime.datetime.now().isoformat()
    }
    pd.DataFrame([cfg]).to_csv(CONFIG_FILE, index=False)
    print(f"✅ Saved {CONFIG_FILE}")

    try:
        r = requests.post("http://localhost:5000/update",
                          json={"Updated":"CNN_PE"}, timeout=5)
        print(f"📡 Bot notified: {r.status_code}")
    except Exception as e:
        print(f"⚠️ Bot not reachable: {e}")


if __name__ == "__main__":
    while True:
        main()
