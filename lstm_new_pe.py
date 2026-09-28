"""
LSTM PE trainer
Outputs: lstm_fixed_pe.h5, scaler_X_pe.pkl, scaler_y_pe.pkl, lstm_model_config_pe.csv
"""
import numpy as np
import pandas as pd
import requests, datetime, joblib, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- CONFIG ----------------
INSTRUMENT_KEY   = "MCX_FO|580936"    # ⬅ PE
ACCESS_TOKEN     = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiIzMjc3NTkiLCJqdGkiOiI2YWJhMTFmOWQ4MGE3OTc3MmY4NWMzZDIiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzkwNTc5MTkzLCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3OTA2MzI4MDB9.0ZX_KioQPLHW3xRTiD_4MyP27tOplMJYxdBaiWjW02s"
INTERVAL         = "1minute"
HISTORY_DAYS     = 20

LOOKBACK         = 30
TARGET_POSITION  = 1

MODEL_FILE       = "lstm_fixed_pe.h5"
SCALER_X_FILE    = "scaler_X_pe.pkl"
SCALER_Y_FILE    = "scaler_y_pe.pkl"
CONFIG_FILE      = "lstm_model_config_pe.csv"

EPOCHS        = 100
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3

FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI']


def make_stationary_features(df):
    feat = pd.DataFrame(index=df.index)
    for c in ['Open', 'High', 'Low', 'Close']:
        feat[c] = np.log(df[c] / df[c].shift(1))
    feat['Volume'] = np.log1p(df['Volume']).diff()
    feat['OI']     = np.log1p(df['OI']).diff()
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.dropna(inplace=True)
    return feat


def fetch_historical_data():
    to_date   = datetime.datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.datetime.now() - datetime.timedelta(days=HISTORY_DAYS)).strftime('%Y-%m-%d')
    url = f"https://api.upstox.com/v2/historical-candle/{INSTRUMENT_KEY}/{INTERVAL}/{to_date}/{from_date}"
    headers = {'Accept': 'application/json',
               'Authorization': f'Bearer {ACCESS_TOKEN}'}
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        print(f"API failed: {r.status_code}"); return None
    candles = r.json().get('data', {}).get('candles', [])
    if not candles:
        print("No candles"); return None
    df = pd.DataFrame(candles,
                      columns=['Datetime','Open','High','Low','Close','Volume','Extra'])
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    for c in ['Open','High','Low','Close','Volume']:
        df[c] = df[c].astype(float)
    if 'Extra' in df.columns:
        df['OI'] = pd.to_numeric(df['Extra'], errors='coerce')
        df['OI'].fillna(df['Volume'], inplace=True)
    else:
        df['OI'] = df['Volume']
    df = df[FEATURE_COLS].copy()
    df.replace([np.inf,-np.inf], np.nan, inplace=True)
    df.ffill(inplace=True); df.bfill(inplace=True); df.fillna(0, inplace=True)
    df.sort_index(inplace=True)
    return df


def build_dataset(df):
    feat_df = make_stationary_features(df)
    close   = df['Close'].reindex(feat_df.index)
    target  = np.log(close.shift(-TARGET_POSITION) / close).reindex(feat_df.index)
    valid = target.notna()
    feat_df = feat_df.loc[valid]; target = target.loc[valid]
    fv = feat_df.values.astype(np.float32)
    tv = target.values.astype(np.float32)
    n = len(feat_df) - LOOKBACK + 1
    X = np.zeros((n, LOOKBACK, fv.shape[1]), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        X[i] = fv[i:i+LOOKBACK]
        y[i] = tv[i+LOOKBACK-1]
    return X, y, feat_df, close


def calc_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten(); y_pred = np.array(y_pred).flatten()
    m = ~(np.isnan(y_true)|np.isnan(y_pred)|np.isinf(y_true)|np.isinf(y_pred))
    y_true, y_pred = y_true[m], y_pred[m]
    if len(y_true) == 0:
        return {'R2': np.nan,'MSE': np.nan,'RMSE': np.nan,'MAE': np.nan,'MAPE': np.nan}
    r2 = r2_score(y_true, y_pred); mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse); mae = mean_absolute_error(y_true, y_pred)
    nz = y_true != 0
    mape = np.mean(np.abs((y_true[nz]-y_pred[nz])/y_true[nz]))*100 if nz.sum() else np.nan
    return {'R2': r2,'MSE': mse,'RMSE': rmse,'MAE': mae,'MAPE': mape}


def build_model(shape):
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=shape),
        Dropout(0.2),
        LSTM(64), Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    m.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return m


def main():
    print("=" * 70)
    print("LSTM PE TRAINER")
    print("=" * 70)

    df = fetch_historical_data()
    if df is None or len(df) < LOOKBACK + TARGET_POSITION + 20:
        print("❌ Not enough data"); return
    print(f"Data: {df.shape}")

    X, y, _, close_full = build_dataset(df)
    print(f"X: {X.shape}  y: {y.shape}")

    split = int(len(X)*0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    n_tr, lb, nf = X_tr.shape
    scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
    scaler_X.fit(X_tr.reshape(-1, nf))
    X_tr_s = scaler_X.transform(X_tr.reshape(-1, nf)).reshape(n_tr, lb, nf)
    X_te_s = scaler_X.transform(X_te.reshape(-1, nf)).reshape(X_te.shape)
    y_tr_s = scaler_y.fit_transform(y_tr.reshape(-1, 1))
    y_te_s = scaler_y.transform(y_te.reshape(-1, 1))

    print(f"\nscaler_X_pe.data_min_: {scaler_X.data_min_}")
    print(f"scaler_X_pe.data_max_: {scaler_X.data_max_}")

    model = build_model((X_tr.shape[1], X_tr.shape[2]))
    model.summary()
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model.fit(X_tr_s, y_tr_s, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_split=0.15, callbacks=[es], verbose=1)

    y_tr_pred_ret = scaler_y.inverse_transform(model.predict(X_tr_s, verbose=0)).flatten()
    y_te_pred_ret = scaler_y.inverse_transform(model.predict(X_te_s, verbose=0)).flatten()

    base_pos = np.arange(len(X)) + LOOKBACK - 1
    tgt_pos  = base_pos + TARGET_POSITION

    def to_prices(pos, rets):
        base_idx = pos - TARGET_POSITION
        prev = close_full.iloc[base_idx].values
        act  = close_full.iloc[pos].values
        pred = prev * np.exp(rets)
        return act, pred

    y_tr_act, y_tr_pred = to_prices(tgt_pos[:split], y_tr_pred_ret)
    y_te_act, y_te_pred = to_prices(tgt_pos[split:], y_te_pred_ret)

    tm = calc_metrics(y_tr_act, y_tr_pred)
    te = calc_metrics(y_te_act, y_te_pred)
    print("\nTRAIN:", {k: round(v,6) for k,v in tm.items()})
    print("TEST :", {k: round(v,6) for k,v in te.items()})

    model.save(MODEL_FILE)
    joblib.dump(scaler_X, SCALER_X_FILE)
    joblib.dump(scaler_y, SCALER_Y_FILE)
    print(f"\n✅ Saved {MODEL_FILE}, {SCALER_X_FILE}, {SCALER_Y_FILE}")

    cfg = {
        'lookback': LOOKBACK,
        'target_position': TARGET_POSITION,
        'feature_columns': FEATURE_COLS,
        'feature_engineering': 'log_return_OHLC + diff_log1p_Volume_OI',
        'target': f'cumulative_log_return_of_Close_over_{TARGET_POSITION}_steps',
        'training_metrics': tm,
        'test_metrics': te,
        'model_filename': MODEL_FILE,
        'scaler_X_filename': SCALER_X_FILE,
        'scaler_y_filename': SCALER_Y_FILE,
        'last_close': float(df['Close'].iloc[-1]),
        'saved_date': datetime.datetime.now().isoformat()
    }
    pd.DataFrame([cfg]).to_csv(CONFIG_FILE, index=False)
    print(f"✅ Saved {CONFIG_FILE}")


if __name__ == "__main__":
    while True:
        main()
