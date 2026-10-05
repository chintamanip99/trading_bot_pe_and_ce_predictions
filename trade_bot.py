"""
4-MODEL ENSEMBLE PAPER TRADING BOT  (HOLD-AWARE CNN v2)
Starting Balance: ₹1,00,000
Fixed 4% Target / 4% SL

CNN RULE (fixed):
  1. Compute probs [P(hold), P(buy), P(sell)]
  2. If P(hold) is the largest class AND >= CNN_HOLD_DOMINANCE → HOLD
  3. Else pick max(P(buy), P(sell)) only if the winner
     beats the loser by >= CNN_MIN_BUY_SELL_EDGE
  4. Otherwise → HOLD
  5. predicted_class is ALWAYS synced with the actual signal
"""

import numpy as np
import time
import pandas as pd
from tensorflow.keras.models import load_model
import json
import requests
import datetime
import threading
import sys
from flask import Flask, request, jsonify
from pyts.image import GramianAngularField
import upstox_client
from tensorflow.nn import softmax
import nest_asyncio
import traceback
from flask_cors import CORS, cross_origin
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
LOT_SIZE = 10
INITIAL_BALANCE = 100000

TARGET_PCT = 0.02
SL_PCT     = 0.04

MAX_HOLD_TIME = 20
CAPITAL_UTILIZATION = 0.95

# ---- CNN Hold-aware gating ----
CNN_IGNORE_HOLD = True
# If P(hold) is the top class AND >= this, force HOLD
CNN_HOLD_DOMINANCE = 0.35          # lowered — 40% Hold should trigger HOLD
# Winner of Buy vs Sell must beat loser by at least this
CNN_MIN_BUY_SELL_EDGE = 0.15       # 15% absolute gap required

LSTM_LOOKBACK = 30
CNN_LOOKBACK  = 10

LSTM_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI']
CNN_FEATURES  = LSTM_FEATURES.copy()

UPSTOX_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiIzMjc3NTkiLCJqdGkiOiI2YWMzMmVkOTE2ZDkzYzNkZWJkYWI5MTEiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzkxMTc2NDA5LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3OTEyMzc2MDB9.fjpYSTukzFYUXdVLO93l1-5HXRylGJqB2iTTcJOGqdM"
INSTRUMENT_TOKEN_CE = "NSE_FO|51330"
INSTRUMENT_TOKEN_PE = "NSE_FO|51331"

nest_asyncio.apply()

# ============================================================
# STATE
# ============================================================
app = Flask(__name__)
CORS(app)

current_position = None
g_current_status = "Initializing..."

ce_price = 0
pe_price = 0

ce_ohlcv_buffer = []
pe_ohlcv_buffer = []

trade_log_global = []
recent_trades = []

last_signal = 0

lstm_model = cnn_model = lstm_scaler = target_scaler = cnn_scaler = None
lstm_model_pe = cnn_model_pe = lstm_scaler_pe = target_scaler_pe = cnn_scaler_pe = None

trades_today = 0
current_trade_date = None

latest_lstm_predicted_price = 0.0
latest_close_diff_predicted = 0.0
latest_lstm_pe_predicted_price = 0.0
latest_close_diff_predicted_pe = 0.0

model_predictions = {
    'lstm':    {'signal': 0, 'confidence': 0.0, 'prediction': 0.0,
                'close_diff_predicted': 0.0},
    'cnn':     {'signal': 0, 'confidence': 0.0, 'class_prob': 0.0,
                'predicted_class': 0, 'lead': 0.0, 'probs': [0.0, 0.0, 0.0],
                'forced_hold': False, 'hold_dominance': 0.0},
    'lstm_pe': {'signal': 0, 'confidence': 0.0, 'prediction': 0.0,
                'close_diff_predicted': 0.0},
    'cnn_pe':  {'signal': 0, 'confidence': 0.0, 'class_prob': 0.0,
                'predicted_class': 0, 'lead': 0.0, 'probs': [0.0, 0.0, 0.0],
                'forced_hold': False, 'hold_dominance': 0.0}
}

last_consensus_debug = {}
diag_log = []
MAX_DIAG = 200


def diag(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    diag_log.append(line)
    if len(diag_log) > MAX_DIAG:
        diag_log.pop(0)


# ============================================================
# FEATURES
# ============================================================
def make_stationary_features(df):
    feat = pd.DataFrame(index=df.index)
    for c in ['Open', 'High', 'Low', 'Close']:
        feat[c] = np.log(df[c] / df[c].shift(1))
    feat['Volume'] = np.log1p(df['Volume']).diff()
    feat['OI']     = np.log1p(df['OI']).diff()
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.dropna(inplace=True)
    return feat


def _is_fitted_scaler(obj):
    return (obj is not None and
            (hasattr(obj, "n_features_in_") or hasattr(obj, "scale_") or
             hasattr(obj, "min_") or hasattr(obj, "center_") or
             hasattr(obj, "mean_")))


# ============================================================
# MODELS
# ============================================================
def load_all_models():
    global lstm_model, cnn_model, lstm_scaler, target_scaler, cnn_scaler
    global lstm_model_pe, cnn_model_pe, lstm_scaler_pe, target_scaler_pe, cnn_scaler_pe

    loaded = {'lstm': False, 'cnn': False, 'lstm_pe': False, 'cnn_pe': False}

    try:
        lstm_model = load_model("lstm_fixed.h5", compile=False)
        loaded['lstm'] = True
        print(f"✅ LSTM(CE) loaded  shape={lstm_model.input_shape}")
    except Exception as e:
        print(f"❌ LSTM(CE): {e}")
        lstm_model = None

    try:
        cnn_model = load_model("cnn_fixed.h5", compile=False)
        loaded['cnn'] = True
        print(f"✅ CNN(CE) loaded  shape={cnn_model.input_shape}")
    except Exception as e:
        print(f"❌ CNN(CE): {e}")
        cnn_model = None

    try:
        lstm_scaler = joblib.load("scaler_X.pkl")
        target_scaler = joblib.load("scaler_y.pkl")
        print(f"✅ LSTM(CE) scalers loaded")
    except Exception as e:
        print(f"❌ LSTM(CE) scalers: {e}")
        lstm_scaler = None
        target_scaler = None

    try:
        cnn_scaler = joblib.load("cnn_feature_scaler.pkl")
        print(f"✅ CNN(CE) scaler loaded")
    except Exception as e:
        print(f"❌ CNN(CE) scaler: {e}")
        cnn_scaler = None

    try:
        lstm_model_pe = load_model("lstm_fixed_pe.h5", compile=False)
        loaded['lstm_pe'] = True
        print(f"✅ LSTM(PE) loaded  shape={lstm_model_pe.input_shape}")
    except Exception as e:
        print(f"❌ LSTM(PE): {e}")
        lstm_model_pe = None

    try:
        cnn_model_pe = load_model("cnn_fixed_pe.h5", compile=False)
        loaded['cnn_pe'] = True
        print(f"✅ CNN(PE) loaded  shape={cnn_model_pe.input_shape}")
    except Exception as e:
        print(f"❌ CNN(PE): {e}")
        cnn_model_pe = None

    try:
        lstm_scaler_pe = joblib.load("scaler_X_pe.pkl")
        target_scaler_pe = joblib.load("scaler_y_pe.pkl")
        print(f"✅ LSTM(PE) scalers loaded")
    except Exception as e:
        print(f"⚠️  LSTM(PE) scalers missing ({e}) — falling back to CE")
        lstm_scaler_pe = lstm_scaler
        target_scaler_pe = target_scaler

    cnn_scaler_pe = cnn_scaler
    return loaded


# ============================================================
# HISTORICAL
# ============================================================
def fetch_historical_ohlcv(instrument_key, days=10):
    to_date = datetime.datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')

    url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/1minute/{to_date}/{from_date}"
    headers = {'Accept': 'application/json',
               'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            print(f"❌ Historical {instrument_key} → {r.status_code}")
            return None
        candles = r.json().get('data', {}).get('candles', [])
        if not candles:
            return None
        df = pd.DataFrame(candles,
                          columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Extra'])
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[c] = df[c].astype(float)
        df['OI'] = pd.to_numeric(df.get('Extra', df['Volume']), errors='coerce')
        df['OI'].fillna(df['Volume'], inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'OI']].copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True); df.bfill(inplace=True); df.fillna(0, inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"❌ Historical error: {e}")
        return None


# ============================================================
# LSTM
# ============================================================
def _predict_lstm_core(df_raw, scaler_X, scaler_y, model):
    if model is None or scaler_X is None or scaler_y is None:
        return None
    feat = make_stationary_features(df_raw)
    if len(feat) < LSTM_LOOKBACK:
        return None
    window = feat.tail(LSTM_LOOKBACK).values.astype(np.float32)
    try:
        ws = scaler_X.transform(window)
    except Exception as e:
        print(f"❌ LSTM scaler: {e}")
        return None
    X = ws.reshape(1, LSTM_LOOKBACK, window.shape[1])
    try:
        ps = model.predict(X, verbose=0)
    except Exception as e:
        print(f"❌ LSTM predict: {e}")
        return None
    try:
        plr = float(scaler_y.inverse_transform(np.array(ps).reshape(-1, 1))[0, 0])
    except Exception as e:
        print(f"❌ LSTM inverse: {e}")
        return None
    cc = float(df_raw['Close'].iloc[-1])
    pp = cc * np.exp(plr)
    return {'pred_price': pp, 'pred_log_return': plr,
            'current_close': cc, 'close_diff': pp - cc}


def predict_lstm_ce(df_raw):
    global model_predictions, latest_lstm_predicted_price, latest_close_diff_predicted
    res = _predict_lstm_core(df_raw, lstm_scaler, target_scaler, lstm_model)
    if res is None:
        return
    cd = res['close_diff']
    latest_lstm_predicted_price = res['pred_price']
    latest_close_diff_predicted = cd

    if cd > 0:
        sig = 1
    elif cd < 0:
        sig = 2
    else:
        sig = 0

    model_predictions['lstm'] = {
        'signal': sig,
        'confidence': abs(cd),
        'prediction': res['pred_price'],
        'close_diff_predicted': cd,
    }
    diag(f"LSTM(CE) sig={sig} Δ={cd:+.6f}")


def predict_lstm_pe(df_raw):
    global model_predictions, latest_lstm_pe_predicted_price, latest_close_diff_predicted_pe
    res = _predict_lstm_core(df_raw, lstm_scaler_pe, target_scaler_pe, lstm_model_pe)
    if res is None:
        return
    cd = res['close_diff']
    latest_lstm_pe_predicted_price = res['pred_price']
    latest_close_diff_predicted_pe = cd

    if cd > 0:
        sig = 1
    elif cd < 0:
        sig = 2
    else:
        sig = 0

    model_predictions['lstm_pe'] = {
        'signal': sig,
        'confidence': abs(cd),
        'prediction': res['pred_price'],
        'close_diff_predicted': cd,
    }
    diag(f"LSTM(PE) sig={sig} Δ={cd:+.6f}")


# ============================================================
# CNN  —  HOLD-AWARE v2 (predicted_class synced with signal)
# ============================================================
def _predict_cnn_core(df_raw, scaler, model):
    if model is None or scaler is None:
        return None
    if len(df_raw) < CNN_LOOKBACK:
        return None
    window = df_raw[CNN_FEATURES].tail(CNN_LOOKBACK).values.astype(np.float64)
    try:
        ws = scaler.transform(window)
    except Exception as e:
        print(f"❌ CNN scaler: {e}")
        return None
    gaf = GramianAngularField(method='summation', image_size=CNN_LOOKBACK)
    try:
        imgs = gaf.fit_transform(ws.T)
    except Exception as e:
        print(f"❌ GAF: {e}")
        return None
    cX = imgs.transpose(1, 2, 0)[np.newaxis, ...]
    expected = model.input_shape
    if expected[1] is not None and cX.shape[1] != expected[1]:
        print(f"❌ CNN shape mismatch got={cX.shape} exp={expected}")
        return None
    try:
        pred = model.predict(cX, verbose=0)
    except Exception as e:
        print(f"❌ CNN predict: {e}")
        return None

    raw = np.array(pred).flatten()
    if raw.size == 1:
        p1 = float(raw[0])
        probs = np.array([1 - p1, p1, 0.0])
    else:
        if np.all(raw >= 0) and np.isclose(np.sum(raw), 1.0, atol=0.05):
            probs = raw
        else:
            probs = softmax(pred).numpy().flatten()

    if probs.size < 3:
        probs = np.pad(probs, (0, 3 - probs.size), constant_values=0.0)

    p_hold = float(probs[0])
    p_buy  = float(probs[1])
    p_sell = float(probs[2])

    # ---- Classify which class is largest ----
    argmax_class = int(np.argmax(probs))

    forced_hold = False

    if not CNN_IGNORE_HOLD:
        # Standard 3-class argmax
        sig = argmax_class
        pc = argmax_class
        cp = float(probs[pc])
        lead = float(np.sort(probs)[::-1][0] - np.sort(probs)[::-1][1])
    else:
        # --- Rule 1: Hold is top class AND above dominance → HOLD ---
        if argmax_class == 0 and p_hold >= CNN_HOLD_DOMINANCE:
            sig = 0
            pc = 0
            cp = p_hold
            lead = p_hold - max(p_buy, p_sell)
            forced_hold = True
        else:
            # --- Rule 2: pick winner of Buy vs Sell, but require edge ---
            if p_buy >= p_sell:
                win_p, lose_p = p_buy, p_sell
                cand_sig, cand_pc = 1, 1
            else:
                win_p, lose_p = p_sell, p_buy
                cand_sig, cand_pc = 2, 2

            edge = win_p - lose_p
            if edge >= CNN_MIN_BUY_SELL_EDGE:
                sig = cand_sig
                pc = cand_pc          # ← SYNC: predicted_class = signal
                cp = win_p
                lead = edge
            else:
                # Too close → HOLD
                sig = 0
                pc = 0                # ← SYNC: predicted_class = 0 = Hold
                cp = p_hold
                lead = abs(p_buy - p_sell)
                forced_hold = True

    return {
        'probs': probs.tolist(),
        'predicted_class': pc,        # always 0/1/2 matching signal
        'class_prob': cp,
        'lead': lead,
        'signal': sig,                # always 0/1/2
        'confidence': cp,
        'forced_hold': forced_hold,
        'hold_dominance': p_hold,
        'argmax_class': argmax_class, # raw argmax (for diagnostics)
    }


def predict_cnn_ce(df_raw):
    global model_predictions
    res = _predict_cnn_core(df_raw, cnn_scaler, cnn_model)
    if res is None:
        return
    model_predictions['cnn'] = res
    fh = " [FORCED HOLD]" if res.get('forced_hold') else ""
    diag(f"CNN(CE) sig={res['signal']} argmax={res['argmax_class']}{fh} "
         f"probs=[H:{res['probs'][0]:.2f} B:{res['probs'][1]:.2f} S:{res['probs'][2]:.2f}]")


def predict_cnn_pe(df_raw):
    global model_predictions
    res = _predict_cnn_core(df_raw, cnn_scaler_pe, cnn_model_pe)
    if res is None:
        return
    model_predictions['cnn_pe'] = res
    fh = " [FORCED HOLD]" if res.get('forced_hold') else ""
    diag(f"CNN(PE) sig={res['signal']} argmax={res['argmax_class']}{fh} "
         f"probs=[H:{res['probs'][0]:.2f} B:{res['probs'][1]:.2f} S:{res['probs'][2]:.2f}]")


# ============================================================
# CONSENSUS
# ============================================================
def ensemble_signal_generation():
    global last_consensus_debug

    lstm_s = model_predictions['lstm']['signal']
    cnn_s  = model_predictions['cnn']['signal']
    lpe_s  = model_predictions['lstm_pe']['signal']
    cpe_s  = model_predictions['cnn_pe']['signal']

    pattern = (lstm_s, cnn_s, lpe_s, cpe_s)
    diag(f"PATTERN={pattern}")

    is_bull = (
        (lstm_s == 1 and cnn_s == 1 and lpe_s == 2 and cpe_s == 2) or
        (lstm_s == 1 and cnn_s == 0 and lpe_s == 2 and cpe_s == 2) or
        (lstm_s == 0 and cnn_s == 1 and lpe_s == 2 and cpe_s == 2) or
        (lstm_s == 1 and cnn_s == 1 and lpe_s == 0 and cpe_s == 2) or
        (lstm_s == 1 and cnn_s == 1 and lpe_s == 2 and cpe_s == 0)
    )
    is_bear = (
        (lstm_s == 2 and cnn_s == 2 and lpe_s == 1 and cpe_s == 1) or
        (lstm_s == 2 and cnn_s == 0 and lpe_s == 1 and cpe_s == 1) or
        (lstm_s == 0 and cnn_s == 2 and lpe_s == 1 and cpe_s == 1) or
        (lstm_s == 2 and cnn_s == 2 and lpe_s == 0 and cpe_s == 1) or
        (lstm_s == 2 and cnn_s == 2 and lpe_s == 1 and cpe_s == 0)
    )

    if is_bull:
        sig = 1
        ctype = "BULL_CE"
        diag(f"✅ CONSENSUS BULL | signal={sig}")
    elif is_bear:
        sig = 2
        ctype = "BEAR_PE"
        diag(f"✅ CONSENSUS BEAR | signal={sig}")
    else:
        last_consensus_debug = {
            'pattern': pattern, 'pattern_str': f"{lstm_s},{cnn_s},{lpe_s},{cpe_s}",
            'reason': 'no template match',
            'timestamp': datetime.datetime.now().isoformat(),
        }
        diag(f"REJECT: pattern {pattern} not matched")
        return 0, 0.0, {'lstm': lstm_s, 'cnn': cnn_s, 'lstm_pe': lpe_s, 'cnn_pe': cpe_s,
                        'consensus': 'REJECT', 'reject_reason': 'no template match',
                        'ensemble_confidence': 0.0}

    last_consensus_debug = {
        'pattern': pattern, 'pattern_str': f"{lstm_s},{cnn_s},{lpe_s},{cpe_s}",
        'reason': 'ACCEPTED', 'consensus_type': ctype, 'final_signal': sig,
        'timestamp': datetime.datetime.now().isoformat(),
    }

    return sig, 1.0, {
        'lstm': lstm_s, 'cnn': cnn_s, 'lstm_pe': lpe_s, 'cnn_pe': cpe_s,
        'consensus': ctype, 'reject_reason': None,
        'ensemble_confidence': 1.0, 'pattern_tuple': pattern,
    }


# ============================================================
# PREDICT DRIVER
# ============================================================
def predict_with_ensemble(ce_df, pe_df, ce_p, pe_p):
    global g_current_status, last_signal
    try:
        if len(ce_df) < LSTM_LOOKBACK or len(pe_df) < LSTM_LOOKBACK:
            g_current_status = f"Insufficient data CE={len(ce_df)} PE={len(pe_df)}"
            return
        predict_lstm_ce(ce_df)
        predict_cnn_ce(ce_df)
        predict_lstm_pe(pe_df)
        predict_cnn_pe(pe_df)
        sig, conf, ms = ensemble_signal_generation()
        last_signal = sig
        paper_trading_bot.execute_paper_trade(sig, ce_p, pe_p, conf, ms)
        update_status(ce_p, pe_p, sig, conf, ms)
    except Exception as e:
        g_current_status = f"predict err: {e}"
        traceback.print_exc()


def update_status(ce_p, pe_p, sig, conf, ms):
    global g_current_status
    sig_txt = {0: 'HOLD', 1: 'BUY_CE', 2: 'BUY_PE'}[sig]
    g_current_status = (
        f"CE:{ce_p:.2f} PE:{pe_p:.2f} | Bal:{paper_trading_bot.balance:,.0f} | "
        f"Pos:{paper_trading_bot.position or 'None'} | Sig:{sig_txt} | "
        f"{datetime.datetime.now().strftime('%H:%M:%S')}"
    )


# ============================================================
# PAPER BOT
# ============================================================
class PaperTradingBot:
    def __init__(self, balance=INITIAL_BALANCE, lot_size=LOT_SIZE):
        self.balance = balance
        self.initial_balance = balance
        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.target = None
        self.shares = 0
        self.lot_size = lot_size
        self.last_order_time = None
        self.order_retry_delay = 1
        self.max_hold_time = MAX_HOLD_TIME
        self.current_instrument = None
        self.active_target_pct = TARGET_PCT
        self.active_sl_pct = SL_PCT
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees_paid = 0
        self.largest_win = 0
        self.largest_loss = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 99999
        self.brokerage_fixed = 5
        self.gst_rate = 0.18
        self.stt_sell = 0.0001
        self.exchange_txn = 0.0000173
        self.stamp_duty = 0.00002
        self.sebi_charges = 0.000001
        self.current_trade_value = 0
        self.entry_fees_paid = 0

    def calculate_max_shares(self, entry_price):
        if entry_price <= 0: return self.lot_size
        budget = self.balance * CAPITAL_UTILIZATION * 0.995
        max_sh = int(budget / entry_price)
        shares = (max_sh // self.lot_size) * self.lot_size
        if shares < self.lot_size:
            if entry_price * self.lot_size <= self.balance:
                return self.lot_size
            return 0
        while shares * entry_price > self.balance and shares > self.lot_size:
            shares -= self.lot_size
        return shares

    def can_place_order(self):
        if self.last_order_time is None: return True
        return (datetime.datetime.now() - self.last_order_time).total_seconds() >= self.order_retry_delay

    def calculate_fees(self, tv, side='entry'):
        br = self.brokerage_fixed
        ex = tv * self.exchange_txn
        sb = tv * self.sebi_charges
        gst = (br + ex + sb) * self.gst_rate
        st = tv * self.stamp_duty if side == 'entry' else 0
        stt = tv * self.stt_sell if side == 'exit' else 0
        return br + ex + sb + gst + st + stt

    def check_exit(self, ce_p, pe_p):
        if self.position is None or self.shares <= 0: return
        cp = ce_p if self.current_instrument == INSTRUMENT_TOKEN_CE else pe_p
        if cp <= 0: return
        if cp >= self.target:
            self._close(cp, f"TARGET (+{self.active_target_pct*100:.2f}%)")
        elif cp <= self.stop_loss:
            self._close(cp, f"STOP (-{self.active_sl_pct*100:.2f}%)")
        elif self.last_order_time:
            hm = (datetime.datetime.now() - self.last_order_time).total_seconds() / 60
            if hm > self.max_hold_time:
                self._close(cp, f"TIME ({hm:.1f}m)")

    def _close(self, xp, reason):
        global trade_log_global, recent_trades, current_position
        try:
            xv = xp * self.shares
            xf = self.calculate_fees(xv, 'exit')
            net = (xp - self.entry_price) * self.shares - self.entry_fees_paid - xf
            pct = (xp - self.entry_price) / self.entry_price * 100
            self.balance += xv - xf
            self.total_fees_paid += xf
            if net > 0:
                self.winning_trades += 1; self.consecutive_losses = 0
                self.largest_win = max(self.largest_win, net)
            else:
                self.losing_trades += 1; self.consecutive_losses += 1
                self.largest_loss = min(self.largest_loss, net)
            inst = "CE" if self.current_instrument == INSTRUMENT_TOKEN_CE else "PE"
            ep, sh = self.entry_price, self.shares
            self.position = None; self.current_instrument = None
            self.entry_price = None; self.stop_loss = None; self.target = None
            self.shares = 0; self.entry_fees_paid = 0; self.current_trade_value = 0
            current_position = None
            msg = f"🚀 EXIT {inst} ({reason}): {ep:.2f}→{xp:.2f} ({pct:+.2f}%) PnL={net:+,.2f} Bal={self.balance:,.2f}"
            print(f"✅ {msg}")
            diag(msg)
            recent_trades.append({'time': datetime.datetime.now(), 'entry_price': ep,
                                  'exit_price': xp, 'type': inst, 'reason': reason,
                                  'net_pnl': net, 'shares': sh})
            trade_log_global.append({'timestamp': datetime.datetime.now().isoformat(), 'message': msg})
            with open("paper_trade_logs.txt", "a") as f:
                f.write(f"{datetime.datetime.now().isoformat()} - {msg}\n")
        except Exception as e:
            print(f"❌ _close: {e}")

    def execute_paper_trade(self, sig, ce_p, pe_p, conf, ms):
        global trade_log_global, trades_today, current_trade_date, current_position

        today = datetime.datetime.now().date()
        if current_trade_date != today:
            current_trade_date = today
            trades_today = 0

        if sig not in (1, 2):
            return

        if self.position is not None:
            diag("SKIP: already in position"); return
        if not self.can_place_order():
            diag("SKIP: order cooldown"); return

        ref = ce_p if sig == 1 else pe_p
        if ref <= 0:
            diag(f"SKIP: ref_price={ref} <= 0"); return

        dyn_t = TARGET_PCT
        dyn_s = SL_PCT
        self.active_target_pct = dyn_t
        self.active_sl_pct = dyn_s

        shares = self.calculate_max_shares(ref)
        if shares <= 0:
            diag(f"SKIP: cannot size shares (ref={ref})"); return

        tp = round(ref * (1 + dyn_t), 2)
        sp = round(ref * (1 - dyn_s), 2)

        tv = ref * shares
        ef = self.calculate_fees(tv, 'entry')
        tc = tv + ef
        while tc > self.balance and shares > self.lot_size:
            shares -= self.lot_size
            tv = ref * shares
            ef = self.calculate_fees(tv, 'entry')
            tc = tv + ef
        if tc > self.balance:
            diag("SKIP: insufficient balance"); return

        self.last_order_time = datetime.datetime.now()
        self.position = "long" if sig == 1 else "put"
        self.entry_price = ref
        self.stop_loss = sp
        self.target = tp
        self.shares = shares
        self.current_instrument = INSTRUMENT_TOKEN_CE if sig == 1 else INSTRUMENT_TOKEN_PE
        current_position = self.position
        self.entry_fees_paid = ef
        self.current_trade_value = tv

        self.balance -= tc
        self.total_fees_paid += ef
        self.total_trades += 1
        trades_today += 1

        side = "CE" if sig == 1 else "PE"
        msg = (f"{'📈' if sig==1 else '📉'} BUY {side} @ {ref:.2f} | "
               f"lots={shares//self.lot_size} | Inv={tv:,.2f} | "
               f"T={tp}(+{dyn_t*100:.2f}%) SL={sp}(-{dyn_s*100:.2f}%) | "
               f"pattern={ms.get('pattern_tuple','?')} | Bal={self.balance:,.2f}")
        print(f"✅ {msg}")
        diag(msg)
        trade_log_global.append({'timestamp': datetime.datetime.now().isoformat(), 'message': msg})
        with open("paper_trade_logs.txt", "a") as f:
            f.write(f"{datetime.datetime.now().isoformat()} - {msg}\n")

    def get_performance_stats(self):
        tot = self.balance - self.initial_balance
        rp = (tot / self.initial_balance) * 100 if self.initial_balance else 0
        wr = (self.winning_trades / self.total_trades * 100) if self.total_trades else 0
        return {'initial_balance': float(self.initial_balance),
                'current_balance': float(self.balance),
                'total_return': float(tot), 'return_percentage': float(rp),
                'total_trades': int(self.total_trades),
                'winning_trades': int(self.winning_trades),
                'losing_trades': int(self.losing_trades),
                'win_rate': float(wr), 'total_fees_paid': float(self.total_fees_paid),
                'largest_win': float(self.largest_win), 'largest_loss': float(self.largest_loss),
                'consecutive_losses': int(self.consecutive_losses),
                'current_position': self.position, 'shares': int(self.shares),
                'active_target_pct': self.active_target_pct * 100,
                'active_sl_pct': self.active_sl_pct * 100,
                'trades_today': int(trades_today)}


paper_trading_bot = PaperTradingBot(balance=INITIAL_BALANCE, lot_size=LOT_SIZE)


# ============================================================
# WEBSOCKET
# ============================================================
def start_websocket():
    global ce_price, pe_price

    ce_min = pe_min = None

    def _extract(feed):
        ltp = None; o = h = l = v = oi = None
        try:
            if 'ltp' in feed: ltp = float(feed['ltp'])
            if 'fullFeed' in feed and 'marketFF' in feed['fullFeed']:
                mff = feed['fullFeed']['marketFF']
                if 'ltpc' in mff and 'ltp' in mff['ltpc']:
                    ltp = float(mff['ltpc']['ltp'])
                if 'ohlc' in mff and isinstance(mff['ohlc'], list) and mff['ohlc']:
                    oc = mff['ohlc'][-1]
                    o = float(oc.get('open', oc.get('o', 0)))
                    h = float(oc.get('high', oc.get('h', 0)))
                    l = float(oc.get('low', oc.get('l', 0)))
                    c_ = float(oc.get('close', oc.get('c', 0)))
                    ltp = c_ if c_ > 0 else ltp
                if 'vtt' in mff: v = float(mff['vtt'])
                if 'oi' in mff: oi = float(mff['oi'])
        except Exception as e:
            print(f"⚠️ extract: {e}")
        return ltp, o, h, l, v, oi

    def _append(buf, ltp, o, h, l, v, oi):
        if not ltp or ltp <= 0: return
        o = o or ltp; h = h or ltp; l = l or ltp
        v = v if v is not None else (buf[-1]['Volume'] if buf else 0.0)
        oi = oi if oi is not None else (buf[-1]['OI'] if buf else v)
        buf.append({'Datetime': pd.Timestamp.now(), 'Open': o, 'High': h,
                    'Low': l, 'Close': ltp, 'Volume': v, 'OI': oi})
        if len(buf) > 500: buf.pop(0)

    def on_message(message):
        nonlocal ce_min, pe_min
        global ce_price, pe_price
        try:
            data = json.loads(message) if isinstance(message, str) else message
            if not isinstance(data, dict): return
            t = data.get('type')
            if t in ('sub', 'market_info'): return
            feeds = data.get('feeds', {})
            if not feeds: return

            now_min = datetime.datetime.now().replace(second=0, microsecond=0)

            if INSTRUMENT_TOKEN_CE in feeds:
                ltp, o, h, l, v, oi = _extract(feeds[INSTRUMENT_TOKEN_CE])
                if ltp and ltp > 0:
                    ce_price = ltp
                    if now_min != ce_min:
                        _append(ce_ohlcv_buffer, ltp, o, h, l, v, oi)
                        ce_min = now_min

            if INSTRUMENT_TOKEN_PE in feeds:
                ltp, o, h, l, v, oi = _extract(feeds[INSTRUMENT_TOKEN_PE])
                if ltp and ltp > 0:
                    pe_price = ltp
                    if now_min != pe_min:
                        _append(pe_ohlcv_buffer, ltp, o, h, l, v, oi)
                        pe_min = now_min

            if ce_price > 0 and pe_price > 0:
                try: paper_trading_bot.check_exit(ce_price, pe_price)
                except Exception as e: print(f"⚠️ check_exit: {e}")

            if (len(ce_ohlcv_buffer) >= LSTM_LOOKBACK and
                len(pe_ohlcv_buffer) >= LSTM_LOOKBACK and
                ce_price > 0 and pe_price > 0):
                ce_df = pd.DataFrame(ce_ohlcv_buffer).set_index('Datetime')
                pe_df = pd.DataFrame(pe_ohlcv_buffer).set_index('Datetime')
                predict_with_ensemble(ce_df, pe_df, ce_price, pe_price)
        except Exception as e:
            print(f"❌ on_message: {e}")
            traceback.print_exc()

    try:
        cfg = upstox_client.Configuration()
        cfg.access_token = UPSTOX_ACCESS_TOKEN
        streamer = upstox_client.MarketDataStreamerV3(
            upstox_client.ApiClient(cfg),
            [INSTRUMENT_TOKEN_CE, INSTRUMENT_TOKEN_PE], "full")
        streamer.on("message", on_message)
        streamer.on("error", lambda e: print(f"❌ WS: {e}"))
        print("📡 Connecting WS...")
        streamer.connect()
    except Exception as e:
        print(f"❌ WS: {e}")
        traceback.print_exc()


# ============================================================
# FLASK ENDPOINTS
# ============================================================
@app.route('/health', methods=['GET'])
@cross_origin()
def health():
    return jsonify({
        'status': 'healthy',
        'initial_balance': INITIAL_BALANCE,
        'current_balance': float(paper_trading_bot.balance),
        'models_loaded': {
            'lstm': lstm_model is not None, 'cnn': cnn_model is not None,
            'lstm_pe': lstm_model_pe is not None, 'cnn_pe': cnn_model_pe is not None,
        },
        'cnn_config': {
            'CNN_IGNORE_HOLD': CNN_IGNORE_HOLD,
            'CNN_HOLD_DOMINANCE': CNN_HOLD_DOMINANCE,
            'CNN_MIN_BUY_SELL_EDGE': CNN_MIN_BUY_SELL_EDGE,
        },
        'ce_buffer': len(ce_ohlcv_buffer),
        'pe_buffer': len(pe_ohlcv_buffer),
    }), 200


@app.route('/update', methods=['POST', 'GET'])
@cross_origin()
def update_models():
    global g_current_status
    try:
        print("\n" + "=" * 70)
        print("🔄 /update — reloading all models + scalers")
        print("=" * 70)
        loaded = load_all_models()
        success = [k for k, v in loaded.items() if v]
        failed  = [k for k, v in loaded.items() if not v]
        g_current_status = f"Models reloaded ({len(success)}/4 OK)"
        diag(f"🔄 /update — reloaded: {success} | failed: {failed}")
        return jsonify({
            'message': 'Models reloaded',
            'models_loaded': loaded,
            'success': success,
            'failed': failed,
            'all_ok': len(failed) == 0,
            'timestamp': datetime.datetime.now().isoformat(),
        }), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/diag_log', methods=['GET'])
@cross_origin()
def get_diag_log():
    return jsonify({
        'diag_log': diag_log,
        'last_consensus_debug': last_consensus_debug,
        'current_predictions': model_predictions,
        'cnn_config': {
            'CNN_IGNORE_HOLD': CNN_IGNORE_HOLD,
            'CNN_HOLD_DOMINANCE': CNN_HOLD_DOMINANCE,
            'CNN_MIN_BUY_SELL_EDGE': CNN_MIN_BUY_SELL_EDGE,
        },
        'timestamp': datetime.datetime.now().isoformat(),
    }), 200


@app.route('/current_pattern', methods=['GET'])
@cross_origin()
def current_pattern():
    p = (model_predictions['lstm']['signal'],
         model_predictions['cnn']['signal'],
         model_predictions['lstm_pe']['signal'],
         model_predictions['cnn_pe']['signal'])
    is_bull = (
        (p[0]==1 and p[1]==1 and p[2]==2 and p[3]==2) or
        (p[0]==1 and p[1]==0 and p[2]==2 and p[3]==2) or
        (p[0]==0 and p[1]==1 and p[2]==2 and p[3]==2) or
        (p[0]==1 and p[1]==1 and p[2]==0 and p[3]==2) or
        (p[0]==1 and p[1]==1 and p[2]==2 and p[3]==0)
    )
    is_bear = (
        (p[0]==2 and p[1]==2 and p[2]==1 and p[3]==1) or
        (p[0]==2 and p[1]==0 and p[2]==1 and p[3]==1) or
        (p[0]==0 and p[1]==2 and p[2]==1 and p[3]==1) or
        (p[0]==2 and p[1]==2 and p[2]==0 and p[3]==1) or
        (p[0]==2 and p[1]==2 and p[2]==1 and p[3]==0)
    )
    return jsonify({
        'pattern': list(p),
        'pattern_str': f"{p[0]},{p[1]},{p[2]},{p[3]}",
        'is_bull': is_bull,
        'is_bear': is_bear,
        'matched': is_bull or is_bear,
        'position': paper_trading_bot.position,
        'signals_detail': {
            'lstm':    model_predictions['lstm'],
            'cnn':     model_predictions['cnn'],
            'lstm_pe': model_predictions['lstm_pe'],
            'cnn_pe':  model_predictions['cnn_pe'],
        },
        'timestamp': datetime.datetime.now().isoformat(),
    }), 200


@app.route('/force_test', methods=['POST'])
@cross_origin()
def force_test():
    try:
        data = request.get_json() or {}
        model_predictions['lstm']['signal'] = int(data.get('lstm', 0))
        model_predictions['cnn']['signal'] = int(data.get('cnn', 0))
        model_predictions['lstm_pe']['signal'] = int(data.get('lstm_pe', 0))
        model_predictions['cnn_pe']['signal'] = int(data.get('cnn_pe', 0))

        sig, conf, ms = ensemble_signal_generation()

        if sig in (1, 2):
            paper_trading_bot.execute_paper_trade(sig, ce_price or 100, pe_price or 100, conf, ms)

        return jsonify({
            'injected': data,
            'result_signal': sig,
            'consensus': ms.get('consensus'),
            'reject_reason': ms.get('reject_reason'),
            'position': paper_trading_bot.position,
            'balance': paper_trading_bot.balance,
        }), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/status', methods=['GET'])
@cross_origin()
def status():
    stats = paper_trading_bot.get_performance_stats()
    p = (model_predictions['lstm']['signal'],
         model_predictions['cnn']['signal'],
         model_predictions['lstm_pe']['signal'],
         model_predictions['cnn_pe']['signal'])
    return jsonify({
        'status': g_current_status,
        'ce_price': float(ce_price), 'pe_price': float(pe_price),
        'balance': float(paper_trading_bot.balance),
        'initial_balance': float(paper_trading_bot.initial_balance),
        'performance_stats': stats,
        'current_position': paper_trading_bot.position,
        'entry_price': float(paper_trading_bot.entry_price or 0),
        'target': float(paper_trading_bot.target or 0),
        'stop_loss': float(paper_trading_bot.stop_loss or 0),
        'shares': int(paper_trading_bot.shares),
        'current_pattern': list(p),
        'pattern_str': f"{p[0]},{p[1]},{p[2]},{p[3]}",
        'last_signal': int(last_signal),
        'model_predictions': model_predictions,
        'last_consensus_debug': last_consensus_debug,
        'current_signals': {
            'lstm': {
                'signal': int(model_predictions['lstm']['signal']),
                'signal_label': {1:'BULLISH',2:'BEARISH',0:'HOLD'}.get(model_predictions['lstm']['signal'],'HOLD'),
                'confidence': float(model_predictions['lstm']['confidence']),
                'predicted_price': float(latest_lstm_predicted_price),
                'close_diff_predicted': float(latest_close_diff_predicted),
            },
            'cnn': {
                'signal': int(model_predictions['cnn']['signal']),
                'signal_label': {1:'BULLISH',2:'BEARISH',0:'HOLD'}.get(model_predictions['cnn']['signal'],'HOLD'),
                'confidence': float(model_predictions['cnn']['confidence']),
                'lead': float(model_predictions['cnn'].get('lead', 0.0)),
                'probs': model_predictions['cnn'].get('probs', [0,0,0]),
                'predicted_class': int(model_predictions['cnn'].get('predicted_class', 0)),
                'forced_hold': bool(model_predictions['cnn'].get('forced_hold', False)),
                'hold_dominance': float(model_predictions['cnn'].get('hold_dominance', 0.0)),
                'argmax_class': int(model_predictions['cnn'].get('argmax_class', 0)),
            },
            'lstm_pe': {
                'signal': int(model_predictions['lstm_pe']['signal']),
                'signal_label': {1:'BULLISH',2:'BEARISH',0:'HOLD'}.get(model_predictions['lstm_pe']['signal'],'HOLD'),
                'confidence': float(model_predictions['lstm_pe']['confidence']),
                'predicted_price': float(latest_lstm_pe_predicted_price),
                'close_diff_predicted': float(latest_close_diff_predicted_pe),
            },
            'cnn_pe': {
                'signal': int(model_predictions['cnn_pe']['signal']),
                'signal_label': {1:'BULLISH',2:'BEARISH',0:'HOLD'}.get(model_predictions['cnn_pe']['signal'],'HOLD'),
                'confidence': float(model_predictions['cnn_pe']['confidence']),
                'lead': float(model_predictions['cnn_pe'].get('lead', 0.0)),
                'probs': model_predictions['cnn_pe'].get('probs', [0,0,0]),
                'predicted_class': int(model_predictions['cnn_pe'].get('predicted_class', 0)),
                'forced_hold': bool(model_predictions['cnn_pe'].get('forced_hold', False)),
                'hold_dominance': float(model_predictions['cnn_pe'].get('hold_dominance', 0.0)),
                'argmax_class': int(model_predictions['cnn_pe'].get('argmax_class', 0)),
            },
        },
        'four_model_pattern': f"{p[0]},{p[1]},{p[2]},{p[3]}",
        'last_actionable_signal': int(last_signal),
        'actionable_signal_label': {0:'HOLD',1:'BUY_CE',2:'BUY_PE'}.get(last_signal,'HOLD'),
        'timestamp': datetime.datetime.now().isoformat(),
    }), 200


@app.route('/trade_logs', methods=['GET'])
@cross_origin()
def trade_logs():
    return jsonify({
        'current_status': g_current_status,
        'initial_balance': float(paper_trading_bot.initial_balance),
        'current_balance': float(paper_trading_bot.balance),
        'global_logs': trade_log_global[-50:],
        'recent_trades': recent_trades[-20:],
        'performance_stats': paper_trading_bot.get_performance_stats(),
        'diag_log': diag_log,
        'last_consensus_debug': last_consensus_debug,
        'target_pct': TARGET_PCT * 100,
        'sl_pct': SL_PCT * 100,
    }), 200


@app.route('/reset', methods=['POST'])
@cross_origin()
def reset():
    global paper_trading_bot, trade_log_global, recent_trades, current_position
    global trades_today, current_trade_date, last_consensus_debug
    paper_trading_bot = PaperTradingBot(balance=INITIAL_BALANCE, lot_size=LOT_SIZE)
    trade_log_global = []
    recent_trades = []
    current_position = None
    trades_today = 0
    current_trade_date = None
    last_consensus_debug = {}
    diag_log.clear()
    return jsonify({
        'message': 'reset',
        'initial_balance': INITIAL_BALANCE,
        'current_balance': float(paper_trading_bot.balance),
    }), 200


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 80)
    print("4-MODEL ENSEMBLE  —  HOLD-AWARE CNN v2")
    print(f"💰 STARTING BALANCE = ₹{INITIAL_BALANCE:,}")
    print("=" * 80)
    print("CNN RULE:")
    print(f"  1. If Hold is top class AND Hold >= {CNN_HOLD_DOMINANCE} → HOLD")
    print(f"  2. Else pick max(Buy, Sell) IF edge >= {CNN_MIN_BUY_SELL_EDGE}")
    print(f"  3. Else → HOLD")
    print(f"  predicted_class is ALWAYS synced with signal")
    print("BULL: (1,1,2,2) OR (1,0,2,2) OR (0,1,2,2) OR (1,1,0,2) OR (1,1,2,0)")
    print("BEAR: (2,2,1,1) OR (2,0,1,1) OR (0,2,1,1) OR (2,2,0,1) OR (2,2,1,0)")
    print(f"TARGET = +{TARGET_PCT*100:.2f}%   SL = -{SL_PCT*100:.2f}%")
    print("=" * 80)

    print("\n🤖 Loading models...")
    loaded = load_all_models()
    print(f"✅ {loaded}")

    if any(not v for v in loaded.values()):
        print("❌ Missing models"); sys.exit(1)

    print("\n📊 Fetching history...")
    ce_hist = fetch_historical_ohlcv(INSTRUMENT_TOKEN_CE, days=10)
    pe_hist = fetch_historical_ohlcv(INSTRUMENT_TOKEN_PE, days=10)
    if ce_hist is None or pe_hist is None:
        print("❌ No history"); sys.exit(1)
    print(f"✅ CE={len(ce_hist)} rows, PE={len(pe_hist)} rows")

    ce_ohlcv_buffer = ce_hist.tail(LSTM_LOOKBACK).reset_index().to_dict('records')
    pe_ohlcv_buffer = pe_hist.tail(LSTM_LOOKBACK).reset_index().to_dict('records')
    for r in ce_ohlcv_buffer: r['Datetime'] = pd.Timestamp(r['Datetime'])
    for r in pe_ohlcv_buffer: r['Datetime'] = pd.Timestamp(r['Datetime'])

    ce_price = float(ce_hist['Close'].iloc[-1])
    pe_price = float(pe_hist['Close'].iloc[-1])

    print("\n🔎 Startup sanity check...")
    predict_lstm_ce(ce_hist)
    predict_cnn_ce(ce_hist)
    predict_lstm_pe(pe_hist)
    predict_cnn_pe(pe_hist)
    sig, conf, ms = ensemble_signal_generation()
    print(f"\n📊 Startup consensus: sig={sig}")
    print(f"   reason: {ms.get('reject_reason') or ms.get('consensus')}\n")

    print(f"💰 Starting Balance: ₹{paper_trading_bot.balance:,.2f}\n")

    g_current_status = "Initialized"

    print("🌐 Starting Flask on :5000 ...")
    threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False),
        daemon=True
    ).start()
    time.sleep(2)
    print("✅ Flask up")
    print(f"\n💰 Starting Balance = ₹{INITIAL_BALANCE:,}")
    print(f"📌 TARGET = +{TARGET_PCT*100:.2f}%  |  SL = -{SL_PCT*100:.2f}%\n")
    print("Endpoints:")
    print("   POST /update          — reload all models + scalers from disk")
    print("   GET  /status          — full state")
    print("   GET  /current_pattern — 4-signal tuple + match")
    print("   GET  /diag_log        — rolling log")
    print("   POST /force_test      — inject signals")
    print("   POST /reset           — reset to ₹1,00,000")
    print()

    try:
        start_websocket()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown")
        stats = paper_trading_bot.get_performance_stats()
        print(f"💰 Final Balance: ₹{stats['current_balance']:,.2f}  "
              f"Return: {stats['return_percentage']:+.2f}%")#19% max reacxhed
