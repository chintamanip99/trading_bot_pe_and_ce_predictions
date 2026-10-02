import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';

const App = () => {
  const [status, setStatus] = useState('Connecting...');
  const [tradeLogs, setTradeLogs] = useState([]);
  const [performance, setPerformance] = useState({
    balance: 0,
    totalReturn: 0,
    returnPercentage: 0,
    totalTrades: 0,
    winningTrades: 0,
    losingTrades: 0,
    winRate: 0,
    totalFeesPaid: 0,
  });
  const [prices, setPrices] = useState({ ce: 0, pe: 0, last: 0 });
  const [position, setPosition] = useState(null);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);   // ⬅ NEW toggle

  // ---- 4-MODEL SIGNALS ----
  const [signals, setSignals] = useState({
    lstm:    { signal: 0, label: 'HOLD', confidence: 0, predictedPrice: 0, closeDiff: 0 },
    cnn:     { signal: 0, label: 'HOLD', confidence: 0, predictedClass: 0, lead: 0, probs: [0, 0, 0] },
    lstm_pe: { signal: 0, label: 'HOLD', confidence: 0, predictedPrice: 0, closeDiff: 0 },
    cnn_pe:  { signal: 0, label: 'HOLD', confidence: 0, predictedClass: 0, lead: 0, probs: [0, 0, 0] },
  });
  const [consensus, setConsensus] = useState({
    status: 'NONE',
    pattern: '0,0,0,0',
    lastSignal: 0,
    lastSignalLabel: 'HOLD',
  });

  // ---- Refs for auto-scroll ----
  const logsScrollRef = useRef(null);      // the scrollable container
  const logsEndRef    = useRef(null);      // the bottom anchor
  const userScrolledUpRef = useRef(false); // has the user manually scrolled up?

  // ============================================================
  // FETCH DATA
  // ============================================================
  const fetchData = async () => {
    try {
      // ---- /status ----
      const statusRes = await fetch('http://192.168.29.2:5000/status');
      if (statusRes.ok) {
        const s = await statusRes.json();

        setStatus(s.status || 'Running');
        setPrices({
          ce: s.ce_price || 0,
          pe: s.pe_price || 0,
          last: s.last_price || 0,
        });
        setPosition(s.current_position || null);

        if (s.current_signals) {
          const cs = s.current_signals;
          setSignals({
            lstm: {
              signal: cs.lstm?.signal ?? 0,
              label: cs.lstm?.signal_label ?? 'HOLD',
              confidence: cs.lstm?.confidence ?? 0,
              predictedPrice: cs.lstm?.predicted_price ?? 0,
              closeDiff: cs.lstm?.close_diff_predicted ?? 0,
            },
            cnn: {
              signal: cs.cnn?.signal ?? 0,
              label: cs.cnn?.signal_label ?? 'HOLD',
              confidence: cs.cnn?.confidence ?? 0,
              predictedClass: cs.cnn?.predicted_class ?? 0,
              lead: cs.cnn?.lead ?? 0,
              probs: cs.cnn?.probs ?? [0, 0, 0],
            },
            lstm_pe: {
              signal: cs.lstm_pe?.signal ?? 0,
              label: cs.lstm_pe?.signal_label ?? 'HOLD',
              confidence: cs.lstm_pe?.confidence ?? 0,
              predictedPrice: cs.lstm_pe?.predicted_price ?? 0,
              closeDiff: cs.lstm_pe?.close_diff_predicted ?? 0,
            },
            cnn_pe: {
              signal: cs.cnn_pe?.signal ?? 0,
              label: cs.cnn_pe?.signal_label ?? 'HOLD',
              confidence: cs.cnn_pe?.confidence ?? 0,
              predictedClass: cs.cnn_pe?.predicted_class ?? 0,
              lead: cs.cnn_pe?.lead ?? 0,
              probs: cs.cnn_pe?.probs ?? [0, 0, 0],
            },
          });

          setConsensus({
            status: s.consensus_status ?? 'NONE',
            pattern: s.four_model_pattern ?? '0,0,0,0',
            lastSignal: s.last_actionable_signal ?? 0,
            lastSignalLabel: s.actionable_signal_label ?? 'HOLD',
          });
        }
      }

      // ---- /trade_logs ----
      const logsRes = await fetch('http://192.168.29.2:5000/trade_logs');
      if (logsRes.ok) {
        const d = await logsRes.json();

        if (d.global_logs) setTradeLogs(d.global_logs);

        if (d.performance_stats) {
          setPerformance({
            balance: d.balance || 0,
            totalReturn: d.performance_stats.total_return || 0,
            returnPercentage: d.performance_stats.return_percentage || 0,
            totalTrades: d.performance_stats.total_trades || 0,
            winningTrades: d.performance_stats.winning_trades || 0,
            losingTrades: d.performance_stats.losing_trades || 0,
            winRate: d.performance_stats.win_rate || 0,
            totalFeesPaid: d.performance_stats.total_fees_paid || 0,
          });
        }
      }

      setLoading(false);
    } catch (err) {
      console.error('fetch error:', err);
      setStatus('Error connecting to server');
      setLoading(false);
    }
  };

  // ---- Fetch interval ----
  useEffect(() => {
    fetchData();
    let interval;
    if (autoRefresh) interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [autoRefresh]);

  // ============================================================
  // AUTO-SCROLL LOGIC (FIXED)
  // ============================================================
  // Detect if user is at (or near) the bottom of the log container.
  // If yes → autoScroll ON stays effective.
  // If user manually scrolls up → pause autoScroll until they return to bottom.
  const handleLogsScroll = useCallback(() => {
    const el = logsScrollRef.current;
    if (!el) return;

    const threshold = 40; // px from bottom
    const atBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;

    userScrolledUpRef.current = !atBottom;

    // If user is back at bottom, allow auto-scroll again
    if (atBottom) setAutoScroll(true);
  }, []);

  // Whenever new logs arrive, scroll to bottom IF autoScroll is on
  // AND user hasn't scrolled up.
  useEffect(() => {
    if (!autoScroll) return;
    if (userScrolledUpRef.current) return;

    const el = logsScrollRef.current;
    if (el) {
      // Scroll only the container, not the whole page
      el.scrollTop = el.scrollHeight;
    }
  }, [tradeLogs, autoScroll]);

  // Manual "jump to bottom" button
  const scrollToBottom = () => {
    const el = logsScrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
      userScrolledUpRef.current = false;
      setAutoScroll(true);
    }
  };

  // ============================================================
  // HELPERS
  // ============================================================
  const getSignalColor = (sig) => {
    if (sig === 1) return 'bullish';
    if (sig === 2) return 'bearish';
    return 'neutral';
  };

  const getSignalLabel = (sig) => {
    if (sig === 1) return '📈 BULLISH';
    if (sig === 2) return '📉 BEARISH';
    return '⏸️ HOLD';
  };

  const formatTime = (ts) => {
    if (!ts) return '';
    try {
      return new Date(ts).toLocaleTimeString('en-IN', {
        hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit',
      });
    } catch { return ts; }
  };

  const parseTradeMessage = (message) => {
    if (!message) return { type: 'info', text: '' };
    if (message.includes('BUY') || message.includes('ENTRY')) return { type: 'entry', text: message };
    if (message.includes('EXIT')) return { type: 'exit', text: message };
    if (message.includes('ERROR')) return { type: 'error', text: message };
    return { type: 'info', text: message };
  };

  // ============================================================
  // MODEL CARD
  // ============================================================
  const ModelCard = ({ title, subtitle, data, isCnn }) => (
    <div className={`model-card ${getSignalColor(data.signal)}`}>
      <div className="model-card-header">
        <span className="model-card-title">{title}</span>
        <span className="model-card-sub">{subtitle}</span>
      </div>

      <div className="model-card-signal">
        <span className={`signal-badge ${getSignalColor(data.signal)}`}>
          {getSignalLabel(data.signal)}
        </span>
      </div>

      <div className="model-card-body">
        <div className="model-card-row">
          <span className="row-label">Confidence</span>
          <span className="row-value">{(data.confidence * 100).toFixed(1)}%</span>
        </div>

        {!isCnn && (
          <>
            <div className="model-card-row">
              <span className="row-label">Pred Price</span>
              <span className="row-value">₹{Number(data.predictedPrice || 0).toFixed(2)}</span>
            </div>
            <div className="model-card-row">
              <span className="row-label">ΔPred</span>
              <span className={`row-value ${data.closeDiff > 0 ? 'positive' : data.closeDiff < 0 ? 'negative' : ''}`}>
                {data.closeDiff > 0 ? '+' : ''}{Number(data.closeDiff || 0).toFixed(4)}
              </span>
            </div>
          </>
        )}

        {isCnn && (
          <>
            <div className="model-card-row">
              <span className="row-label">Class</span>
              <span className="row-value">
                {['Hold', 'Buy', 'Sell'][data.predictedClass] || 'Hold'}
              </span>
            </div>
            <div className="model-card-row">
              <span className="row-label">Lead</span>
              <span className="row-value">{Number(data.lead || 0).toFixed(3)}</span>
            </div>
            <div className="model-card-probs">
              {(data.probs || [0, 0, 0]).map((p, i) => (
                <div key={i} className="prob-bar-wrap">
                  <span className="prob-label">{['H', 'B', 'S'][i]}</span>
                  <div className="prob-bar">
                    <div
                      className={`prob-fill prob-${['hold', 'buy', 'sell'][i]}`}
                      style={{ width: `${(p * 100).toFixed(0)}%` }}
                    />
                  </div>
                  <span className="prob-val">{(p * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );

  // ============================================================
  // RENDER
  // ============================================================
  return (
    <div className="app">
      <header className="header">
        <h1>🤖 4-Model Ensemble Trading Bot</h1>
        <div className="header-badge">
          <span className="badge">PAPER TRADING</span>
          <span className="badge">LSTM_CE + CNN_CE + LSTM_PE + CNN_PE</span>
          <span className="badge">STRICT 4/4</span>
        </div>
      </header>

      <div className="main-container">
        {/* ================= DASHBOARD ================= */}
        <div className="dashboard">
          <div className="status-card">
            <h2>📊 Live Status</h2>
            <div className="status-content">
              <div className="status-text">
                <span className="status-label">Status:</span>
                <span className="status-value">{status}</span>
              </div>
              <div className="status-metrics">
                <div className="metric">
                  <span className="metric-label">CE Price</span>
                  <span className="metric-value price">₹{prices.ce.toFixed(2)}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">PE Price</span>
                  <span className="metric-value price">₹{prices.pe.toFixed(2)}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Position</span>
                  <span className={`metric-value position ${position ? 'active' : 'none'}`}>
                    {position ? position.toUpperCase() : 'None'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="performance-card">
            <h2>💰 Performance</h2>
            <div className="performance-grid">
              <div className="perf-item">
                <span className="perf-label">Balance</span>
                <span className="perf-value">₹{performance.balance.toFixed(2)}</span>
              </div>
              <div className="perf-item">
                <span className="perf-label">Return</span>
                <span className={`perf-value ${performance.returnPercentage >= 0 ? 'positive' : 'negative'}`}>
                  {performance.returnPercentage >= 0 ? '+' : ''}{performance.returnPercentage.toFixed(2)}%
                </span>
              </div>
              <div className="perf-item">
                <span className="perf-label">Trades</span>
                <span className="perf-value">{performance.totalTrades}</span>
              </div>
              <div className="perf-item">
                <span className="perf-label">Win Rate</span>
                <span className="perf-value">{performance.winRate.toFixed(1)}%</span>
              </div>
              <div className="perf-item">
                <span className="perf-label">Wins/Losses</span>
                <span className="perf-value">{performance.winningTrades}/{performance.losingTrades}</span>
              </div>
              <div className="perf-item">
                <span className="perf-label">Fees Paid</span>
                <span className="perf-value">₹{performance.totalFeesPaid.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>

        {/* ================= 4 MODEL BOXES ================= */}
        <div className="models-section">
          <div className="models-section-header">
            <h2>🧠 Model Predictions (4-Model Ensemble)</h2>
            <div className="consensus-summary">
              <span className="consensus-pattern-label">Pattern:</span>
              <code className="consensus-pattern">{consensus.pattern}</code>
              <span className={`consensus-status consensus-${consensus.status.toLowerCase()}`}>
                {consensus.status.replace(/_/g, ' ')}
              </span>
            </div>
          </div>

          <div className="models-grid">
            <ModelCard title="LSTM CE" subtitle="lstm_fixed.h5"    data={signals.lstm}    isCnn={false} />
            <ModelCard title="CNN CE"  subtitle="cnn_fixed.h5"     data={signals.cnn}     isCnn={true}  />
            <ModelCard title="LSTM PE" subtitle="lstm_fixed_pe.h5" data={signals.lstm_pe} isCnn={false} />
            <ModelCard title="CNN PE"  subtitle="cnn_fixed_pe.h5"  data={signals.cnn_pe}  isCnn={true}  />
          </div>

          <div className={`consensus-banner consensus-${consensus.status.toLowerCase()}`}>
            <div className="consensus-banner-left">
              <span className="consensus-banner-title">🎯 Final Actionable Signal</span>
              <span className={`consensus-banner-signal ${
                consensus.lastSignal === 1 ? 'bullish' :
                consensus.lastSignal === 2 ? 'bearish' : 'neutral'}`}>
                {consensus.lastSignalLabel}
              </span>
            </div>
            <div className="consensus-banner-right">
              <span className="consensus-rule">BUY CE: L(C)=1, C(C)=1, L(P)=2, C(P)=2</span>
              <span className="consensus-rule">BUY PE: L(C)=2, C(C)=2, L(P)=1, C(P)=1</span>
            </div>
          </div>
        </div>

        {/* ================= TRADE LOGS ================= */}
        <div className="logs-container">
          <div className="logs-header">
            <h2>📋 Trade Logs</h2>
            <div className="logs-controls">
              <button
                className={`auto-refresh-btn ${autoScroll ? 'active' : ''}`}
                onClick={() => setAutoScroll((v) => !v)}
                title={autoScroll ? 'Auto-scroll ON' : 'Auto-scroll OFF'}
              >
                {autoScroll ? '⬇️ Auto-scroll ON' : '⏸️ Auto-scroll OFF'}
              </button>
              <button className="auto-refresh-btn" onClick={scrollToBottom}>
                ⤓ Jump to Latest
              </button>
              <button
                className={`auto-refresh-btn ${autoRefresh ? 'active' : ''}`}
                onClick={() => setAutoRefresh(!autoRefresh)}
              >
                {autoRefresh ? '⏸️ Pause' : '▶️ Resume'}
              </button>
              <span className="log-count">{tradeLogs.length} entries</span>
            </div>
          </div>

          {/* ⭐ scrollable container with onScroll handler and ref */}
          <div
            className="logs-scroll"
            ref={logsScrollRef}
            onScroll={handleLogsScroll}
          >
            {loading ? (
              <div className="loading">Loading trades...</div>
            ) : tradeLogs.length === 0 ? (
              <div className="no-logs">No trade logs available</div>
            ) : (
              tradeLogs.map((log, idx) => {
                const parsed = parseTradeMessage(log.message || log);
                return (
                  <div key={idx} className={`log-entry ${parsed.type}`}>
                    <span className="log-time">{formatTime(log.timestamp)}</span>
                    <span className="log-message">{parsed.text}</span>
                  </div>
                );
              })
            )}
            <div ref={logsEndRef} />
          </div>
        </div>
      </div>

      <footer className="footer">
        <span>⚡ Dynamic Target/SL based on |ΔPred|</span>
        <span>🔄 Auto-refresh: {autoRefresh ? 'ON' : 'OFF'}</span>
        <span>📜 Auto-scroll: {autoScroll ? 'ON' : 'OFF'}</span>
        <span>📡 Server: 192.168.29.2:5000</span>
      </footer>
    </div>
  );
};

export default App;
