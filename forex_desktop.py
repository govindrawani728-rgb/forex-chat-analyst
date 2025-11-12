# forex_desktop.py
# Simple desktop Forex Chat Analyst using PySimpleGUI + yfinance + matplotlib
# Features:
# - Fetch historical FX data from Yahoo Finance
# - Compute EMA, RSI, MACD, Supertrend
# - Show candlestick-like chart (matplotlib) with EMA and buy/sell markers
# - Chat box: simple rule-based answers (buy/sell, SL/TP, trend, risk-lot)
# - Manual Refresh button (press to update). Works cross-platform.

import threading
import io
import math
from datetime import datetime
import traceback

import numpy as np
import pandas as pd
import yfinance as yf
import PySimpleGUI as sg
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ------------------ Indicators ------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    line = ema_fast - ema_slow
    signal_line = ema(line, signal)
    hist = line - signal_line
    return line, signal_line, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([ (high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs() ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    hl2 = (df["High"] + df["Low"]) / 2.0
    _atr = atr(df, period)
    upperband = hl2 + multiplier * _atr
    lowerband = hl2 - multiplier * _atr
    st_upper = pd.Series(index=df.index, dtype="float64")
    st_lower = pd.Series(index=df.index, dtype="float64")
    trend = pd.Series(index=df.index, dtype="int64")
    for i in range(len(df)):
        if i == 0:
            st_upper.iat[i] = upperband.iat[i]
            st_lower.iat[i] = lowerband.iat[i]
            trend.iat[i] = 1
            continue
        st_upper.iat[i] = upperband.iat[i] if (upperband.iat[i] < st_upper.iat[i-1] or df["Close"].iat[i-1] > st_upper.iat[i-1]) else st_upper.iat[i-1]
        st_lower.iat[i] = lowerband.iat[i] if (lowerband.iat[i] > st_lower.iat[i-1] or df["Close"].iat[i-1] < st_lower.iat[i-1]) else st_lower.iat[i-1]
        if df["Close"].iat[i] > st_upper.iat[i-1]:
            trend.iat[i] = 1
        elif df["Close"].iat[i] < st_lower.iat[i-1]:
            trend.iat[i] = -1
        else:
            trend.iat[i] = trend.iat[i-1]
        if trend.iat[i] == 1 and st_lower.iat[i] < st_lower.iat[i-1]:
            st_lower.iat[i] = st_lower.iat[i-1]
        if trend.iat[i] == -1 and st_upper.iat[i] > st_upper.iat[i-1]:
            st_upper.iat[i] = st_upper.iat[i-1]
    st_line = pd.Series(np.where(trend == 1, st_lower, st_upper), index=df.index, name="Supertrend")
    return st_line, trend

# ------------------ Strategy / Signals ------------------

def compute_indicators(df: pd.DataFrame, params):
    df = df.copy()
    df["EMA_fast"] = ema(df["Close"], params["ema_fast"])
    df["EMA_slow"] = ema(df["Close"], params["ema_slow"])
    df["RSI"] = rsi(df["Close"], params["rsi_len"])
    macd_line, macd_signal, macd_hist = macd(df["Close"], params["macd_fast"], params["macd_slow"], params["macd_signal"])
    df["MACD"] = macd_line
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist
    st_line, st_trend = supertrend(df, params["st_len"], params["st_mult"])
    df["Supertrend"] = st_line
    df["ST_trend"] = st_trend
    df["EMA_cross"] = (df["EMA_fast"] > df["EMA_slow"]).astype(int) * 2 - 1
    df["MACD_cross"] = (df["MACD"] > df["MACD_signal"]).astype(int) * 2 - 1
    return df.dropna()

def generate_signals(df: pd.DataFrame, params):
    d = df.copy()
    buy = (
        (d["EMA_fast"].shift(1) <= d["EMA_slow"].shift(1)) & (d["EMA_fast"] > d["EMA_slow"])
        & (d["RSI"] > params["rsi_buy"]) & (d["MACD"] > d["MACD_signal"]) & (d["ST_trend"] == 1)
    )
    sell = (
        (d["EMA_fast"].shift(1) >= d["EMA_slow"].shift(1)) & (d["EMA_fast"] < d["EMA_slow"])
        & (d["RSI"] < params["rsi_sell"]) & (d["MACD"] < d["MACD_signal"]) & (d["ST_trend"] == -1)
    )
    d["Signal"] = np.where(buy, 1, np.where(sell, -1, 0))
    return d

def summarize_bias(row):
    votes = 0
    votes += 1 if row["EMA_cross"] > 0 else -1
    votes += 1 if row["MACD_cross"] > 0 else -1
    votes += 1 if row["RSI"] >= 50 else -1
    votes += 1 if row["ST_trend"] == 1 else -1
    if votes >= 3:
        return "Strong Bullish"
    if votes == 2:
        return "Bullish"
    if votes == -2:
        return "Bearish"
    if votes <= -3:
        return "Strong Bearish"
    return "Neutral"

def pip_size(sym):
    return 0.01 if "JPY" in sym.upper() else 0.0001

def position_size(balance, risk_pct, stop_pips, pip_value):
    if stop_pips <= 0 or pip_value <= 0:
        return 0.0
    risk_amount = balance * (risk_pct/100.0)
    lots = risk_amount / (stop_pips * pip_value)
    return max(lots, 0.0)

# ------------------ Data fetch ------------------

def fetch_data(symbol, period="7d", interval="15m"):
    try:
        df = yf.download(tickers=symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        if df.empty:
            return None
        df = df.dropna().copy()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print("Fetch error:", e)
        return None

# ------------------ GUI Helpers ------------------

def draw_figure(canvas, figure):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

# ------------------ Chat / Analyst ------------------

def chat_answer(query, latest, params, balance, risk_pct, stop_pips, pip_val, symbol):
    q = (query or "").lower()
    bias = summarize_bias(latest)
    price = latest["Close"]
    pip = pip_size(symbol)
    sl_long = price - (stop_pips * pip)
    tp_long = price + (2 * stop_pips * pip)
    sl_short = price + (stop_pips * pip)
    tp_short = price - (2 * stop_pips * pip)
    if "buy" in q or "long" in q:
        ok = (latest["EMA_fast"]>latest["EMA_slow"]) and (latest["MACD"]>latest["MACD_signal"]) and (latest["ST_trend"]==1) and (latest["RSI"]>params["rsi_buy"])
        verdict = "Setup aligns for LONG ✅" if ok else "Weak/No LONG setup ⚠️"
        return f"{verdict}\nBias: {bias}\nLONG entry ≈ {price:.5f}\nSL ≈ {sl_long:.5f}\nTP ≈ {tp_long:.5f}"
    if "sell" in q or "short" in q:
        ok = (latest["EMA_fast"]<latest["EMA_slow"]) and (latest["MACD"]<latest["MACD_signal"]) and (latest["ST_trend"]==-1) and (latest["RSI"]<params["rsi_sell"])
        verdict = "Setup aligns for SHORT ✅" if ok else "Weak/No SHORT setup ⚠️"
        return f"{verdict}\nBias: {bias}\nSHORT entry ≈ {price:.5f}\nSL ≈ {sl_short:.5f}\nTP ≈ {tp_short:.5f}"
    if "rsi" in q:
        return f"RSI: {latest['RSI']:.1f} — >70 overbought, <30 oversold. Bias: {bias}"
    if "stop" in q or "sl" in q:
        return f"SL idea: ~{stop_pips} pips. Long SL ≈ {sl_long:.5f}; Short SL ≈ {sl_short:.5f}"
    if "tp" in q or "target" in q:
        return f"TP example (1:2 RR): Long TP ≈ {tp_long:.5f}; Short TP ≈ {tp_short:.5f}"
    if "trend" in q:
        return f"Supertrend: {'UP' if latest['ST_trend']==1 else 'DOWN'}; Bias: {bias}"
    if "risk" in q or "lot" in q or "position" in q:
        lots = position_size(balance, risk_pct, stop_pips, pip_val)
        return f"Suggested position ≈ {lots:.2f} lots at {risk_pct}% risk with {stop_pips} pip stop."
    # default
    return f"Price {price:.5f} | Bias: {bias}\nEMA({params['ema_fast']}/{params['ema_slow']}): {'bull' if latest['EMA_fast']>latest['EMA_slow'] else 'bear'} | RSI: {latest['RSI']:.1f}"

# ------------------ Main GUI ------------------

def run_app():
    global params
    sg.theme("DarkBlue3")
    layout = [
        [sg.Text("Forex Chat Analyst — Desktop", font=("Helvetica", 14))],
        [sg.Text("Symbol (Yahoo):"), sg.Input("EURUSD=X", key="-SYMBOL-", size=(12,1)),
         sg.Text("Interval:"), sg.Combo(["1m","2m","5m","15m","30m","60m","1h","4h","1d"], default_value="15m", key="-INTERVAL-"),
         sg.Text("History:"), sg.Combo(["1d","2d","5d","10d","30d","60d","3mo"], default_value="7d", key="-PERIOD-"),
         sg.Button("Refresh", key="-REFRESH-"), sg.Button("Auto Refresh:Off", key="-AUTO-")],
        [sg.Frame(layout=[
            [sg.Text("EMA Fast"), sg.Input("9", key="-EMA_FAST-", size=(5,1)),
             sg.Text("EMA Slow"), sg.Input("21", key="-EMA_SLOW-", size=(5,1)),
             sg.Text("RSI len"), sg.Input("14", key="-RSI_LEN-", size=(5,1))],
            [sg.Text("MACD Fast"), sg.Input("12", key="-MACD_F-", size=(5,1)),
             sg.Text("MACD Slow"), sg.Input("26", key="-MACD_S-", size=(5,1)),
             sg.Text("MACD Sig"), sg.Input("9", key="-MACD_SIG-", size=(5,1))],
            [sg.Text("ST ATR len"), sg.Input("10", key="-ST_LEN-", size=(5,1)),
             sg.Text("ST Mult"), sg.Input("3.0", key="-ST_MULT-", size=(5,1))]
        ], title="Strategy Params")],
        [sg.Frame(layout=[
            [sg.Text("Balance"), sg.Input("5000", key="-BAL-", size=(8,1)),
             sg.Text("Risk %"), sg.Input("1.0", key="-RISK-", size=(6,1)),
             sg.Text("Stop pips"), sg.Input("30", key="-STOP-", size=(6,1)),
             sg.Text("Pip value"), sg.Input("10", key="-PIPVAL-", size=(6,1)),
             sg.Button("Calc Lots", key="-CALCLOTS-"), sg.Text("", key="-LOTSOUT-")]
        ], title="Risk")],
        [sg.Canvas(key="-CANVAS-")],
        [sg.Multiline("", size=(70,6), key="-CHATLOG-"), sg.Column([
            [sg.Input("", key="-CHATQ-", size=(30,1)), sg.Button("Ask", key="-ASK-")],
            [sg.Text("Status:"), sg.Text("", key="-STATUS-")]
        ])]
    ]

    window = sg.Window("Forex Chat Analyst", layout, finalize=True, element_justification="left", resizable=True)

    canvas_elem = window["-CANVAS-"]
    canvas = canvas_elem.TKCanvas

    figure_agg = None
    auto_refresh = False
    stop_auto = threading.Event()

    def update_chart(symbol, period, interval, params):
        try:
            window["-STATUS-"].update("Fetching...")
            df = fetch_data(symbol, period, interval)
            if df is None:
                window["-STATUS-"].update("No data returned.")
                return None, None
            dfi = compute_indicators(df, params)
            dfs = generate_signals(dfi, params)
            latest = dfs.iloc[-1]
            # Plot
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(dfs.index, dfs["Close"], label="Close")
            ax.plot(dfs.index, dfs["EMA_fast"], label=f"EMA{params['ema_fast']}")
            ax.plot(dfs.index, dfs["EMA_slow"], label=f"EMA{params['ema_slow']}")
            ax.plot(dfs.index, dfs["Supertrend"], label="Supertrend", linewidth=1, alpha=0.7)
            buys = dfs[dfs["Signal"]==1]
            sells = dfs[dfs["Signal"]==-1]
            if not buys.empty:
                ax.scatter(buys.index, buys["Close"], marker="^", color="green", label="Buy", zorder=5)
            if not sells.empty:
                ax.scatter(sells.index, sells["Close"], marker="v", color="red", label="Sell", zorder=5)
            ax.set_title(f"{symbol} {interval} (last {len(dfs)} bars)")
            ax.legend(loc="upper left", fontsize="small")
            ax.grid(True)
            fig.tight_layout()
            window["-STATUS-"].update("Updated: " + datetime.now().strftime("%H:%M:%S"))
            return fig, dfs
        except Exception as e:
            tb = traceback.format_exc()
            window["-STATUS-"].update("Error: see console")
            print(tb)
            return None, None

    def auto_refresh_thread():
        while not stop_auto.is_set():
            try:
                sg.popup_no_wait("Auto-refreshing...", keep_on_top=False)
            except Exception:
                pass
            # trigger a Refresh event
            window.write_event_value("-DO_REFRESH-", "")
            stop_auto.wait(10)  # default 10 sec between auto updates

    current_fig = None
    current_dfs = None

    while True:
        event, values = window.read(timeout=100)
        if event == sg.WIN_CLOSED:
            break

        if event == "-REFRESH-" or event == "-DO_REFRESH-":
            # gather params
            symbol = values["-SYMBOL-"].strip()
            interval = values["-INTERVAL-"]
            period = values["-PERIOD-"]
            params = {
                "ema_fast": int(values["-EMA_FAST-"]),
                "ema_slow": int(values["-EMA_SLOW-"]),
                "rsi_len": int(values["-RSI_LEN-"]),
                "macd_fast": int(values["-MACD_F-"]),
                "macd_slow": int(values["-MACD_S-"]),
                "macd_signal": int(values["-MACD_SIG-"]),
                "st_len": int(values["-ST_LEN-"]),
                "st_mult": float(values["-ST_MULT-"]),
                "rsi_buy": 35,
                "rsi_sell": 65
            }
            fig, dfs = update_chart(symbol, period, interval, params)
            if fig is not None:
                # draw
                if current_fig:
                    plt.close(current_fig)
                current_fig = fig
                if figure_agg:
                    # destroy previous
                    for child in canvas.winfo_children():
                        child.destroy()
                figure_agg = draw_figure(canvas, fig)
                current_dfs = dfs

        if event == "-CALCLOTS-":
            try:
                bal = float(values["-BAL-"])
                risk_pct = float(values["-RISK-"])
                stop_pips = float(values["-STOP-"])
                pip_val = float(values["-PIPVAL-"])
                lots = position_size(bal, risk_pct, stop_pips, pip_val)
                window["-LOTSOUT-"].update(f"{lots:.2f} lots")
            except Exception as e:
                window["-LOTSOUT-"].update("Err")

        if event == "-ASK-":
            q = values["-CHATQ-"]
            if current_dfs is None:
                window["-CHATLOG-"].print("No data loaded. Press Refresh first.")
            else:
                latest = current_dfs.iloc[-1]
                try:
                    bal = float(values["-BAL-"])
                    risk_pct = float(values["-RISK-"])
                    stop_pips = float(values["-STOP-"])
                    pip_val = float(values["-PIPVAL-"])
                except Exception:
                    bal, risk_pct, stop_pips, pip_val = 5000, 1.0, 30.0, 10.0
                ans = chat_answer(q, latest, params, bal, risk_pct, stop_pips, pip_val, values["-SYMBOL-"])
                window["-CHATLOG-"].print(f"> {q}\n{ans}\n---")

        if event == "-AUTO-":
            if not auto_refresh:
                # start auto thread
                stop_auto.clear()
                t = threading.Thread(target=auto_refresh_thread, daemon=True)
                t.start()
                auto_refresh = True
                window["-AUTO-"].update("Auto Refresh:On")
            else:
                stop_auto.set()
                auto_refresh = False
                window["-AUTO-"].update("Auto Refresh:Off")

    try:
        if figure_agg:
            figure_agg.get_tk_widget().forget()
            plt.close(current_fig)
    except Exception:
        pass
    window.close()

if __name__ == "__main__":
    run_app()
