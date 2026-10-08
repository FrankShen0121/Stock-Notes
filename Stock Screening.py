# ============================================================
# 台股選股（六條件） + GitHub Pages 網頁生成（支援前端點擊匯出 Excel）
# ============================================================
#
# 【選股條件】
# 1. 6個月績效 < 50%
# 2. RSI(14) > 50
# 3. ADX(14) > 20
# 4. MA50 > MA200
# 5. 今日成交量 > 30日平均成交量 且 30日平均成交量 > 2,000張
# 6. 股價 < MA20 × 0.99
# ============================================================

import os
import requests
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# 選股參數設定
# ============================================================

RSI_PERIOD = 14
RSI_MIN = 50

ADX_PERIOD = 14
ADX_MIN = 20

MA20_PERIOD = 20
MA50_PERIOD = 50
MA200_PERIOD = 200

VOLUME_PERIOD = 30
MIN_AVG_VOLUME_SHARES = 2_000_000  # 2,000 張 = 2,000,000 股

SIX_MONTH_DAYS = 126  # 約126個交易日
MAX_6M_RETURN = 50

MA20_FACTOR = 0.99  # 股價 < MA20 × 0.99

BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 3

program_start = time.time()

print()
print("=" * 110)
print(
    "★★★★★ 台股選股（六條件）+ GitHub Pages 網頁生成（前端點擊匯出 Excel） ★★★★★"
)
print("=" * 110)


# ============================================================
# 技術指標計算函數
# ============================================================


def calculate_rsi(close, period=14):
    close = pd.Series(close, dtype="float64")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(avg_loss != 0, 100)
    return rsi


def calculate_adx(high, low, close, period=14):
    high = pd.Series(high, dtype="float64")
    low = pd.Series(low, dtype="float64")
    close = pd.Series(close, dtype="float64")

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0),
        index=high.index,
        dtype="float64",
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0),
        index=high.index,
        dtype="float64",
    )

    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_dm_smoothed = plus_dm.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    minus_dm_smoothed = minus_dm.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()

    plus_di = 100 * plus_dm_smoothed / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_smoothed / atr.replace(0, np.nan)

    denominator = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / denominator.replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    return adx


def download_batch(batch_symbols, retries=MAX_RETRIES):
    for attempt in range(1, retries + 1):
        try:
            data = yf.download(
                tickers=batch_symbols,
                period="1y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker",
                timeout=30,
            )
            if data is not None and not data.empty:
                return data
        except Exception as e:
            if attempt == retries:
                print(f"批次下載失敗：{e}")
        if attempt < retries:
            time.sleep(RETRY_DELAY)
    return None


# ============================================================
# 1. 取得台股股票清單 (FinMind API)
# ============================================================

print("\n【1/6】正在取得台股股票清單...")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
try:
    response = requests.get(
        FINMIND_URL, params={"dataset": "TaiwanStockInfo"}, timeout=30
    )
    response.raise_for_status()
    json_data = response.json()
except Exception as e:
    raise RuntimeError(f"取得 FinMind 股票清單失敗：{e}")

if "data" not in json_data:
    raise RuntimeError("FinMind 沒有回傳股票資料")

stocks = pd.DataFrame(json_data["data"])
stocks["stock_id"] = stocks["stock_id"].astype(str)

# 篩選 上市/上櫃、排除 ETF、4碼股票
stocks = stocks[stocks["type"].isin(["twse", "tpex"])].copy()
stocks = stocks[
    stocks["industry_category"].fillna("").str.upper() != "ETF"
].copy()
stocks = stocks[stocks["stock_id"].str.match(r"^\d{4}$")].copy()
stocks = stocks.drop_duplicates(subset=["stock_id"]).reset_index(drop=True)

# 建立 Yahoo Finance Tickers
stocks["symbol"] = np.where(
    stocks["type"].eq("twse"),
    stocks["stock_id"] + ".TW",
    np.where(stocks["type"].eq("tpex"), stocks["stock_id"] + ".TWO", None),
)
stocks = stocks.dropna(subset=["symbol"]).reset_index(drop=True)
stock_info = (
    stocks[["stock_id", "stock_name", "type", "symbol"]]
    .copy()
    .set_index("symbol")
)

print(f"上市 + 上櫃普通股票：{len(stocks):,} 檔")


# ============================================================
# 2. 批次下載 Yahoo Finance 資料
# ============================================================

print("\n【2/6】正在批次下載 Yahoo Finance 股價資料...")

symbols = stocks["symbol"].tolist()
all_data = {}
download_start = time.time()

for start in range(0, len(symbols), BATCH_SIZE):
    batch = symbols[start : start + BATCH_SIZE]
    batch_data = download_batch(batch)

    if batch_data is None or batch_data.