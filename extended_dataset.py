import pandas as pd
import yfinance as yf


START = "2016-01-01"
END   = "2025-01-01"
SYMBOL = "BTC-USD"

SAR_RATE = 3.75   # USD -> SAR kur çarpanı (istersen değiştir)

OUT_CSV = "dc_extended.csv"

# --- Veri indir ---
data = yf.download(
    SYMBOL,
    start=START,
    end=END,
    interval="1d",
    auto_adjust=False,
)

# index = Date, columns = Open, High, Low, Close, Adj Close, Volume
data = data.reset_index()

# --- rename columns as USD ---
df = data.rename(
    columns={
        "Date": "date",
        "Open": "open_USD",
        "High": "high_USD",
        "Low": "low_USD",
        "Close": "close_USD",
        "Volume": "volume",
    }
)


df = df[["date", "open_USD", "high_USD", "low_USD", "close_USD", "volume"]]

# --- SAR kolonlarını USD'den türet ---
df["open_SAR"]  = df["open_USD"]  * SAR_RATE
df["high_SAR"]  = df["high_USD"]  * SAR_RATE
df["low_SAR"]   = df["low_USD"]   * SAR_RATE
df["close_SAR"] = df["close_USD"] * SAR_RATE

#Cols
df = df[
    [
        "date",
        "open_SAR",
        "open_USD",
        "high_SAR",
        "high_USD",
        "low_SAR",
        "low_USD",
        "close_SAR",
        "close_USD",
        "volume",
    ]
]

df = df.sort_values("date").reset_index(drop=True)

# --- save csv format ---
df.to_csv(OUT_CSV, index=False)

print(f"Saved extended dataset to {OUT_CSV}")
print(df.head())
print(df.tail())
