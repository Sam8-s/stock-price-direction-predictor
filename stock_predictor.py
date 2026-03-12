import os
import pandas as pd
import pandas_ta as ta
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def predict_stock(company):

    stocks_path = os.path.expanduser(
        "~/.cache/kagglehub/datasets/jacksoncrow/stock-market-dataset/versions/2/stocks"
    )

    file_path = os.path.join(stocks_path, company + ".csv")

    if not os.path.exists(file_path):
        return None, None, None

    df = pd.read_csv(file_path)

    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["SMA_50"] = ta.sma(df["Close"], length=50)

    df["EMA_20"] = ta.ema(df["Close"], length=20)
    df["EMA_50"] = ta.ema(df["Close"], length=50)

    df["RSI"] = ta.rsi(df["Close"], length=14)

    macd = ta.macd(df["Close"])
    df["MACD"] = macd["MACD_12_26_9"]

    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"])

    df["ROC"] = ta.roc(df["Close"])

    df["Return_1"] = df["Close"].pct_change(1)
    df["Return_2"] = df["Close"].pct_change(2)
    df["Return_3"] = df["Close"].pct_change(3)

    df["Return"] = df["Close"].pct_change().shift(-1)

    df["Target"] = 0
    df.loc[df["Return"] > 0.01, "Target"] = 1

    df.dropna(inplace=True)

    features = [
        "SMA_20","SMA_50",
        "EMA_20","EMA_50",
        "RSI","MACD","ATR","ROC",
        "Return_1","Return_2","Return_3",
        "Volume"
    ]

    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    latest = X.iloc[[-1]]

    latest_scaled = scaler.transform(latest)

    prediction = model.predict(latest_scaled)[0]

    importance = pd.Series(model.feature_importances_, index=features)

    return prediction, accuracy, importance