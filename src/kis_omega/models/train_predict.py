# src/kis_omega/models/train_predict.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[2]
FEAT_DIR = ROOT / "data" / "features"

def train_symbol(symbol: str):
    path = FEAT_DIR / f"{symbol}_features.csv"
    df = pd.read_csv(path)

    X = df[[f"{symbol}_sent"]]
    y = df[f"{symbol}_target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"📊 Results for {symbol}")
    print(classification_report(y_test, y_pred, digits=3))

def main():
    train_symbol("UNH")
    train_symbol("SRPT")

if __name__ == "__main__":
    main()
