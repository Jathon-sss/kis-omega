# src/kis_omega/scripts/train_model_samsung.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def main():
    # ==============================
    # 1) 데이터 불러오기
    # ==============================
    df = pd.read_csv("data/samsung_features.csv")

    # 입력(X), 라벨(y) 분리
    feature_cols = ["ma5", "ma20", "ma60", "vol_change", "return", "rsi14"]
    X = df[feature_cols]
    y = df["label"]

    # ==============================
    # 2) 학습/테스트 분할
    # ==============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # ==============================
    # 3) 모델 학습
    # ==============================
    print("▶ Logistic Regression 학습")
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)

    print("▶ Random Forest 학습")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # ==============================
    # 4) 성능 평가
    # ==============================
    for name, model in [("Logistic Regression", logreg), ("Random Forest", rf)]:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n{name} 정확도: {acc:.3f}")
        print(classification_report(y_test, y_pred))

    # ==============================
    # 5) 모델 저장
    # ==============================
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/samsung_rf.pkl")
    joblib.dump(logreg, "models/samsung_logreg.pkl")
    print("\n[OK] 학습된 모델 저장 완료 (models/ 폴더)")

if __name__ == "__main__":
    main()
