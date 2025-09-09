### https://github.com/Jathon-sss/kis-omega/ ###

### 1. Python/Anaconda 환경
```powershell
conda create -n kisomega python=3.11 -y
'conda activate kisomega'

## 실행 순서
# 1. 뉴스 수집 (스크래핑)
'python -m kis_omega.news.fetch'

결과: src/data/news_raw/latest.parquet

# 2. 감정 분석 (토픽별)
모든 토픽:
'python -m kis_omega.news.enrich_topics'

특정 토픽(예: UNH):
'python -m kis_omega.news.enrich_topics --topic unh'

결과:
src/data/news/<topic>/latest_rows.parquet (기사별 점수)
src/data/news/<topic>/index_history.parquet (날짜별 지수)

# 3. 주가 데이터 다운로드
'python -m kis_omega.scripts.download_prices'

결과: src/data/prices/UNH.csv, src/data/prices/SRPT.csv

# 4. 뉴스 + 주가 결합
'python -m kis_omega.features.merge_features'

결과: src/data/features/UNH_features.csv, src/data/features/SRPT_features.csv

# 5. 모델 학습/예측
'python -m kis_omega.models.train_predict'

결과: 콘솔에 예측 성능 리포트 출력