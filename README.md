# 📈 KOSPI 다음날 종가 예측 머신러닝 프로젝트  
**S&P500, VIX, KOSPI 지표를 활용한 시계열 기반 주가 예측 모델**

---

## 📌 프로젝트 개요  
이 프로젝트는 글로벌 금융 지표(S&P500, VIX)와 한국 증시 지표(KOSPI)를 활용해  
**KOSPI의 다음 날 종가를 예측하는 머신러닝 모델**을 만드는 것을 목표로 합니다.  

데이터 전처리 → 모델링 → 평가 → 특성 중요도 분석까지  
전체 머신러닝 파이프라인을 직접 구성했습니다.

---

## 기술 스택
- Python
- Pandas, NumPy
- Scikit-learn (RandomForestClassifier, GridSearchCV 등)
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 📊 사용 데이터  
| 변수명 | 설명 |
|--------|------|
| `S&P500` | 미국 주요 주가지수 |
| `VIX` | 시장 변동성 지수 |
| `KOSPI_open` | 한국 증시 시가 |
| `KOSPI_close` | 한국 증시 종가 |
| `변동량` | 시가 대비 종가 변동 값 |

데이터는 하루 단위로 정렬되며, **5일 단위 입력 → 다음날 종가 예측** 구조로 변환했습니다.

---

## 🧹 데이터 전처리  
- 결측치 제거 및 보정  
- MinMax Scaling 적용  
- 슬라이딩 윈도우 방식으로 5일 묶음 구조 생성  
- 학습 / 검증 / 테스트 세트 분할  

---

## 🧠 모델 아키텍처  
Dense 기반의 신경망 모델을 사용했습니다.

Input (5일 × N개 변수) 
↓
Dense(64) + ReLU
↓
Dense(32) + ReLU
↓
Dropout(0.2)
↓
Dense(1) → 다음날 KOSPI 종가

---

## 🏋️ 모델 학습 코드  

```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)
```

---

## 성능 평가

test_loss = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)

- Loss: MSE 기준
- 모델의 성능을 테스트 데이터셋에서 최종적으로 평가

---

## 개선 방향
- LSTM / GRU 기반 시계열 모델로 확장
- Hyperparameter Tuning 자동화
- 뉴스, 금리, 경제지표 등 외부 변수 추가
- 더 짧은 기간의 데이터 활용

---

## 폴더 구조
```
project/
 ├── data/
 │    ├── kospi_20y_daily.csv
 │    ├── kospi_data_final_after_lagged.csv
 ├── images/
 │    ├── result_arima.png
 ├── notebooks/
 └── README.md
```
