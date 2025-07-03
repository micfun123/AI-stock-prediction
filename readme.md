```
           Historical Prices ───► ARIMA ───┐
                                          │
           Time Series Data ───►  LSTM ───┤
                                          │
           Time Series Data ───►  GRU  ───┤
                                          │
  Technical Indicators & Features ─►  RF ─┤
                                          │
   News / Tweets / Headlines ─► Stock-BERT ─┘
                                          ▼
                                Meta Learner: XGBoost
                                  (Final Prediction)
```