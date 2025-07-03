```
           Historical Prices ───► ARIMA ───┐
                                          │
           Time Series Data ───►  LSTM ───┤
                                          │
           Time Series Data ───►  GRU  ───┤
                                          │
  Technical Indicators & Features ─►  RF ─┤
                                          │
                    unsure ─► Stock-BERT ─┘
                                          ▼
                                Meta Learner: XGBoost
                                  (Final Prediction)
```
