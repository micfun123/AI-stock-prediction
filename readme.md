```
          Historical Prices ───► ARIMA ───┐
                                          │
           Time Series Data ───►  LSTM ───┤
                                          │
           Time Series Data ───►  GRU  ───┤
                                          │
  Technical Indicators & Features ─►  RF ─┤
                                          │
                                          ▼
                                Meta Learner: XGBoost
                                  (Final Prediction)
```
