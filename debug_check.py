import joblib, numpy as np, pandas as pd

FEATURES_32 = [
    "Gold_Price","Silver","Platinum","Copper",
    "SP500","DXY","JPY_USD","GBP_USD","CNY_USD","US_2Y_Yield","TIP_ETF","Real_Rate_10Y",
    "PPI","Treasury_Spread","Unemployment","GDP_Growth","Industrial_Prod","Consumer_Conf",
    "Return_7d","Return_14d","Return_30d","MACD","MACD_Hist","Volatility_7d","Volatility_30d",
    "Momentum_14d","Gold_Silver_Ratio","Gold_Oil_Ratio","Gold_VIX_interaction",
    "Rolling_std_7","Price_diff","Month_cos",
]

DEFAULTS = {
    "Gold_Price":3320.00,"Silver":32.80,"Platinum":988.00,"Copper":4.62,
    "SP500":5610.00,"DXY":100.20,"JPY_USD":0.0065,"GBP_USD":1.3260,"CNY_USD":0.1380,
    "US_2Y_Yield":3.98,"TIP_ETF":109.20,"Real_Rate_10Y":1.92,"PPI":2.40,
    "Treasury_Spread":0.42,"Unemployment":4.20,"GDP_Growth":2.40,
    "Industrial_Prod":103.10,"Consumer_Conf":97.20,
    "Return_7d":0.016,"Return_14d":0.027,"Return_30d":0.048,
    "MACD":8.40,"MACD_Hist":2.90,"Volatility_7d":0.0082,"Volatility_30d":0.0095,
    "Momentum_14d":42.00,"Gold_Silver_Ratio":101.22,"Gold_Oil_Ratio":44.27,
    "Gold_VIX_interaction":55776.0,"Rolling_std_7":15.20,"Price_diff":9.80,"Month_cos":0.5,
}

row = [DEFAULTS[f] for f in FEATURES_32]
X   = np.array(row).reshape(1,-1)
X_df = pd.DataFrame(X, columns=FEATURES_32)

print("Gold_Price in input:", X_df["Gold_Price"].values[0])
print()

# Test lasso
pca   = joblib.load('models/pca_transformer.pkl')
lasso = joblib.load('models/lasso_model.pkl')
X_pca = pca.transform(X_df)
print("PCA output shape:", X_pca.shape)
print("PCA output values:", X_pca[0].round(4))
pred_lasso = lasso.predict(X_pca)[0]
print("Lasso prediction:", pred_lasso)
print()

# Test ridge_pca
ridge_pca = joblib.load('models/ridge_model.pkl')
pred_ridge_pca = ridge_pca.predict(X_pca)[0]
print("Ridge PCA prediction:", pred_ridge_pca)
print()

# Test scaler + ridge_gold
scaler     = joblib.load('models/scaler.pkl')
ridge_gold = joblib.load('models/ridge_gold.pkl')
X_scaled   = scaler.transform(X_df)
print("Scaler output (first 5):", X_scaled[0][:5].round(4))
pred_ridge_gold = ridge_gold.predict(X_scaled)[0]
print("Ridge Gold prediction:", pred_ridge_gold)
print()

# Test xgboost
xgb = joblib.load('models/xgboost_gold.pkl')
pred_xgb = xgb.predict(X_df)[0]
print("XGBoost prediction:", pred_xgb)
print()

# Test lightgbm
lgb = joblib.load('models/lightgbm_gold.pkl')
pred_lgb = lgb.predict(X_df)[0]
print("LightGBM prediction:", pred_lgb)
print()

# Now test what the TARGET is - what does the model think the label is?
# Check: what happens if we feed Gold_Price=2000 instead?
print("="*50)
print("TEST: what if Gold_Price=2000?")
DEFAULTS2 = dict(DEFAULTS)
DEFAULTS2["Gold_Price"]           = 2000.0
DEFAULTS2["Gold_Silver_Ratio"]    = 2000 / 32.8
DEFAULTS2["Gold_Oil_Ratio"]       = 2000 / 75.0
DEFAULTS2["Gold_VIX_interaction"] = 2000 * 16.8
DEFAULTS2["Momentum_14d"]         = 2000 - (2000/1.027)
DEFAULTS2["Return_7d"]            = 0.016
DEFAULTS2["MACD"]                 = 2000*0.0022
DEFAULTS2["MACD_Hist"]            = 2000*0.0022*0.35
row2 = [DEFAULTS2[f] for f in FEATURES_32]
X2   = pd.DataFrame([row2], columns=FEATURES_32)
X2_pca = pca.transform(X2)
print("Lasso @2000:", lasso.predict(X2_pca)[0])
print("XGBoost @2000:", xgb.predict(X2)[0])

print("="*50)
print("TEST: what if Gold_Price=3000?")
DEFAULTS3 = dict(DEFAULTS)
DEFAULTS3["Gold_Price"]           = 3000.0
DEFAULTS3["Gold_Silver_Ratio"]    = 3000 / 32.8
DEFAULTS3["Gold_Oil_Ratio"]       = 3000 / 75.0
DEFAULTS3["Gold_VIX_interaction"] = 3000 * 16.8
DEFAULTS3["Momentum_14d"]         = 3000 - (3000/1.027)
DEFAULTS3["MACD"]                 = 3000*0.0022
DEFAULTS3["MACD_Hist"]            = 3000*0.0022*0.35
row3 = [DEFAULTS3[f] for f in FEATURES_32]
X3   = pd.DataFrame([row3], columns=FEATURES_32)
X3_pca = pca.transform(X3)
print("Lasso @3000:", lasso.predict(X3_pca)[0])
print("XGBoost @3000:", xgb.predict(X3)[0])
