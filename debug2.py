# ضيف ده مؤقتاً في compare_all route قبل parse_form
# بعد السطر: X, missing_feats, oor_feats = parse_form(request.form, FEATURES_32)
# اضف:
# print("DEBUG Gold_Price from form:", request.form.get("Gold_Price"))
# print("DEBUG Return_7d from form:", request.form.get("Return_7d"))
# print("DEBUG Momentum_14d from form:", request.form.get("Momentum_14d"))
# print("DEBUG X[0][0] Gold_Price:", X[0][0])
# print("DEBUG X[0][18] Return_7d:", X[0][18])

# لكن الأسهل — شغلي debug_check2.py ده:
import joblib, numpy as np, pandas as pd

FEATURES_32 = [
    "Gold_Price","Silver","Platinum","Copper",
    "SP500","DXY","JPY_USD","GBP_USD","CNY_USD","US_2Y_Yield","TIP_ETF","Real_Rate_10Y",
    "PPI","Treasury_Spread","Unemployment","GDP_Growth","Industrial_Prod","Consumer_Conf",
    "Return_7d","Return_14d","Return_30d","MACD","MACD_Hist","Volatility_7d","Volatility_30d",
    "Momentum_14d","Gold_Silver_Ratio","Gold_Oil_Ratio","Gold_VIX_interaction",
    "Rolling_std_7","Price_diff","Month_cos",
]

# القيم اللي طلعت $1991.67 — ايه Gold_Price اللي ادت لـ 1991?
pca   = joblib.load('models/pca_transformer.pkl')
lasso = joblib.load('models/lasso_model.pkl')

# Binary search: find the Gold_Price that gives 1991.67
DEFAULTS_FIXED = {
    "Gold_Price":3320.00,"Silver":32.80,"Platinum":988.00,"Copper":4.62,
    "SP500":5610.00,"DXY":100.20,"JPY_USD":0.0065,"GBP_USD":1.3260,"CNY_USD":0.1380,
    "US_2Y_Yield":3.98,"TIP_ETF":109.20,"Real_Rate_10Y":1.92,"PPI":2.40,
    "Treasury_Spread":0.42,"Unemployment":4.20,"GDP_Growth":2.40,
    "Industrial_Prod":103.10,"Consumer_Conf":97.20,
    "Return_7d":0.016,"Return_14d":0.027,"Return_30d":0.048,
    "MACD":8.40,"MACD_Hist":2.90,"Volatility_7d":0.0082,"Volatility_30d":0.0095,
    "Momentum_14d":42.00,"Gold_Silver_Ratio":101.22,"Gold_Oil_Ratio":44.27,
    "Gold_VIX_interaction":55788.0,"Rolling_std_7":14.50,"Price_diff":6.20,
    "Month_cos":0.5,
}

# Try: what if Gold_VIX_interaction uses the DEFAULT (55788) but Gold_Price is different?
# Or: what if ALL features are defaults but Gold_Price=2000?
for test_gold in [2000, 2100, 1991, 3320, 3000]:
    d = dict(DEFAULTS_FIXED)
    d["Gold_Price"] = float(test_gold)
    # Keep derived features as-is (don't recompute)
    row = [d[f] for f in FEATURES_32]
    X = pd.DataFrame([row], columns=FEATURES_32)
    X_pca = pca.transform(X)
    pred = lasso.predict(X_pca)[0]
    print(f"Gold_Price={test_gold:5.0f} → Lasso={pred:,.2f}")

print()
# Now test: what if Gold_VIX_interaction is NOT updated (stays at 55788 for Gold=2000)?
print("=== Testing with stale Gold_VIX_interaction ===")
for test_gold in [2000, 2500, 3000, 3320]:
    d = dict(DEFAULTS_FIXED)
    d["Gold_Price"] = float(test_gold)
    d["Gold_Silver_Ratio"]    = test_gold / 32.8
    d["Gold_Oil_Ratio"]       = test_gold / 75.0
    d["Gold_VIX_interaction"] = test_gold * 16.8  # updated
    d["Momentum_14d"]         = test_gold - (test_gold/1.027)
    d["Return_7d"]            = 0.016
    d["Return_14d"]           = 0.027
    d["Return_30d"]           = 0.048
    row = [d[f] for f in FEATURES_32]
    X = pd.DataFrame([row], columns=FEATURES_32)
    X_pca = pca.transform(X)
    pred = lasso.predict(X_pca)[0]
    print(f"Gold_Price={test_gold:5.0f} Gold_VIX={d['Gold_VIX_interaction']:,.0f} → Lasso={pred:,.2f}")