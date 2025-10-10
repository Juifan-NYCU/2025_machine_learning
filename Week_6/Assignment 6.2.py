import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# === (1) 讀取資料並合成 ===
clf_data = pd.read_csv("classification_dataset.csv")  # lon, lat, label
reg_data = pd.read_csv("regression_dataset.csv")      # lon, lat, value

data = pd.merge(clf_data, reg_data, on=["lon", "lat"], how="inner")

# === (2) 建立 X 與 y ===
X = data[["lon", "lat"]]
y_clf = data["label"]       # 分類標籤 (0/1)
y_reg = data["value"]       # 溫度數值或 -999

# === (3) 建立簡單模型/分別訓練 ===
clf_model = RandomForestClassifier(random_state=42)
reg_model = RandomForestRegressor(random_state=42)

clf_model.fit(X, y_clf)
reg_model.fit(X, y_reg)

# === (4) 定義組合函數 h(x) ===
def h_function(X):
    C_pred = clf_model.predict(X)
    R_pred = reg_model.predict(X)
    h_pred = np.where(C_pred == 1, R_pred, -999)
    return h_pred

# === (5) 套用模型並驗證分段 ===
data["C_pred"] = clf_model.predict(X)
data["R_pred"] = reg_model.predict(X)
data["h_pred"] = h_function(X)

print(data.head(10))
print("\n檢查：")
print("C=1 時是否 h=R:", np.allclose(data.loc[data.C_pred==1,"h_pred"], data.loc[data.C_pred==1,"R_pred"]))
print("C=0 時是否 h=-999:", np.all(data.loc[data.C_pred==0,"h_pred"]==-999))

# === (6) 可視化結果 ===
plt.figure(figsize=(10,6))
sc = plt.scatter(data["lon"], data["lat"], c=data["h_pred"], cmap="coolwarm", s=60, edgecolor="k")
plt.colorbar(sc, label="h(x) value")
plt.title("Combined Model Output h(x)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# (ChatGPT建議)顯示前20筆預測結果
display(data[["lon","lat","C_pred","R_pred","h_pred"]].head(20))
