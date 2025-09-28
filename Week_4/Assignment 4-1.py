import xml.etree.ElementTree as ET
import pandas as pd
import re
import math

# == 資料匯入 ==
xml_file = "temperature.xml"
n_lon = 67
n_lat = 120
lon0 = 120.00
lat0 = 21.88
res = 0.03
ns = {"cwa": "urn:cwa:gov:tw:cwacommon:0.1"}

# == 此處要抓XML檔案的Content(溫度觀測資料)內容 ==
tree = ET.parse(xml_file)
root = tree.getroot()

contents = root.findall(".//cwa:Content", ns)
if not contents:
    raise ValueError("找不到 <Content> 標籤（確認 namespace 與檔案）")

full_text = "".join([(c.text or "") for c in contents])

# == 抽出資料的科學記號(匹配XML檔-999.0E+00),並轉成 float(浮點數)==

num_tokens = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", full_text)

values = [float(tok) for tok in num_tokens]

print("抓到的數值筆數：", len(values))

# == 確認長度(ChatGPT建議) ==
expected = n_lon * n_lat
if len(values) != expected:
    # 若長度不合，ChatGPT建議印出前100個資料來debug.
    print(f"警告：數量不符合預期 {len(values)} != {expected}")
    print("前 200 個 token（供檢查）：")
    print(num_tokens[:200])
    raise ValueError(f"數量不對：抓到 {len(values)} 個數字，預期 {expected} 個")

# == 作業部份: 將資料轉換出 classification(label=0 or 1) & regression datasets (label=Value) ==
rows_class = []
rows_reg = []
for j in range(n_lat):
    for i in range(n_lon):
        val = values[j * n_lon + i]
        lon = lon0 + i * res
        lat = lat0 + j * res

        # 判定無效值-999.0的方式:用接近比較以避免浮點誤差(ChatGPT建議）
        if math.isclose(val, -999.0, abs_tol=1e-8):
            label = 0
        else:
            label = 1
        rows_class.append((lon, lat, label))

        if not math.isclose(val, -999.0, abs_tol=1e-8):
            rows_reg.append((lon, lat, val))
#== 轉 DataFrame 並輸出成csv檔 ==
df_class = pd.DataFrame(rows_class, columns=["lon", "lat", "label"])
df_reg = pd.DataFrame(rows_reg, columns=["lon", "lat", "value"])

df_class.to_csv("classification_dataset.csv", index=False)
df_reg.to_csv("regression_dataset.csv", index=False)

print("完成：已輸出 classification_dataset.csv 和 regression_dataset.csv")
