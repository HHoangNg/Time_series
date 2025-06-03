import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMAResults
file_path = "DOM_hourly.csv"
df = pd.read_csv(file_path, parse_dates=['Datetime'])
# plt.plot(df['Datetime'], df['DOM_MW'])
# plt.title("Tiêu thụ điện")
# plt.xlabel("Datetime")
# plt.ylabel("DOM_MW (MW)")
# plt.show()
series = df['DOM_MW']
def test_stationarity(timeseries):
    result = adfuller(timeseries.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    return result[1]

# Kiểm tra chuỗi gốc
print("🔍 Kiểm tra chuỗi gốc:")
pval = test_stationarity(series)

# Tiến hành biến đổi nếu cần
if pval > 0.05:
    series_diff = series.diff().dropna()
    print("\n🔁 Kiểm tra chuỗi sau sai phân:")
    pval = test_stationarity(series_diff)
    if pval > 0.05:
        series_log_diff = np.log(series).diff().dropna()
        print("\n🔁 Kiểm tra chuỗi sau log + sai phân:")
        pval = test_stationarity(series_log_diff)
        final_series = series_log_diff
        transform_used = "log + diff"
    else:
        final_series = series_diff
        transform_used = "diff"
else:
    final_series = series
    transform_used = "original"

print(f"\n✅ Chuỗi đã dừng với phương pháp: {transform_used}")

# Chọn phần dữ liệu để huấn luyện (ví dụ lấy 5000 điểm cuối)
final_series = final_series[-5000:]

# Fit mô hình ARIMA
model = ARIMA(final_series, order=(24, 0, 24))
model_fit = model.fit()

# Dự đoán 100 bước tiếp theo
forecast = model_fit.predict(start=len(final_series), end=len(final_series)+100)

# Vẽ kết quả
plt.figure(figsize=(12, 5))
plt.plot(final_series, label='Dữ liệu đầu vào')
plt.plot(forecast, label='Dự báo ARIMA', color='purple')
plt.title(f"Dự báo ARIMA với chuỗi đã dừng ({transform_used})")
plt.xlabel("Thời gian")
plt.ylabel("Giá trị")
plt.legend()
plt.tight_layout()
plt.show()